"""Daily LightGBM trained from a Kaggle export, with feature families left out (feature ablation, EMTG-70).

Reads an export folder (python -m earlymule.data.export_events): rows_<split>, features_<split> (same row
order), nodes (initial_deposit) and manifest.json. Trains with the daily settings of configs/daily.yaml
(regular accounts, 1% of train negatives per seed, early stopping on validation PR-AUC only) on the
features the ablation keeps, as listed in configs/lightgbm_ablation.yaml.
Writes scores_val.parquet, scores_test.parquet (acct_id, snapshot, score), importance.csv, model.txt and
run_meta.json to --out; test rows carry no labels, so evaluation runs locally with
python -m earlymule.evaluation.import_scores --from <out>.
With --label-fraction F and --label-cases <csv> (python -m earlymule.data.label_fraction), only the training mules
of a seeded share F of cases keep their label; the others become negatives or, with --unlabeled drop, leave training.
Run: python -m earlymule.models.lightgbm_export --data <export folder> --ablation no_churn [--seed N] [--out DIR]
     [--label-fraction F --label-cases CSV [--unlabeled negative|drop]]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.data.label_fraction import kept_mules, leave_out, relabel
from earlymule.experiment_log import git_state, run_dir, set_seed
from earlymule.models.tabular import fit, load_config

KEYS = ["acct_id", "snapshot"]


def feature_families(columns: list[str], families: dict[str, list[str]]) -> dict[str, str]:
    """Map every feature column to its family; raise if a column matches no family or more than one."""
    out = {}
    for col in columns:
        hits = [name for name, patterns in families.items() if any(re.search(p, col) for p in patterns)]
        if len(hits) != 1:
            raise ValueError(f"feature {col!r} matches families {hits}; it must match exactly one")
        out[col] = hits[0]
    return out


def kept_features(columns: list[str], cfg: dict, ablation: str) -> list[str]:
    """Feature columns the ablation keeps, in export order.

    Args:
        columns: model feature columns of the export, initial_deposit included.
        cfg: lightgbm_ablation.yaml with families, windows_patterns and ablations.
        ablation: key of cfg["ablations"].
    """
    spec = cfg["ablations"][ablation]
    family = feature_families(columns, cfg["families"])
    drop_family = set(spec.get("families", []))
    drop_patterns = [p for w in spec.get("windows", []) for p in cfg["windows_patterns"][w]]
    kept = [c for c in columns if family[c] not in drop_family and not any(re.search(p, c) for p in drop_patterns)]
    if not kept or (spec and len(kept) == len(columns)):
        raise ValueError(f"ablation {ablation!r} keeps {len(kept)} of {len(columns)} features")
    return kept


def train_mask(label: pd.Series, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Keep every positive and a `rate` share of negatives, as build_split does for the daily train split."""
    return (label.eq(1) | pd.Series(rng.random(len(label)) < rate, index=label.index)).to_numpy()


def load_split(data: Path, split: str, deposit: pd.Series, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Rows of one split with their features and initial_deposit; rows and features are paired by position.

    Args:
        mask: optional row selection, applied before the deposit join to keep memory small.
    """
    # 1. Rows (labels where the export has them) and features, checked to be in the same order
    rows = pd.read_parquet(data / f"rows_{split}.parquet").drop(columns=["node"], errors="ignore")
    feats = pd.read_parquet(data / f"features_{split}.parquet")
    if len(rows) != len(feats) or not (rows[KEYS].to_numpy() == feats[KEYS].to_numpy()).all():
        raise ValueError(f"rows_{split} and features_{split} are not in the same order")
    if mask is not None:
        rows, feats = rows[mask], feats[mask]
    frame = pd.concat([rows.reset_index(drop=True), feats.drop(columns=KEYS).reset_index(drop=True)], axis=1)
    # 2. Static input last, as build_split appends it after the behaviour features
    frame["initial_deposit"] = deposit.reindex(frame.acct_id).to_numpy(dtype="float32")
    return frame


def main() -> None:
    """Train one (ablation, seed) run from an export and write its scores and run record."""
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="export folder (manifest.json, rows_*, features_*, nodes)")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--config", default="lightgbm_ablation.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", help="output folder; default experiments/runs/<run_name>/<commit>/seed<N>")
    parser.add_argument("--label-fraction", type=float, help="share of training cases that keep their mule labels")
    parser.add_argument("--label-cases", help="case map of the training mules (acct_id, alert_id)")
    parser.add_argument("--unlabeled", default="negative", choices=["negative", "drop"])
    parser.add_argument("--shuffle-labels", action="store_true", help="permute the training labels (pipeline check)")
    parser.add_argument("--leave-out-typology", help="drop every training row of the mules of this typology")
    args = parser.parse_args()
    if (args.label_fraction is not None or args.leave_out_typology) and args.label_cases is None:
        parser.error("--label-fraction and --leave-out-typology need --label-cases")
    cfg = load_config(args.config)
    cfg["seed"] = cfg["seed"] if args.seed is None else args.seed
    set_seed(cfg["seed"])
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())

    # 1. Train rows: labelled regular account-days, every positive and 1% of negatives for this seed
    deposit = pd.read_parquet(data / "nodes.parquet").set_index("acct_id").initial_deposit
    train_rows = pd.read_parquet(data / "rows_train.parquet", columns=["acct_id", "label"])
    train_label, n_mules = train_rows.label, int(train_rows.acct_id[train_rows.label == 1].nunique())
    if args.label_fraction is not None:
        kept = kept_mules(pd.read_csv(args.label_cases), args.label_fraction, cfg["seed"])
        train_label = pd.Series(relabel(train_label.to_numpy(), train_rows.acct_id.to_numpy(), kept, args.unlabeled))
        n_mules = len(kept)
    if args.leave_out_typology:
        # Whole accounts leave training, pre-onset rows too, so the model never sees this typology's mules
        gone = leave_out(pd.read_csv(args.label_cases), args.leave_out_typology)
        train_label = train_label.where(~train_rows.acct_id.isin(gone).to_numpy())
        n_mules -= len(gone)
    if args.shuffle_labels:
        # Own random stream, so the negative sample below stays the one of the seed's normal run
        train_label = pd.Series(np.random.default_rng([cfg["seed"], 1]).permutation(train_label.to_numpy()))
    # Same random draw as the full-label run of the seed, so the sample of true negatives is the same;
    # dropped mule rows (NaN) are never kept
    mask = train_mask(train_label, cfg["train_negative_rate"], np.random.default_rng(cfg["seed"])) & train_label.notna().to_numpy()
    frames = {"train": load_split(data, "train", deposit, mask),
              "val": load_split(data, "val", deposit), "test": load_split(data, "test", deposit)}
    frames["train"]["label"] = train_label[mask].to_numpy()

    # 2. Features the ablation keeps; the export already leaves out n_tx_hist and days_since_first_tx
    columns = [c for c in frames["train"].columns if c not in (*KEYS, "label")]
    features = kept_features(columns, cfg, args.ablation)

    # 3. Train with early stopping on validation PR-AUC, then score every validation and test row
    model = fit(frames["train"], frames["val"], features, cfg)
    scores = {s: frames[s][KEYS + (["label"] if s == "val" else [])].assign(
        score=model.predict(frames[s][features], raw_score=True)) for s in ["val", "test"]}

    # 4. Scores, importance and model file; evaluation runs locally on these files (import_scores)
    commit, dirty = git_state()
    dataset = manifest["dataset"]
    suffix = "" if args.label_fraction is None else f"_lf{args.label_fraction:g}" + ("_drop" if args.unlabeled == "drop" else "")
    suffix += ("_shuffled" if args.shuffle_labels else "") + (f"_wo_{args.leave_out_typology}" if args.leave_out_typology else "")
    run_name = f"{cfg['run_name']}_{args.ablation}{suffix}" + ("" if dataset == load_paths().dataset else f"_{dataset}")
    out = Path(args.out) if args.out else run_dir(run_name, cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    for s in ["val", "test"]:
        scores[s].to_parquet(out / f"scores_{s}.parquet", index=False)
    booster = model.booster_
    pd.DataFrame({"feature": features, "gain": booster.feature_importance("gain"),
                  "split": booster.feature_importance("split")}).to_csv(out / "importance.csv", index=False)
    booster.save_model(str(out / "model.txt"))
    meta = {"run_name": run_name, "dataset": dataset, "export_commit": manifest["commit"], "commit": commit,
            "dirty": dirty, "seed": cfg["seed"], "ablation": args.ablation, "features": features,
            "config": {k: v for k, v in cfg.items() if k not in ("families", "windows_patterns", "ablations")}
            | {"ablation": args.ablation, "dropped": cfg["ablations"][args.ablation],
               "label_fraction": args.label_fraction or 1.0, "unlabeled": args.unlabeled, "labelled_mules": n_mules,
               "shuffle_labels": args.shuffle_labels, "left_out_typology": args.leave_out_typology},
            "device": "cpu", "lightgbm_version": lgb.__version__, "train_rows": int(mask.sum()),
            "best_iteration": int(model.best_iteration_), "best_val_pr_auc": round(model.best_score_["valid_0"]["average_precision"], 4),
            "epochs": None, "seconds": round(time.time() - start, 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ["run_name", "seed", "best_iteration", "best_val_pr_auc", "seconds"]}
                     | {"features": len(features), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
