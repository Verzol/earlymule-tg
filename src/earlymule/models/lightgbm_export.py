"""Daily LightGBM trained from a Kaggle export, with feature families left out (feature ablation, EMTG-70).

Reads an export folder (python -m earlymule.data.export_events): rows_<split>, features_<split> (same row
order), nodes (initial_deposit) and manifest.json. Trains with the daily settings of configs/daily.yaml
(regular accounts, 1% of train negatives per seed, early stopping on validation PR-AUC only) on the
features the ablation keeps, as listed in configs/lightgbm_ablation.yaml.
Writes scores_val.parquet, scores_test.parquet (acct_id, snapshot, score), importance.csv, model.txt and
run_meta.json to --out; test rows carry no labels, so evaluation runs locally with
python -m earlymule.evaluation.import_scores --from <out>.
Run: python -m earlymule.models.lightgbm_export --data <export folder> --ablation no_churn [--seed N] [--out DIR]
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["seed"] = cfg["seed"] if args.seed is None else args.seed
    set_seed(cfg["seed"])
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())

    # 1. Train rows: labelled regular account-days, every positive and 1% of negatives for this seed
    deposit = pd.read_parquet(data / "nodes.parquet").set_index("acct_id").initial_deposit
    train_label = pd.read_parquet(data / "rows_train.parquet", columns=["label"]).label
    mask = train_mask(train_label, cfg["train_negative_rate"], np.random.default_rng(cfg["seed"]))
    frames = {"train": load_split(data, "train", deposit, mask),
              "val": load_split(data, "val", deposit), "test": load_split(data, "test", deposit)}

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
    run_name = f"{cfg['run_name']}_{args.ablation}" + ("" if dataset == load_paths().dataset else f"_{dataset}")
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
            | {"ablation": args.ablation, "dropped": cfg["ablations"][args.ablation]},
            "device": "cpu", "lightgbm_version": lgb.__version__, "train_rows": int(mask.sum()),
            "best_iteration": int(model.best_iteration_), "best_val_pr_auc": round(model.best_score_["valid_0"]["average_precision"], 4),
            "epochs": None, "seconds": round(time.time() - start, 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ["run_name", "seed", "best_iteration", "best_val_pr_auc", "seconds"]}
                     | {"features": len(features), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
