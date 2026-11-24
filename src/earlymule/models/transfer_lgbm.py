"""LightGBM at payment initiation (second task): score every outgoing transfer, hold the riskiest, measure mules caught.

Three steps, so training can run on Kaggle CPU:
  export (local): one folder with the transfer features (python -m earlymule.features.transfer_time, both
    cuts), the payee risk score and risk propagation from daily account scores (out-of-fold on train days),
    the transfer labels (illicit; illicit outflow of a mule) and each mule's first illicit outflow.
  train (local or Kaggle): one (variant, seed) from that folder; early stopping on validation PR-AUC; holds of
    H = 10, 20, 50 new accounts a day with the threshold chosen on validation. Writes scores, per-mule hold
    tables, model, importance, metrics.json and config.json to --out, and logs the run when --log is given.
  import (local): log runs trained elsewhere from their metrics.json and config.json.
Run: python -m earlymule.models.transfer_lgbm export
     python -m earlymule.models.transfer_lgbm train --data DIR --variant full [--seed N] [--out DIR] [--log]
     python -m earlymule.models.transfer_lgbm import --from DIR
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from earlymule.config import load_paths, repo_root
from earlymule.data import amlsim
from earlymule.data.splits import chronological_splits
from earlymule.evaluation.transfer_holds import first_outflows, hold_metrics, hold_table, holds, threshold_for_budget
from earlymule.experiment_log import git_state, log_run, run_dir, set_seed
from earlymule.features.point_in_time import load_transactions
from earlymule.features.transfer_time import KEYS, risk_features
from earlymule.models.tabular import fit, load_config


def variant_features(columns: list[str], cfg: dict, variant: str) -> list[str]:
    """Feature columns of a variant: its groups, minus its drops and the configured exclusions.

    Raises if a column matches no group or more than one, so a new feature cannot slip in unassigned.
    """
    spec = cfg["variants"][variant]
    group = {}
    for c in columns:
        hits = [g for g, pats in cfg["groups"].items() if any(re.search(p, c) for p in pats)]
        if len(hits) != 1:
            raise ValueError(f"feature {c!r} matches groups {hits}; it must match exactly one")
        group[c] = hits[0]
    drop = list(spec.get("drop_patterns", [])) + (cfg[f"{spec['drop']}_drop"] if "drop" in spec else [])
    return [c for c in columns if group[c] in spec["groups"] and c not in cfg["exclude_features"]
            and c not in spec.get("drop_columns", []) and not any(re.search(p, c) for p in drop)]


def daily_scores(cfg: dict) -> pd.DataFrame:
    """Daily account scores: out-of-fold on train days, the full train model on validation and test days."""
    root = repo_root()
    oof = pd.concat(pd.read_parquet(p) for p in sorted((root / cfg["oof_scores"]).glob("oof_train_fold*.parquet")))
    full = [pd.read_parquet(root / cfg["full_scores"] / f"scores_{s}.parquet", columns=["acct_id", "snapshot", "score"])
            for s in ("val", "test")]
    return pd.concat([oof, *full], ignore_index=True)


def export(cfg: dict) -> Path:
    """Write the training folder of the configured dataset; returns its path."""
    paths = load_paths()
    base = amlsim.read_base_date(paths.amlsim_conf)
    out = paths.processed / f"transfer_task_{paths.dataset}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Features of both cuts, as built by transfer_time
    for suffix in ("", "_conservative"):
        pd.read_parquet(paths.processed / f"transfer_features_{paths.dataset}{suffix}.parquet").to_parquet(
            out / f"features{suffix}.parquet", index=False)
    keys = pd.read_parquet(out / "features.parquet", columns=KEYS)

    # 2. Risk columns depend only on day d - 1, so one file serves both cuts
    tx = load_transactions(paths.dataset_dir, base)
    risk_features(keys, tx, daily_scores(cfg), cfg["hot_share"]).assign(seq=keys.seq.to_numpy()).to_parquet(
        out / "risk.parquet", index=False)

    # 3. Labels and mule outcomes from the case files (evaluation only; never joined as features)
    illicit = amlsim.load_alert_transactions(paths.dataset_dir, base)
    gt = pd.read_parquet(paths.dataset_processed / "ground_truth.parquet")
    mules = gt[gt.is_mule_candidate]
    labels = keys[["seq"]].assign(illicit=keys.seq.isin(illicit.tran_id).astype("int8"))
    labels["mule_illicit"] = (labels.illicit.astype(bool) & keys.orig_acct.isin(mules.acct_id)).astype("int8")
    labels.to_parquet(out / "labels.parquet", index=False)
    first_outflows(illicit, mules).to_parquet(out / "first_outflows.parquet", index=False)
    commit, dirty = git_state()
    (out / "manifest.json").write_text(json.dumps({"dataset": paths.dataset, "commit": commit, "dirty": dirty,
                                                   "rows": len(keys)}, indent=1))
    print(f"{len(keys):,} transfers -> {out}")
    return out


def load_rows(data: Path, cfg: dict, conservative: bool) -> pd.DataFrame:
    """Features of one cut with risk columns, labels and split name, from an export folder."""
    rows = pd.read_parquet(data / f"features{'_conservative' if conservative else ''}.parquet")
    rows = rows.merge(pd.read_parquet(data / "risk.parquet"), on="seq", how="left")
    rows = rows.merge(pd.read_parquet(data / "labels.parquet"), on="seq", how="left")
    splits = chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))
    rows["split"] = pd.cut(rows.day, [s.first_day - 0.5 for s in splits] + [splits[-1].last_day + 0.5],
                           labels=[s.name for s in splits]).astype(str)
    return rows


def train(data: Path, cfg: dict, variant: str, out: Path) -> dict:
    """Train one (variant, seed), evaluate holds on validation and test, write everything to out."""
    start = time.time()
    set_seed(cfg["seed"])
    spec = cfg["variants"][variant]

    # 1. Rows, label of this variant, features; every negative is kept (no sampling noise)
    rows = load_rows(data, cfg, spec.get("conservative", False))
    rows["label"] = rows.illicit if spec.get("label") == "any_illicit" else rows.mule_illicit
    meta_cols = KEYS + ["illicit", "mule_illicit", "split", "label"]
    features = variant_features([c for c in rows.columns if c not in meta_cols], cfg, variant)
    parts = {s: rows[rows.split.eq(s)].reset_index(drop=True) for s in ("train", "val", "test")}
    del rows

    # 2. Train with early stopping on validation PR-AUC, score validation and test
    model = fit(parts["train"], parts["val"], features, cfg)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        parts[s]["score"] = model.predict(parts[s][features], raw_score=True)
        parts[s][KEYS + ["score"]].to_parquet(out / f"scores_{s}.parquet", index=False)

    # 3. Holds per budget: threshold on validation, applied unchanged to test
    first_out = pd.read_parquet(data / "first_outflows.parquet")
    metrics = {"best_iteration": int(model.best_iteration_), "features": len(features),
               "train_rows": len(parts["train"]), "train_positives": int(parts["train"].label.sum())}
    for s in ("val", "test"):
        # PR-AUC against the mule label in every variant, so the wide-label variant is comparable
        metrics[f"{s}_pr_auc"] = round(average_precision_score(parts[s].mule_illicit, parts[s].score), 4)
    for h in cfg["hold_budgets"]:
        theta = threshold_for_budget(parts["val"], h, cfg["cooldown_days"])
        metrics[f"h{h}"] = {"theta": round(theta, 4)}
        for s in ("val", "test"):
            held = holds(parts[s], theta, cfg["cooldown_days"])
            table = hold_table(held, first_out, pd.Index(parts[s].seq))
            table.to_parquet(out / f"holds_{s}_h{h}.parquet", index=False)
            # Hold precision is counted against mule outflows in every variant
            metrics[f"h{h}"][s] = hold_metrics(parts[s].assign(label=parts[s].mule_illicit), held, table,
                                               cfg["bootstrap_samples"], cfg["seed"])
    metrics["seconds"] = round(time.time() - start, 1)

    # 4. Model, importance, run record
    model.booster_.save_model(str(out / "model.txt"))
    pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(out / "importance.csv", index=False)
    commit, dirty = git_state()
    manifest = json.loads((data / "manifest.json").read_text())
    config = {k: v for k, v in cfg.items() if k not in ("variants", "groups")} | {
        "variant": variant, "spec": spec, "feature_names": features, "run_name": f"{cfg['run_name']}_{variant}",
        "commit": commit, "dirty": dirty, "export_commit": manifest["commit"], "lightgbm_version": lgb.__version__}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "config.json").write_text(json.dumps(config, indent=1))
    head = metrics[f"h{cfg['headline_h']}"]["test"]
    print(json.dumps({"variant": variant, "seed": cfg["seed"], "best_iteration": metrics["best_iteration"],
                      "val_pr_auc": metrics["val_pr_auc"], "test_pr_auc": metrics["test_pr_auc"],
                      "caught_all": head["caught_all"], "caught_can_precede": head["caught_can_precede"],
                      "holds_per_day": head["holds_per_day"], "seconds": metrics["seconds"]}), flush=True)
    return config | {"metrics": metrics}


def import_runs(folder: Path) -> int:
    """Log every run folder under `folder` (metrics.json + config.json) with the commit it ran at."""
    logged = 0
    for m in sorted(folder.rglob("metrics.json")):
        config = json.loads((m.parent / "config.json").read_text())
        log_run(config["run_name"], config=config, seed=config["seed"], metrics=json.loads(m.read_text()),
                commit=config["commit"], dirty=config["dirty"], notes="trained on Kaggle CPU")
        logged += 1
    print(f"logged {logged} runs from {folder}")
    return logged


def main() -> None:
    """Dispatch export, train and import."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("export")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--variant", default="full")
    t.add_argument("--seed", type=int)
    t.add_argument("--out", help="default experiments/runs/transfer_lgbm_<variant>/<commit>/seed<N>")
    t.add_argument("--log", action="store_true", help="append the run to experiments.csv")
    i = sub.add_parser("import")
    i.add_argument("--from", dest="folder", required=True)
    parser.add_argument("--config", default="transfer_scoring.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step == "export":
        export(cfg)
    elif args.step == "import":
        import_runs(Path(args.folder))
    else:
        cfg["seed"] = cfg["seed"] if args.seed is None else args.seed
        out = Path(args.out) if args.out else run_dir(f"{cfg['run_name']}_{args.variant}", cfg["seed"])
        run = train(Path(args.data), cfg, args.variant, out)
        if args.log:
            log_run(run["run_name"], config={k: v for k, v in run.items() if k != "metrics"}, seed=cfg["seed"],
                    metrics=run["metrics"])


if __name__ == "__main__":
    main()
