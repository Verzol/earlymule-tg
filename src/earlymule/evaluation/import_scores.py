"""Evaluate daily scores of a model trained elsewhere (e.g. TGN on Kaggle) exactly as LightGBM is evaluated.

The folder holds run_meta.json and scores_{val,test}.parquet (acct_id, snapshot, score) written by the model run.
Labels and typologies are attached here from the local ground truth, so test labels never leave this machine.
Run: python -m earlymule.evaluation.import_scores --from <folder> [--run NAME] [--no-log]
Writes the run folder experiments/runs/<run>/<commit of the model run>/seed<N>/ and appends one row to
experiments/experiments.csv with that commit (--no-log: evaluate and print only, for debugging).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.evaluation.daily import attach_labels, evaluate_daily
from earlymule.experiment_log import log_run
from earlymule.models.tabular import load_config, load_inputs


def check_val_labels(scores: pd.DataFrame, labelled: pd.DataFrame) -> None:
    """Raise if the labels the model saw on validation differ from the local ones (stale export)."""
    if "label" not in scores:
        return
    a, b = scores.label.to_numpy(), labelled.label.to_numpy()
    if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]):
        raise ValueError("validation labels of the run differ from the local ground truth; re-export the data")


def main() -> None:
    """Attach labels, evaluate, save the run folder and log the run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", required=True, help="folder with run_meta.json and scores")
    parser.add_argument("--run", help="run name in the log; default run_meta's run_name")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    source = Path(args.source)
    meta = json.loads((source / "run_meta.json").read_text())

    # 1. Ground truth of the run's dataset, and the alert settings of the daily evaluation
    paths = load_paths()
    inputs = load_inputs(paths.with_dataset(meta["dataset"]))
    cfg = {**load_config("daily.yaml"), "seed": meta["seed"]}

    # 2. Labels from the local ground truth; validation labels must match what the model was tuned on
    scored = {}
    for name in ["val", "test"]:
        raw = pd.read_parquet(source / f"scores_{name}.parquet")
        scored[name] = attach_labels(raw[["acct_id", "snapshot", "score"]], inputs["mules"])
        if name == "val":
            check_val_labels(raw, scored[name])

    # 3. Evaluate into the run folder named after the model run's commit
    run_name = args.run or meta["run_name"]
    folder = meta["commit"] + ("-dirty" if meta["dirty"] else "")
    out_dir = paths.experiments / "runs" / ("debug" if args.no_log else "") / run_name / folder / f"seed{meta['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "run_meta.json", out_dir / "run_meta.json")
    metrics = {k: meta[k] for k in ["best_epoch", "best_val_pr_auc", "params", "seconds", "device"] if k in meta}
    metrics.update(evaluate_daily(scored, inputs, cfg, out_dir))

    # 4. Log with the model run's commit, not this machine's
    if not args.no_log:
        config = {**meta["config"], "dataset": meta["dataset"], "export_commit": meta["export_commit"],
                  "epochs": meta["epochs"], "evaluation": "daily.yaml"}
        log_run(run_name, config=config, seed=meta["seed"], metrics=metrics, commit=meta["commit"], dirty=meta["dirty"])
    k = next(x for x in metrics["test"]["early_detection"] if x["k_per_day"] == cfg["headline_k"])
    print(json.dumps({"run": run_name, "seed": meta["seed"], "out": str(out_dir),
                      "val_pr_auc": round(metrics["val"]["ranking"]["pr_auc"], 4),
                      "test_pr_auc": round(metrics["test"]["ranking"]["pr_auc"], 4),
                      "detected_before_forwarding": k["detected_before_forwarding"]}, indent=1))


if __name__ == "__main__":
    main()
