"""Daily scoring for early detection: score regular accounts every day, alert the top K, time the detections.

A mule is positive while active (onset <= t <= completion). Train: daily snapshots of the train
split with subsampled negatives. Validation and test: every day, regular accounts only.
Run: python -m earlymule.models.daily_scoring   (appends one row to experiments/experiments.csv)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from earlymule.data.splits import active_labels, chronological_splits
from earlymule.evaluation.early_detection import early_detection
from earlymule.evaluation.metrics import evaluate
from earlymule.experiment_log import log_run, set_seed
from earlymule.models.tabular import fit, load_config, load_inputs, model_features, build_split


def main() -> None:
    """Train on daily train snapshots, score validation and test daily, log detection timing."""
    start = time.time()
    cfg = load_config("daily.yaml")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # 1. Inputs and labels: positive while the mule case is running
    inputs = load_inputs()
    mules = inputs["mules"]
    onset, completion = mules.onset_day.astype(float), mules.completion_day.astype(float)
    accts = inputs["accounts"].index
    label_fn = lambda t: active_labels(onset, completion, accts, t)  # noqa: E731
    regular = lambda f: f.n_tx_hist.ge(cfg["regular_min_tx"])  # noqa: E731

    # 2. Snapshots: train every `train_every_days`, validation and test every day on regular accounts
    splits = {s.name: s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    days = {"train": range(cfg["min_history_days"], splits["train"].last_day + 1, cfg["train_every_days"]),
            "val": range(splits["val"].first_day, splits["val"].last_day + 1),
            "test": range(splits["test"].first_day, splits["test"].last_day + 1)}
    data = {"train": build_split(inputs, label_fn, days["train"], cfg, cfg["train_negative_rate"], rng),
            "val": build_split(inputs, label_fn, days["val"], cfg, keep=regular),
            "test": build_split(inputs, label_fn, days["test"], cfg, keep=regular)}
    features = model_features(data["train"], cfg)

    # 3. Train, then rank and time the detections under each daily alert budget
    model = fit(data["train"], data["val"], features, cfg)
    table = mules.reset_index()[["acct_id", "onset_day", "completion_day", "typology"]]
    metrics = {"best_iteration": int(model.best_iteration_), "features": len(features)}
    for name in ["val", "test"]:
        frame = data[name].assign(score=model.predict_proba(data[name][features])[:, 1])
        metrics[name] = {"ranking": evaluate(frame, cfg["recall_k"]),
                         "early_detection": [early_detection(frame, table, k) for k in cfg["alert_budgets"]]}
    metrics["seconds"] = round(time.time() - start, 1)

    # 4. Log and report
    config = {**cfg, "splits": {n: [s.first_day, s.last_day] for n, s in splits.items()},
              "train_rows": len(data["train"]), "feature_names": features}
    log_run(cfg["run_name"], config=config, seed=cfg["seed"], metrics=metrics)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
