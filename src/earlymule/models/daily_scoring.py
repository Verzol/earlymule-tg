"""Daily scoring for early detection: score regular accounts every day, alert the top K, time the detections.

A mule is positive while active (onset <= t <= completion). Train: daily snapshots of the train
split with subsampled negatives. Validation and test: every day, regular accounts; test also on all accounts.
Alerts: top K new accounts per day with a cooldown, plus the research plan's score threshold chosen on validation.
Run: python -m earlymule.models.daily_scoring [--seed N] [--config daily.yaml]
Writes scores, per-mule outcomes and the model to experiments/runs/<run_name>/<commit>/seed<N>/ and appends
one row to experiments/experiments.csv.
"""

from __future__ import annotations

import json
import time

import numpy as np

from earlymule.data.splits import active_labels, chronological_splits
from earlymule.evaluation.daily import evaluate_daily, mule_table
from earlymule.evaluation.early_detection import early_detection
from earlymule.evaluation.metrics import evaluate
from earlymule.experiment_log import log_run, run_dir, set_seed
from earlymule.models.tabular import build_split, fit, config_from_cli, load_inputs, model_features


def main() -> None:
    """Train on daily train snapshots, score validation and test daily, log detection timing."""
    start = time.time()
    cfg = config_from_cli("daily.yaml")
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
    # Train on the population that is scored, unless configured otherwise
    train_keep = regular if cfg["train_population"] == "regular" else None
    data = {"train": build_split(inputs, label_fn, days["train"], cfg, cfg["train_negative_rate"], rng, keep=train_keep),
            "val": build_split(inputs, label_fn, days["val"], cfg, keep=regular, keep_unlabeled=True),
            "test": build_split(inputs, label_fn, days["test"], cfg, keep=regular, keep_unlabeled=True)}
    features = model_features(data["train"], cfg)

    # 3. Train, then score validation and test and evaluate them as every model is (evaluation.daily)
    model = fit(data["train"], data["val"], features, cfg)
    out_dir = run_dir(cfg["run_name"], cfg["seed"])
    model.booster_.save_model(str(out_dir / "model.txt"))
    # Log-odds, not probabilities: probabilities saturate at 1.0 and tie at the top of the ranking
    scored = {name: data[name].assign(score=model.predict(data[name][features], raw_score=True))
              for name in ["val", "test"]}
    metrics = {"best_iteration": int(model.best_iteration_), "features": len(features),
               **evaluate_daily(scored, inputs, cfg, out_dir)}
    table, cooldown = mule_table(mules), cfg["alert_cooldown_days"]

    # 4. All accounts on test, reported next to the regular population; scored day by day to bound memory
    score_day = lambda f: f[["snapshot", "label", "typology"]].assign(  # noqa: E731
        score=model.predict(f[features], raw_score=True))
    everyone = build_split(inputs, label_fn, days["test"], cfg, keep_unlabeled=True, reduce=score_day)
    metrics["test_all_accounts"] = {
        "ranking": evaluate(everyone, cfg["recall_k"]),
        "early_detection": [early_detection(everyone, table, k, cooldown, inputs["other_members"])
                            for k in cfg["alert_budgets"]]}
    metrics["seconds"] = round(time.time() - start, 1)

    # 5. Log and report
    config = {**cfg, "splits": {n: [s.first_day, s.last_day] for n, s in splits.items()},
              "train_rows": len(data["train"]), "feature_names": features}
    log_run(cfg["run_name"], config=config, seed=cfg["seed"], metrics=metrics)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
