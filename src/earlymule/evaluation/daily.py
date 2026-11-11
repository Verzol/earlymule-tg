"""Daily evaluation shared by every model: ranking, alert-budget timing, the threshold rule and saved outputs.

Input: scored validation and test rows of regular accounts with acct_id, snapshot, label, typology, score.
Used by models.daily_scoring (LightGBM, trained here) and evaluation.import_scores (models trained elsewhere).
Writes scores_{val,test}.parquet and detection_test.parquet (per-mule outcomes for paired comparisons).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.evaluation.early_detection import (detection_table, early_detection, threshold_alerts,
                                                  threshold_on_validation)
from earlymule.evaluation.metrics import evaluate, threshold_classification
from earlymule.evaluation.uncertainty import bootstrap_detection, bootstrap_ranking

SCORE_COLUMNS = ["acct_id", "snapshot", "label", "typology", "score"]


def mule_table(mules: pd.DataFrame) -> pd.DataFrame:
    """Per-mule timing columns the detection metrics need (mules: load_inputs()["mules"])."""
    return mules.reset_index()[["acct_id", "alert_id", "onset_day", "completion_day", "first_illicit_outflow_day",
                                "typology"]]


def attach_labels(scores: pd.DataFrame, mules: pd.DataFrame) -> pd.DataFrame:
    """Add the daily label and typology to scored rows, as splits.active_labels defines them.

    1 while a mule is active (onset <= t <= completion), NaN after completion, 0 otherwise; typology on positives.
    """
    onset, completion = scores.acct_id.map(mules.onset_day.astype(float)), scores.acct_id.map(mules.completion_day.astype(float))
    t = scores.snapshot
    label = pd.Series(0.0, index=scores.index)
    label[onset.le(t) & completion.ge(t)] = 1.0
    label[completion.lt(t)] = np.nan
    typology = scores.acct_id.map(mules.typology).where(label.eq(1))
    return scores.assign(label=label, typology=typology)


def evaluate_daily(scored: dict[str, pd.DataFrame], inputs: dict, cfg: dict, out_dir: Path) -> dict:
    """Validation and test metrics of one run; saves its scores and per-mule outcomes to out_dir.

    Args:
        scored: "val" and "test" frames with SCORE_COLUMNS.
        inputs: load_inputs output (mules, other_members).
        cfg: alert_budgets, alert_cooldown_days, recall_k, bootstrap_samples, seed.
    """
    mules, others = inputs["mules"], inputs["other_members"]
    table, cooldown = mule_table(mules), cfg["alert_cooldown_days"]
    n, seed = cfg["bootstrap_samples"], cfg["seed"]

    # 1. Ranking and timing under each daily alert budget
    metrics = {}
    for name in ["val", "test"]:
        frame = scored[name]
        # Kept for error analysis and paired comparisons; evaluation-only file, never a model input
        frame[SCORE_COLUMNS].to_parquet(out_dir / f"scores_{name}.parquet")
        metrics[name] = {"ranking": evaluate(frame, cfg["recall_k"]),
                         "early_detection": [early_detection(frame, table, k, cooldown, others)
                                             for k in cfg["alert_budgets"]]}

    # 2. Case-bootstrap intervals on test; per-mule outcomes saved for paired comparisons between models
    test = scored["test"]
    metrics["test"]["ranking"].update(bootstrap_ranking(test, mules.alert_id, cfg["recall_k"], n, seed))
    tables = []
    for k, row in zip(cfg["alert_budgets"], metrics["test"]["early_detection"]):
        tables.append(detection_table(test, table, k, cooldown).assign(k=k))
        row.update(bootstrap_detection(tables[-1], n, seed))
    pd.concat(tables).to_parquet(out_dir / "detection_test.parquet")

    # 3. Research plan's alert rule: one score threshold chosen on validation, no budget and no cooldown
    theta = threshold_on_validation(scored["val"])
    metrics["test"]["threshold"] = {"theta": round(theta, 4), **early_detection(
        test, table, None, 0, others, threshold_alerts(test, theta)),
        "classification": threshold_classification(test, theta)}
    return metrics
