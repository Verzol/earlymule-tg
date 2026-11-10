"""Daily results without hard negatives, from the saved test scores of logged runs (no retraining).

Sources, exits and one-sided hubs of laundering cases are removed before ranking; alerts, cooldown and
metrics otherwise follow the source run (evaluation protocol section 5).
Run: python -m earlymule.evaluation.views [--run lightgbm_daily] [--commit C]
Appends one row per seed to experiments/experiments.csv as <run>_no_hard_negatives.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from earlymule.evaluation.early_detection import detection_table, early_detection
from earlymule.evaluation.metrics import evaluate
from earlymule.evaluation.uncertainty import bootstrap_detection
from earlymule.experiment_log import log_run, read_log, run_folder, select_runs
from earlymule.models.tabular import load_inputs


def without_hard_negatives(scores: pd.DataFrame, others: pd.Index) -> pd.DataFrame:
    """Scored rows minus the non-mule alert members; a bank could not do this, so it is a comparison view only."""
    return scores[~scores.acct_id.isin(others)]


def main() -> None:
    """Re-evaluate each seed of a logged daily run without hard negatives and log the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="lightgbm_daily")
    parser.add_argument("--commit")
    args = parser.parse_args()

    # 1. Source runs and the ground truth they were scored against
    runs = select_runs(read_log(), args.run, args.commit)
    inputs = load_inputs()
    table = inputs["mules"].reset_index()[["acct_id", "alert_id", "onset_day", "completion_day",
                                          "first_illicit_outflow_day", "typology"]]

    for _, row in runs.iterrows():
        # 2. Same alert settings as the source run, on its saved test scores
        cfg, seed = row.config, int(row.seed)
        scores = pd.read_parquet(run_folder(args.run, row.commit, seed) / "scores_test.parquet")
        kept = without_hard_negatives(scores, inputs["other_members"])
        cooldown = cfg["alert_cooldown_days"]
        detection = []
        for k in cfg["alert_budgets"]:
            d = early_detection(kept, table, k, cooldown)
            d.update(bootstrap_detection(detection_table(kept, table, k, cooldown), cfg["bootstrap_samples"], seed))
            detection.append(d)
        metrics = {"test": {"ranking": evaluate(kept, cfg["recall_k"]), "early_detection": detection}}

        # 3. Log; the config points back to the source run
        config = {"source_run": args.run, "source_commit": row.commit, "removed": "non-mule alert members",
                  "removed_rows": len(scores) - len(kept)}
        log_run(f"{args.run}_no_hard_negatives", config=config, seed=seed, metrics=metrics)
        k10 = next(x for x in detection if x["k_per_day"] == cfg["headline_k"])
        print(json.dumps({"seed": seed, "pr_auc": round(metrics["test"]["ranking"]["pr_auc"], 4),
                          "detected_before_forwarding": k10["detected_before_forwarding"]}))


if __name__ == "__main__":
    main()
