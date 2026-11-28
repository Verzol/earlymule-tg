"""Reference rows for the daily task: the rule after Circular 17/2024 Appendix 02 and random ranking.

The rule's form was fixed before any score (reports/bank_feasibility.md section 9); nothing here is tuned.
Input: transfers and ground truth of one AMLSim run; test period, regular accounts, as every model is scored.
Output: one experiments.csv row per rule variant (code 3; code 3 or 4) and per random repetition, under
experiments/runs/<run_name>/<commit>/seed<N>/ (scores_test.parquet, detection_test.parquet).
Run: python -m earlymule.models.reference_rules [--dataset <AMLSim run>]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.data.splits import active_labels, chronological_splits
from earlymule.evaluation.daily import SCORE_COLUMNS, mule_table
from earlymule.evaluation.early_detection import daily_alerts, detection_table, early_detection
from earlymule.evaluation.uncertainty import bootstrap_detection
from earlymule.experiment_log import log_run, run_dir
from earlymule.models.tabular import build_split, load_config, load_inputs


def code3_fires(rows: pd.DataFrame, min_senders: int, min_share: float, w: int) -> pd.Series:
    """Code 3: many different senders in the window and most of the inflow forwarded within a day."""
    # fwd_1d is NaN without inflow, so an account that received nothing never fires
    return rows[f"peers_in_{w}d"].ge(min_senders) & rows[f"fwd_1d_{w}d"].ge(min_share)


def listed_inflow(tx: pd.DataFrame, rows: pd.DataFrame, fired: pd.Series) -> pd.Series:
    """Code 4 count per row: transfers received up to day t from accounts listed by the end of day t - 1.

    The list holds every account that met code 3 on an earlier scored day (alerted or not); it starts empty
    on the first scored day, as the alert history does.
    Args:
        tx: all transfers (orig_acct, bene_acct, day).
        rows: scored rows (acct_id, snapshot); fired: code 3 flag aligned with rows.
    Returns:
        count aligned with rows.
    """
    # 1. Day each account joins the list: the day after its first code-3 day
    listed_from = rows[fired].groupby("acct_id").snapshot.min() + 1
    # 2. A transfer from a listed account counts from the later of its own day and the sender's listing day
    t = tx[tx.orig_acct.isin(listed_from.index)]
    t = pd.DataFrame({"acct_id": t.bene_acct.to_numpy(),
                      "from_day": np.maximum(t.day.to_numpy(), t.orig_acct.map(listed_from).to_numpy())})
    counts = t.groupby(["acct_id", "from_day"]).size().groupby(level=0).cumsum().rename("n").reset_index()
    # 3. Running count per receiver, looked up at each scored day
    left = rows[["acct_id", "snapshot"]].reset_index().sort_values("snapshot")
    merged = pd.merge_asof(left, counts.sort_values("from_day"), left_on="snapshot", right_on="from_day",
                           by="acct_id", direction="backward")
    return merged.set_index("index").n.reindex(rows.index).fillna(0)


def rule_scores(rows: pd.DataFrame, primary: pd.Series, primary_key: pd.Series,
                secondary: pd.Series | None = None, secondary_key: pd.Series | None = None) -> pd.DataFrame:
    """Scores that put fired rows in review order; rows that do not fire get no score (NaN).

    Primary rows come first by primary_key; secondary-only rows follow by secondary_key.
    """
    # Tier plus a key scaled into [0, 1) keeps the two groups apart in one score column
    score = pd.Series(np.nan, index=rows.index)
    score[primary] = 1 + primary_key[primary] / (1 + primary_key[primary].max())
    if secondary is not None:
        only = secondary & ~primary
        score[only] = secondary_key[only] / (1 + secondary_key[only].max())
    return rows.assign(score=score)


def evaluate_alerts(scored: pd.DataFrame, inputs: dict, cfg: dict, out_dir=None) -> dict:
    """Detection metrics under each alert budget, alerting only rows with a score.

    Rows without a score stay in `scored`, so the scored population, the eligible mules and alerts per day
    are the same as for a model; they just never take an alert slot.
    """
    table, cooldown, others = mule_table(inputs["mules"]), cfg["alert_cooldown_days"], inputs["other_members"]
    candidates = scored[scored.score.notna()]
    metrics, tables = {"early_detection": []}, []
    for k in cfg["alert_budgets"]:
        alerts = daily_alerts(candidates[["acct_id", "snapshot", "score"]], k, cooldown)
        row = early_detection(scored, table, k, cooldown, others, alerts)
        tables.append(detection_table(scored, table, k, cooldown, alerts).assign(k=k))
        row.update(bootstrap_detection(tables[-1], cfg["bootstrap_samples"], cfg["seed"]))
        metrics["early_detection"].append(row)
    metrics["fired_rows_per_day"] = round(len(candidates) / scored.snapshot.nunique(), 2)
    if out_dir is not None:
        scored[SCORE_COLUMNS].to_parquet(out_dir / "scores_test.parquet")
        pd.concat(tables).to_parquet(out_dir / "detection_test.parquet")
    return metrics


def headline(metrics: dict, k: int) -> dict:
    """Primary and secondary detection shares at the headline budget, for the printed summary."""
    row = next(r for r in metrics["early_detection"] if r["k_per_day"] == k)
    keys = ["detected_before_forwarding", "detected_before_completion", "detected_by_completion", "alerts_per_day"]
    return {key: row[key] for key in keys}


def main() -> None:
    """Score the rule variants, the sensitivity grid and the random repetitions on the test period."""
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reference_rules.yaml")
    parser.add_argument("--dataset", help="AMLSim run folder; default from configs/paths.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rule, w = cfg["rule"], cfg["rule"]["window_days"]

    # 1. Inputs of the dataset; another dataset than the main one logs as <run_name>_<dataset>
    paths = load_paths()
    suffix = ""
    if args.dataset:
        paths = paths.with_dataset(args.dataset)
        suffix = "" if paths.dataset == paths.main_dataset else f"_{paths.dataset}"
    inputs = load_inputs(paths)
    mules = inputs["mules"]
    onset, completion = mules.onset_day.astype(float), mules.completion_day.astype(float)
    accts = inputs["accounts"].index

    # 2. Test rows of regular accounts, as daily_scoring builds them; only the rule's columns are kept
    test = next(s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"])) if s.name == "test")
    keep_cols = ["snapshot", "label", "typology", f"peers_in_{w}d", f"fwd_1d_{w}d", f"amt_in_{w}d"]
    rows = build_split(inputs, lambda t: active_labels(onset, completion, accts, t),
                       range(test.first_day, test.last_day + 1), cfg,
                       keep=lambda f: f.n_tx_hist.ge(cfg["regular_min_tx"]), keep_unlabeled=True,
                       reduce=lambda f: f[keep_cols])
    base_config = {**cfg, "test_days": [test.first_day, test.last_day]}
    summary = {}

    # 3. Rule rows: code 3 alone, then code 3 or code 4
    fired = code3_fires(rows, rule["min_senders"], rule["min_forward_share"], w)
    listed = listed_inflow(inputs["tx"], rows, fired)
    variants = {
        "code3": rule_scores(rows, fired, rows[f"amt_in_{w}d"]),
        "code3_or_code4": rule_scores(rows, fired, rows[f"amt_in_{w}d"],
                                      listed.ge(rule["listed_min_transfers"]), listed),
    }
    for name, scored in variants.items():
        run_name = f"{cfg['run_name']}_{name}{suffix}"
        out_dir = run_dir(run_name, cfg["seed"])
        metrics = evaluate_alerts(scored, inputs, cfg, out_dir)
        if name == "code3":
            # 4. Sensitivity grid, stored with the main row; same budget, cooldown and ordering
            metrics["grid"] = [
                {"min_senders": s, "min_forward_share": q, **headline(evaluate_alerts(
                    rule_scores(rows, code3_fires(rows, s, q, w), rows[f"amt_in_{w}d"]), inputs, cfg), cfg["headline_k"])}
                for s in cfg["grid"]["min_senders"] for q in cfg["grid"]["min_forward_share"]]
        log_run(run_name, config=base_config, seed=cfg["seed"], metrics=metrics, dataset=paths.dataset)
        summary[name] = headline(metrics, cfg["headline_k"])

    # 5. Random ranking: one uniform score per account-day, one logged row per repetition
    for seed in range(cfg["random_repetitions"]):
        run_name = f"{cfg['random_run_name']}{suffix}"
        scored = rows.assign(score=np.random.default_rng(seed).random(len(rows)))
        metrics = evaluate_alerts(scored, inputs, {**cfg, "seed": seed}, run_dir(run_name, seed))
        log_run(run_name, config={**base_config, "seed": seed}, seed=seed, metrics=metrics, dataset=paths.dataset)
        summary[f"random_seed{seed}"] = headline(metrics, cfg["headline_k"])

    summary["seconds"] = round(time.time() - start, 1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
