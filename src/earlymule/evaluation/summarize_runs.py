"""Summarize repeated runs in experiments/experiments.csv: mean ± sd over seeds, intervals of each seed.

Run: python -m earlymule.evaluation.summarize_runs [run_name ...]
"""

from __future__ import annotations

import sys

import pandas as pd

from earlymule.experiment_log import read_log

# (label, path into the metrics JSON); a missing path is skipped
WEEKLY = [("test PR-AUC (all)", ("test", "all", "pr_auc")), ("test Recall@100 (all)", ("test", "all", "recall_at_100")),
          ("test PR-AUC (regular)", ("test", "regular", "pr_auc")),
          ("test Recall@100 (regular)", ("test", "regular", "recall_at_100"))]


def _get(d: dict, path: tuple):
    """Nested lookup; None if any key is missing."""
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def daily_rows(metrics: dict) -> dict:
    """Flatten the daily metrics: ranking, each alert budget, the threshold rule and all-account results."""
    out = {"test PR-AUC": _get(metrics, ("test", "ranking", "pr_auc"))}
    th = _get(metrics, ("test", "threshold")) or {}
    for key in ["detected_before_forwarding", "detected_before_completion", "alerts_per_day",
                "alert_precision_active_mule"]:
        if key in th:
            out[f"threshold {key}"] = th[key]
    out["all accounts PR-AUC"] = _get(metrics, ("test_all_accounts", "ranking", "pr_auc"))
    for e in _get(metrics, ("test_all_accounts", "early_detection")) or []:
        out[f"all accounts K={e['k_per_day']} before forwarding"] = e["detected_before_forwarding"]
    for e in _get(metrics, ("test", "early_detection")) or []:
        k = e["k_per_day"]
        # Runs before the protocol review (2026-11-05) lack the forwarding metric and used "detected_early"
        out[f"K={k} before forwarding"] = e.get("detected_before_forwarding")
        out[f"K={k} by completion"] = e["detected_by_completion"]
        out[f"K={k} before completion"] = e.get("detected_before_completion", e.get("detected_early"))
        out[f"K={k} before completion, multi-day"] = e.get("detected_before_completion_multi_day")
        out[f"K={k} precision"] = e["alert_precision_active_mule"]
        out[f"K={k} alerts on other members"] = e.get("alert_share_other_alert_members")
    return out


def summarize(log: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Mean and sd over the latest clean commit's runs of run_name, one row per metric.

    Runs from uncommitted code are left out (protocol rule); a seed rerun on the same commit counts once,
    with its latest row.
    """
    runs = log[log.run_name.eq(run_name) & ~log.dirty.astype(str).str.lower().eq("true")]
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs.commit.eq(runs.commit.iloc[-1])].drop_duplicates("seed", keep="last")
    flat = [daily_rows(m) if "ranking" in m.get("test", {}) else {lab: _get(m, p) for lab, p in WEEKLY}
            for m in runs.metrics]
    table = pd.DataFrame(flat, index=runs.seed.to_numpy()).dropna(axis=1, how="all")
    return pd.DataFrame({"mean": table.mean(), "sd": table.std(ddof=1), "seeds": len(table)}).round(3)


def main() -> None:
    """Print a summary for each requested run name (default: every name in the log)."""
    log = read_log()
    for name in sys.argv[1:] or log.run_name.unique():
        print(f"\n== {name}\n{summarize(log, name).to_string()}")


if __name__ == "__main__":
    main()
