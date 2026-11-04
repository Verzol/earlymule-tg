"""Early-detection metrics from daily scores: top-K alerts per day, time to detect, lead time.

Day resolution: detection on the completion day is "by completion", strictly earlier is "early".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_alerts(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k accounts of each day. scores: acct_id, snapshot, score."""
    return scores.sort_values("score", ascending=False, kind="stable").groupby("snapshot").head(k)


def detection_table(scores: pd.DataFrame, mules: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per mule whose case starts in the scored period and who was scored while active.

    Args:
        scores: acct_id, snapshot, score for every scored account-day.
        mules: acct_id, alert_id, onset_day, completion_day, typology of mule candidates.
    Returns:
        mules with detect_day (first alert on or after onset, NaN if none), by_completion, early, multi_day.
    """
    # 1. Days as float: with nullable Int64, comparisons against missing values give <NA>,
    #    which .mean() silently skips
    first, last = scores.snapshot.min(), scores.snapshot.max()
    m = mules.astype({"onset_day": float, "completion_day": float})
    m = m[m.onset_day.between(first, last)].set_index("acct_id")
    scored = scores[scores.acct_id.isin(m.index)].merge(m, left_on="acct_id", right_index=True)
    m = m.loc[m.index.isin(scored.acct_id[scored.snapshot.between(scored.onset_day, scored.completion_day)])]

    # 2. First alert on or after onset
    hit = daily_alerts(scores, k).merge(m, left_on="acct_id", right_index=True)
    m["detect_day"] = hit[hit.snapshot.ge(hit.onset_day)].groupby("acct_id").snapshot.min().reindex(m.index)
    m["by_completion"] = m.detect_day.le(m.completion_day)
    m["early"] = m.detect_day.lt(m.completion_day)
    m["multi_day"] = m.completion_day.gt(m.onset_day)
    return m.reset_index()


def early_detection(scores: pd.DataFrame, mules: pd.DataFrame, k: int,
                    other_members: pd.Index | None = None) -> dict:
    """Detection timing of mules under a budget of k alerts per day.

    Args:
        other_members: alert members that are not mules (sources, exits, one-sided hubs), the hard negatives.
    Returns:
        counts and shares; TTD and lead time in days (medians over mules detected by completion).
    """
    # 1. Per-mule outcomes, and the daily alerts
    t = detection_table(scores, mules, k)
    alerts = daily_alerts(scores, k)
    det = t[t.by_completion]
    hit = alerts.merge(t.set_index("acct_id"), left_on="acct_id", right_index=True)
    pre_onset = hit[hit.snapshot.lt(hit.onset_day) & hit.snapshot.ge(hit.onset_day - 7)].acct_id.nunique()

    # 2. Precision over all alerts: a hit is an alert on a mule during its own case
    active = alerts.merge(t.set_index("acct_id")[["onset_day", "completion_day"]], left_on="acct_id",
                          right_index=True, how="left")
    precision = active.snapshot.between(active.onset_day, active.completion_day).fillna(False).sum() / len(alerts)
    out = {
        "k_per_day": k, "mules": len(t), "alerts": len(alerts),
        "alert_precision_active_mule": round(float(precision), 4),
        "detected_by_completion": round(float(t.by_completion.mean()), 4),
        "detected_early": round(float(t.early.mean()), 4),
        "multi_day_mules": int(t.multi_day.sum()),
        "detected_early_among_multi_day": round(float(t.early[t.multi_day].mean()), 4) if t.multi_day.any() else np.nan,
        "median_ttd_days": float((det.detect_day - det.onset_day).median()) if len(det) else np.nan,
        "median_lead_days": float((det.completion_day - det.detect_day).median()) if len(det) else np.nan,
        "mules_alerted_in_7_days_before_onset": int(pre_onset),
        "detected_by_completion_by_typology": t.groupby("typology").by_completion.mean().round(4).to_dict(),
    }
    # 3. Share of alerts spent on hard negatives
    if other_members is not None:
        out["alert_share_other_alert_members"] = round(float(alerts.acct_id.isin(other_members).mean()), 4)
    return out
