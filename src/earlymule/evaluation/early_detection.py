"""Early-detection metrics from daily scores: top-K alerts per day, time to detect, lead time.

Day resolution: detection on the completion day is "by completion", strictly earlier is "early".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_alerts(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k accounts of each day. scores: acct_id, snapshot, score."""
    return scores.sort_values("score", ascending=False, kind="stable").groupby("snapshot").head(k)


def early_detection(scores: pd.DataFrame, mules: pd.DataFrame, k: int) -> dict:
    """Detection timing of mules under a budget of k alerts per day.

    Args:
        scores: acct_id, snapshot, score for every scored account-day.
        mules: acct_id, onset_day, completion_day, typology of mule candidates.
    Returns:
        counts and shares; TTD and lead time in days (medians over mules detected by completion).
    """
    # 1. Mules that were scored on at least one day of their activity. Days as float: with nullable
    #    Int64, comparisons against missing values give <NA>, which .mean() silently skips.
    first, last = scores.snapshot.min(), scores.snapshot.max()
    mules = mules.astype({"onset_day": float, "completion_day": float})
    m = mules[mules.onset_day.between(first, last)].set_index("acct_id")
    scored = scores[scores.acct_id.isin(m.index)].merge(m, left_on="acct_id", right_index=True)
    in_case = scored.snapshot.between(scored.onset_day, scored.completion_day)
    m = m.loc[m.index.isin(scored.acct_id[in_case])]

    # 2. First alert on or after onset; alerts before onset are counted apart
    alerts = daily_alerts(scores, k)
    hit = alerts.merge(m, left_on="acct_id", right_index=True)
    detect = hit[hit.snapshot.ge(hit.onset_day)].groupby("acct_id").snapshot.min().reindex(m.index)
    pre_onset = hit[hit.snapshot.lt(hit.onset_day) & hit.snapshot.ge(hit.onset_day - 7)].acct_id.nunique()

    # 3. Timing versus completion; same-day cases can only be "by completion", never "early"
    by_completion = detect.le(m.completion_day)
    early = detect.lt(m.completion_day)
    ttd = (detect - m.onset_day)[by_completion]
    lead = (m.completion_day - detect)[by_completion]
    # Precision over all alerts: a hit is an alert on a mule during its own case
    active = alerts.merge(m, left_on="acct_id", right_index=True, how="left")
    precision = active.snapshot.between(active.onset_day, active.completion_day).fillna(False).sum() / len(alerts)
    feasible = m.completion_day.gt(m.onset_day)
    by_type = pd.DataFrame({"typology": m.typology, "by_completion": by_completion})
    return {
        "k_per_day": k, "mules": len(m), "alerts": len(alerts),
        "alert_precision_active_mule": round(float(precision), 4),
        "detected_by_completion": round(float(by_completion.mean()), 4),
        "detected_early": round(float(early.mean()), 4),
        "multi_day_mules": int(feasible.sum()),
        "detected_early_among_multi_day": round(float(early[feasible].mean()), 4) if feasible.any() else np.nan,
        "median_ttd_days": float(ttd.median()) if len(ttd) else np.nan,
        "median_lead_days": float(lead.median()) if len(lead) else np.nan,
        "mules_alerted_in_7_days_before_onset": int(pre_onset),
        "detected_by_completion_by_typology": by_type.groupby("typology").by_completion.mean().round(4).to_dict(),
    }
