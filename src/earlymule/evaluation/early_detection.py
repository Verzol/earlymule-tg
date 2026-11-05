"""Early-detection metrics from daily scores: top-K new alerts per day, detection before forwarding and completion.

Day resolution: an alert on day d uses transfers up to the end of day d. Detection on the completion day is
"by completion", strictly earlier is "before completion"; strictly before the first illicit outflow day is
"before forwarding".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_alerts(scores: pd.DataFrame, k: int, cooldown: int = 0) -> pd.DataFrame:
    """Top-k accounts of each day, skipping accounts alerted in the previous `cooldown` days.

    Args:
        scores: acct_id, snapshot, score.
        cooldown: days an alerted account stays reviewed; it can be alerted again on day d + cooldown.
            0 = no suppression (the same account may take a slot every day).
    """
    ranked = scores.sort_values("score", ascending=False, kind="stable")
    if cooldown <= 0:
        return ranked.groupby("snapshot").head(k)
    # Walk the days in order: the suppression on day d depends on the alerts of earlier days
    last_alert: dict[int, int] = {}
    picked = []
    for day, g in ranked.groupby("snapshot", sort=True):
        taken = []
        for idx, acct in zip(g.index, g.acct_id.to_numpy()):
            if day - last_alert.get(acct, -cooldown) >= cooldown:
                taken.append(idx)
                last_alert[acct] = day
                if len(taken) == k:
                    break
        picked.extend(taken)
    return ranked.loc[picked]


def detection_table(scores: pd.DataFrame, mules: pd.DataFrame, k: int, cooldown: int = 0) -> pd.DataFrame:
    """One row per mule whose case starts in the scored period and who was scored while active.

    Args:
        scores: acct_id, snapshot, score for every scored account-day (finished mules included).
        mules: acct_id, alert_id, onset_day, completion_day, first_illicit_outflow_day, typology of mule candidates.
    Returns:
        mules with detect_day (first alert on or after onset, NaN if none) and the flags by_completion,
        before_completion, multi_day, can_precede_forwarding, before_forwarding.
    """
    # 1. Days as float: with nullable Int64, comparisons against missing values give <NA>,
    #    which .mean() silently skips
    first, last = scores.snapshot.min(), scores.snapshot.max()
    days = ["onset_day", "completion_day", "first_illicit_outflow_day"]
    m = mules.astype({c: float for c in days})
    m = m[m.onset_day.between(first, last)].set_index("acct_id")
    scored = scores[scores.acct_id.isin(m.index)].merge(m, left_on="acct_id", right_index=True)
    m = m.loc[m.index.isin(scored.acct_id[scored.snapshot.between(scored.onset_day, scored.completion_day)])]

    # 2. First alert on or after onset; an alert before onset can suppress the mule under a cooldown
    hit = daily_alerts(scores, k, cooldown).merge(m, left_on="acct_id", right_index=True)
    m["detect_day"] = hit[hit.snapshot.ge(hit.onset_day)].groupby("acct_id").snapshot.min().reindex(m.index)
    m["by_completion"] = m.detect_day.le(m.completion_day)
    m["before_completion"] = m.detect_day.lt(m.completion_day)
    m["multi_day"] = m.completion_day.gt(m.onset_day)
    # Catching a mule before it forwards needs a day between its onset and its first outflow
    m["can_precede_forwarding"] = m.first_illicit_outflow_day.gt(m.onset_day)
    m["before_forwarding"] = m.detect_day.lt(m.first_illicit_outflow_day)
    return m.reset_index()


def early_detection(scores: pd.DataFrame, mules: pd.DataFrame, k: int, cooldown: int = 0,
                    other_members: pd.Index | None = None) -> dict:
    """Detection timing of mules under a budget of k new alerts per day.

    Args:
        cooldown: see daily_alerts.
        other_members: alert members that are not mules (sources, exits, one-sided hubs), the hard negatives.
    Returns:
        counts and shares; TTD and lead time in days (medians over mules detected by completion).
    """
    # 1. Per-mule outcomes, and the daily alerts
    t = detection_table(scores, mules, k, cooldown)
    alerts = daily_alerts(scores, k, cooldown)
    det = t[t.by_completion]
    elig = t[t.can_precede_forwarding]
    hit = alerts.merge(t.set_index("acct_id"), left_on="acct_id", right_index=True)
    pre_onset = hit[hit.snapshot.lt(hit.onset_day) & hit.snapshot.ge(hit.onset_day - 7)].acct_id.nunique()

    # 2. Precision over all alerts: a hit is an alert on a mule during its own case, including mules that
    #    started before the scored period; alerts on finished mules miss
    spans = mules.astype({"onset_day": float, "completion_day": float}).set_index("acct_id")
    active = alerts.merge(spans[["onset_day", "completion_day"]], left_on="acct_id", right_index=True, how="left")
    precision = active.snapshot.between(active.onset_day, active.completion_day).fillna(False).sum() / len(alerts)
    share = lambda s: round(float(s.mean()), 4) if len(s) else np.nan  # noqa: E731
    out = {
        "k_per_day": k, "cooldown_days": cooldown, "mules": len(t), "alerts": len(alerts),
        "distinct_alerted_accounts": int(alerts.acct_id.nunique()),
        "alert_precision_active_mule": round(float(precision), 4),
        "mules_can_precede_forwarding": len(elig),
        "detected_before_forwarding": share(elig.before_forwarding),
        "detected_before_forwarding_all_mules": share(t.before_forwarding),
        "detected_by_completion": share(t.by_completion),
        "detected_before_completion": share(t.before_completion),
        "multi_day_mules": int(t.multi_day.sum()),
        "detected_before_completion_multi_day": share(t.before_completion[t.multi_day]),
        "median_ttd_days": float((det.detect_day - det.onset_day).median()) if len(det) else np.nan,
        "median_lead_days": float((det.completion_day - det.detect_day).median()) if len(det) else np.nan,
        "mules_alerted_in_7_days_before_onset": int(pre_onset),
        "detected_before_forwarding_by_typology": elig.groupby("typology").before_forwarding.mean().round(4).to_dict(),
        "detected_by_completion_by_typology": t.groupby("typology").by_completion.mean().round(4).to_dict(),
    }
    # 3. Share of alerts spent on hard negatives
    if other_members is not None:
        out["alert_share_other_alert_members"] = round(float(alerts.acct_id.isin(other_members).mean()), 4)
    return out
