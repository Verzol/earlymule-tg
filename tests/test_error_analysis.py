"""Tests for the error analysis: best rank before forwarding with the cooldown, and alert kinds."""

import pandas as pd

from earlymule.analysis.error_analysis import alert_kind, missed_rank


def test_missed_rank_skips_accounts_in_cooldown():
    """Accounts alerted in the cooldown days are not counted above the mule; the best day wins."""
    scores = pd.DataFrame({"acct_id": [1, 2, 3, 1, 2, 3], "snapshot": [5, 5, 5, 6, 6, 6],
                           "score": [0.5, 0.9, 0.8, 0.5, 0.9, 0.4]})
    alerts = pd.DataFrame({"acct_id": [2], "snapshot": [4]})
    mules = pd.DataFrame({"acct_id": [1], "onset_day": [5], "first_illicit_outflow_day": [7]})
    r = missed_rank(scores, alerts, mules, cooldown=30).iloc[0]
    assert (r.best_rank, r.best_day, r.in_cooldown) == (1, 6, False)


def test_alert_kind_separates_mules_members_and_ordinary_accounts():
    """Mule in its case days, mule after it, a source during its case, and an account outside every case."""
    gt = pd.DataFrame({"alert_id": [0, 0], "acct_id": [1, 2], "role": ["relay", "source"],
                       "is_mule_candidate": [True, False], "onset_day": [5, 3], "completion_day": [6, 3]})
    alerts = pd.DataFrame({"acct_id": [1, 1, 2, 9], "snapshot": [5, 9, 4, 5]})
    assert alert_kind(alerts, gt).tolist() == ["active mule", "mule outside its case", "member source, during case",
                                               "ordinary account"]
