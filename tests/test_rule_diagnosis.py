"""Tests for the rule diagnosis: per-mule flags over case days and the per-typology summary."""

import pandas as pd

from earlymule.analysis.rule_diagnosis import per_mule, summary


def test_per_mule_needs_both_conditions_on_the_same_day():
    """Mule 1 meets each condition on a different day, so the rule never fires; on illicit money alone it does."""
    rows = pd.DataFrame({"acct_id": [1, 1], "senders": [3, 1], "share": [0.2, 0.9],
                         "senders_ok": [True, False], "share_ok": [False, True],
                         "ill_senders": [3, 3], "ill_share": [0.9, 0.9],
                         "ill_senders_ok": [True, True], "ill_share_ok": [True, True]})
    mules = pd.DataFrame({"typology": ["relay"], "onset_day": [5], "completion_day": [6]}, index=pd.Index([1]))
    m = per_mule(rows, mules)
    assert (m.loc[1, "senders_ok"], m.loc[1, "share_ok"], m.loc[1, "both"], m.loc[1, "ill_both"]) == (True, True, False, True)
    s = summary(m)
    assert s.loc["all", "both"] == 0 and s.loc["relay", "mules"] == 1
