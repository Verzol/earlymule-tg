"""Tests for the Circular 17 reference rule: code 3 thresholds, the code-4 list timing, and review order."""

import numpy as np
import pandas as pd

from earlymule.models.reference_rules import code3_fires, listed_inflow, rule_scores


def test_code3_needs_both_senders_and_forwarding():
    """Fires only with at least 3 senders and 80% forwarded; no inflow (NaN share) never fires."""
    rows = pd.DataFrame({"peers_in_7d": [3, 2, 5, 4], "fwd_1d_7d": [0.8, 0.9, 0.79, np.nan]})
    assert code3_fires(rows, 3, 0.8, 7).tolist() == [True, False, False, False]


def test_listed_inflow_counts_only_after_the_sender_is_listed():
    """Account 1 meets code 3 on day 10, so it is listed from day 11; its transfers to 2 count from then on."""
    rows = pd.DataFrame({"acct_id": [1, 2, 2, 2], "snapshot": [10, 10, 11, 12]})
    fired = pd.Series([True, False, False, False])
    tx = pd.DataFrame({"orig_acct": [1, 1, 1, 3], "bene_acct": [2, 2, 2, 2], "day": [5, 10, 12, 11]})
    # Day 10: list empty. Day 11: the two earlier transfers from 1 count. Day 12: the third one too; 3 is never listed
    assert listed_inflow(tx, rows, fired).tolist() == [0, 0, 2, 3]


def test_rule_scores_order_primary_before_secondary():
    """Code-3 rows outrank code-4-only rows whatever the keys; rows that fire nothing have no score."""
    rows = pd.DataFrame({"acct_id": [1, 2, 3, 4]})
    primary = pd.Series([True, True, False, False])
    secondary = pd.Series([False, True, True, False])
    s = rule_scores(rows, primary, pd.Series([10.0, 500.0, 0, 0]), secondary, pd.Series([0, 9.0, 99.0, 0])).score
    assert s[1] > s[0] > s[2] and np.isnan(s[3])
