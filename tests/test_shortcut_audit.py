"""Tests for the single-feature shortcut screening."""

import numpy as np
import pandas as pd

from earlymule.data.shortcut_audit import account_activity, single_feature_auc


def test_account_activity_counts_normal_transfers_only():
    """Illicit transfers are ignored; accounts without normal activity get zeros."""
    accounts = pd.DataFrame({"acct_id": [1, 2, 3], "initial_deposit": [10.0, 20.0, 30.0]})
    tx = pd.DataFrame({"orig_acct": [1, 1, 2, 3], "bene_acct": [2, 3, 1, 1],
                       "base_amt": [100.0, 300.0, 50.0, 999.0], "is_sar": [False, False, "false", "True"]})
    f = account_activity(accounts, tx)
    assert f.loc[1, ["normal_out", "normal_in", "n_out_peers", "mean_out_amt"]].tolist() == [2, 1, 2, 200.0]
    assert f.loc[3, ["normal_out", "normal_in", "normal_total"]].tolist() == [0, 1, 1]
    assert np.isnan(f.loc[3, "mean_out_amt"])


def test_single_feature_auc_direction_and_populations():
    """1.0 and 0.0 both mark a shortcut; a one-class population gives NaN."""
    f = pd.DataFrame({"up": [1, 2, 3, 4], "down": [4, 3, 2, 1]}, index=[10, 11, 12, 13])
    y = pd.Series([False, False, True, True], index=f.index)
    pops = {"all": pd.Series(True, index=f.index), "negatives": ~y}
    t = single_feature_auc(f, y, pops)
    assert (t.loc["up", "all"], t.loc["down", "all"]) == (1.0, 0.0)
    assert t["negatives"].isna().all()
