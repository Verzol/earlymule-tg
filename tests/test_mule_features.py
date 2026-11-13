"""Hand-checked values of the research plan's mule-behaviour features (point_in_time.mule_behaviour)."""

import numpy as np
import pandas as pd
import pytest

from earlymule.features.point_in_time import snapshot_features

ACCTS = pd.Index([1, 2, 3, 4, 5], name="acct_id")


def tx(rows):
    """Transfers from (orig, bene, amount, day) tuples, sorted by day."""
    return pd.DataFrame(rows, columns=["orig_acct", "bene_acct", "base_amt", "day"]).sort_values("day").reset_index(drop=True)


def test_rapid_forward_and_dwell_time():
    """Account 1 receives 100 on day 5 and 100 on day 8, sends 60 on day 5 and 100 on day 10 (t = 10, w = 7)."""
    f = snapshot_features(tx([(2, 1, 100.0, 5), (1, 3, 60.0, 5), (4, 1, 100.0, 8), (1, 3, 100.0, 10)]),
                          ACCTS, t=10, windows=(7,))
    # Same day: day 5 receipt covered by 60 of the 60 sent that day; day 8 receipt by nothing -> 60 / 200
    assert f.loc[1, "fwd_0d_7d"] == pytest.approx(0.3)
    # Within 1 day: day 5 -> days 5-6 cover 60; day 8 -> days 8-9 cover 0
    assert f.loc[1, "fwd_1d_7d"] == pytest.approx(0.3)
    # Within 3 days: day 8 -> days 8-11 clipped at 10 cover 100 -> (60 + 100) / 200
    assert f.loc[1, "fwd_3d_7d"] == pytest.approx(0.8)
    # Dwell: receipt day 5 -> sending day 5 (0 days); receipt day 8 -> day 10 (2 days)
    assert f.loc[1, "dwell_min_7d"] == 0 and f.loc[1, "dwell_med_7d"] == 1
    # Account 4 only sends, account 5 does nothing: no receipt, no value
    assert np.isnan(f.loc[4, "fwd_0d_7d"]) and np.isnan(f.loc[5, "dwell_med_7d"])
    # Account 2 received nothing after... it never receives; account 3 receives but never sends: dwell undefined
    assert np.isnan(f.loc[3, "dwell_min_7d"]) and f.loc[3, "fwd_3d_7d"] == 0


def test_imbalance_entropy_and_two_hop():
    """Signed imbalance, amount-share entropy per direction and the window totals of the partners."""
    f = snapshot_features(tx([(2, 1, 50.0, 9), (3, 1, 50.0, 10), (1, 4, 300.0, 10), (5, 2, 20.0, 10), (4, 5, 7.0, 10)]),
                          ACCTS, t=10, windows=(7,))
    assert f.loc[1, "imbalance_7d"] == pytest.approx((300 - 100) / 400)
    assert f.loc[1, "ent_in_7d"] == pytest.approx(np.log(2)) and f.loc[1, "ent_out_7d"] == pytest.approx(0.0)
    assert not np.isnan(f.loc[5, "imbalance_7d"]) and np.isnan(f.loc[3, "ent_in_7d"])
    # Upstream: inflow of senders 2 (20 from 5) and 3 (0) -> 20; downstream: outflow of receiver 4 (7) -> 7
    assert f.loc[1, "up_amt_in_7d"] == 20 and f.loc[1, "down_amt_out_7d"] == 7


def test_burstiness():
    """Regular gaps give −1; same-day bursts plus a long gap give a value near 1; too few gaps give no value."""
    regular = tx([(1, 2, 1.0, d) for d in (2, 4, 6, 8, 10)])
    bursty = tx([(1, 2, 1.0, 1), (1, 2, 1.0, 1), (1, 2, 1.0, 1), (1, 2, 1.0, 1), (1, 2, 1.0, 10)])
    r = snapshot_features(regular, ACCTS, t=10, windows=(30,))
    b = snapshot_features(bursty, ACCTS, t=10, windows=(30,))
    assert r.loc[1, "burst_30d"] == pytest.approx(-1.0)
    assert b.loc[1, "burst_30d"] == pytest.approx((np.std([0, 0, 0, 9]) - 2.25) / (np.std([0, 0, 0, 9]) + 2.25))
    assert np.isnan(r.loc[3, "burst_30d"]) and "burst_1d" not in snapshot_features(regular, ACCTS, 10, (1,)).columns


def test_new_features_ignore_transfers_after_t_and_before_the_window():
    """Transfers after t, or before the window, leave the window's features unchanged."""
    base = tx([(2, 1, 100.0, 5), (1, 3, 60.0, 5), (4, 1, 100.0, 8), (1, 3, 100.0, 10), (1, 2, 5.0, 2)])
    later = pd.concat([base, tx([(1, 3, 999.0, 11), (5, 1, 999.0, 12)])], ignore_index=True)
    earlier = pd.concat([tx([(5, 1, 999.0, 1), (1, 5, 999.0, 3)]), base], ignore_index=True)
    a = snapshot_features(base, ACCTS, t=10, windows=(7,))
    pd.testing.assert_frame_equal(a, snapshot_features(later, ACCTS, t=10, windows=(7,)))
    new = [c for c in a.columns if c.split("_")[0] in ("burst", "fwd", "dwell", "up", "down", "ent", "imbalance")]
    pd.testing.assert_frame_equal(a[new], snapshot_features(earlier, ACCTS, t=10, windows=(7,))[new])
