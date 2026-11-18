"""Transfer-time features: hand-checked values, no use of later transfers, conservative cut, brute-force match."""

import numpy as np
import pandas as pd
import pytest

from earlymule.features.transfer_time import order_features, risk_features, transfer_features


def tx(rows):
    """Transfers from (orig, bene, amount, day) tuples in execution order; seq = 1-based position."""
    t = pd.DataFrame(rows, columns=["orig_acct", "bene_acct", "base_amt", "day"])
    t["seq"] = np.arange(1, len(t) + 1)
    return t.sort_values(["day", "seq"], kind="stable").reset_index(drop=True)


def at(t, seq):
    """Position of the transfer with this seq."""
    return np.flatnonzero(t.seq.to_numpy() == seq)


BASE = [(9, 1, 40.0, 1), (2, 1, 100.0, 3), (1, 4, 30.0, 3), (3, 1, 50.0, 4), (2, 1, 20.0, 4), (1, 5, 90.0, 4)]


def test_hand_checked_values():
    """Account 1 receives 100 (day 3), sends 30, receives 50 (new sender 3) and 20, then sends 90 to new payee 5."""
    t = tx(BASE)
    f = order_features(t, np.full(len(t), 180.0), at(t, 6)).iloc[0]
    assert f.amt == 90 and f.amt_over_in_since_last_out == pytest.approx(90 / 70)
    assert f.s_in_amt_today == 70 and f.s_in_n_today == 2 and f.s_in_new_amt_today == 50
    assert f.amt_over_balance == pytest.approx(0.5) and f.amt_over_max_out == 3 and f.amt_over_last_in == 4.5
    assert f.days_since_last_in == 0 and f.tx_since_last_in == 0 and f.first_to_payee == 1
    # Payee 4 sent nothing back, payee 5 was never seen: no pair history either way
    assert f.pair_n_before == 0 and f.reverse_n_before == 0 and f.r_in_n_before == 0


def test_later_transfers_change_nothing():
    """Adding transfers after x, on the same day or later, leaves every feature of x unchanged."""
    t = tx(BASE)
    later = tx(BASE + [(1, 5, 7.0, 4), (5, 1, 3.0, 4), (1, 5, 1.0, 5)])
    a = transfer_features(t, np.full(len(t), 180.0), at(t, 6), windows=(1, 7))
    b = transfer_features(later, np.full(len(later), 180.0), at(later, 6), windows=(1, 7))
    pd.testing.assert_frame_equal(a, b)


def test_earlier_same_day_transfer_moves_only_the_exact_cut():
    """A transfer earlier on x's day changes the exact features, not the conservative ones."""
    extra = BASE[:3] + [(7, 1, 500.0, 4)] + BASE[3:]
    t, e = tx(BASE), tx(extra)
    balance_t, balance_e = np.full(len(t), 180.0), np.full(len(e), 680.0)
    exact = order_features(t, balance_t, at(t, 6)), order_features(e, balance_e, at(e, 7))
    assert exact[1].s_in_amt_today.iloc[0] == exact[0].s_in_amt_today.iloc[0] + 500
    cons = order_features(t, balance_t, at(t, 6), True), order_features(e, balance_e, at(e, 7), True)
    # The extra 500 raised the balance before x by 500 and is backed out: end-of-day 3 balance in both
    pd.testing.assert_frame_equal(cons[0], cons[1])
    assert cons[0].s_in_amt_today.iloc[0] == 0


def test_matches_brute_force():
    """Cumulative look-ups equal a direct scan of the earlier transfers, on random data."""
    rng = np.random.default_rng(0)
    n = 300
    t = tx(list(zip(rng.integers(0, 6, n), rng.integers(0, 6, n), rng.integers(1, 100, n).astype(float),
                    np.sort(rng.integers(0, 8, n)))))
    rows = np.arange(n)
    f = order_features(t, np.full(n, 1000.0), rows)
    for p in rng.choice(n, 40, replace=False):
        x, before = t.iloc[p], t.iloc[:p]
        s, r = x.orig_acct, x.bene_acct
        today = before[before.day == x.day]
        assert f.pair_n_before[p] == ((before.orig_acct == s) & (before.bene_acct == r)).sum()
        assert f.reverse_n_before[p] == ((before.orig_acct == r) & (before.bene_acct == s)).sum()
        assert f.s_in_amt_today[p] == pytest.approx(today.base_amt[today.bene_acct == s].sum())
        assert f.r_out_n_today[p] == (today.orig_acct == r).sum()
        outs = before.base_amt[before.orig_acct == s]
        assert f.amt_over_max_out[p] == pytest.approx(x.base_amt / outs.max()) if len(outs) else np.isnan(f.amt_over_max_out[p])
        sent = before.index[before.orig_acct == s]
        since = before.loc[(before.index > (sent.max() if len(sent) else -1)) & (before.bene_acct == s), "base_amt"].sum()
        expected = x.base_amt / since if since > 0 else np.nan
        assert f.amt_over_in_since_last_out[p] == pytest.approx(expected, nan_ok=True, rel=1e-5)


def test_risk_features_use_scores_of_the_previous_day():
    """Payee score is the day d - 1 score; propagation counts inflow of days d - 7 .. d - 1 from top senders of d - 1."""
    t = tx([(7, 1, 30.0, 2), (8, 1, 70.0, 3), (7, 1, 999.0, 5), (1, 5, 10.0, 5)])
    keys = t[t.seq == 4][["seq", "day", "orig_acct", "bene_acct"]]
    scores = pd.DataFrame({"acct_id": [5, 5, 7, 8] + list(range(100, 300)), "snapshot": [4, 5, 4, 4] + [4] * 200,
                           "score": [2.0, 9.0, 50.0, -1.0] + [0.0] * 200})
    f = risk_features(keys, t, scores, hot_share=0.01).iloc[0]
    # Payee 5 scored 2.0 on day 4 (its day-5 score is not used); sender 7 is the top 1% of day 4 (1 of 203)
    assert f.r_risk == 2.0 and f.hot_in_share_7d == pytest.approx(30 / 100)
