"""Tests for recovering the within-day execution order from AMLSim's balance log (data.intraday_order)."""

import pandas as pd

from earlymule.data.intraday_order import balance_chain, tran_id_matches_log


def log(rows):
    """tx_log rows from (step, orig, dest, amount) tuples, with balances computed in the given order."""
    bal, out = {1: 100.0, 2: 50.0, 3: 0.0}, []
    for step, o, d, amt in rows:
        out.append({"step": step, "nameOrig": o, "nameDest": d, "amount": amt,
                    "oldbalanceOrig": bal[o], "newbalanceOrig": bal[o] - amt,
                    "oldbalanceDest": bal[d], "newbalanceDest": bal[d] + amt})
        bal[o] -= amt
        bal[d] += amt
    return pd.DataFrame(out)


def test_chain_holds_in_execution_order_and_breaks_when_swapped():
    """Account 2 receives then forwards on the same day: the chain holds only in that order."""
    rows = [(0, 1, 2, 30.0), (0, 2, 3, 70.0), (1, 1, 3, 5.0)]
    chain = balance_chain(log(rows))
    assert chain.ok.all() and chain.same_day.sum() == 1  # only account 2 has two transfers on day 0
    swapped = log(rows).iloc[[1, 0, 2]].reset_index(drop=True)
    assert not balance_chain(swapped).ok.all()


def test_tran_id_points_at_log_row():
    """tran_id is the 1-based tx_log row; a dropped log row does not shift the others."""
    lg = log([(0, 1, 2, 30.0), (0, 1, 2, 30.0), (0, 2, 3, 70.0)])
    tx = pd.DataFrame({"tran_id": [1, 3], "orig_acct": [1, 2], "bene_acct": [2, 3], "base_amt": [30.0, 70.0]})
    assert tran_id_matches_log(tx, lg)
    assert not tran_id_matches_log(tx.assign(tran_id=[1, 2]), lg)
