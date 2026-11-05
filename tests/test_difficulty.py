"""Tests for the difficulty check: evaluation populations and unlearned rule scores."""

import pandas as pd

from earlymule.models.difficulty import populations, rule_score


def test_populations_nest():
    """Regular needs enough history; the last population also drops non-mule alert members."""
    frame = pd.DataFrame({"acct_id": [1, 2, 3], "n_tx_hist": [5, 20, 30]})
    m = populations(frame, other_members=pd.Index([3]), regular_min_tx=20)
    assert m["all"].tolist() == [True, True, True]
    assert m["regular"].tolist() == [False, True, True]
    assert m["regular_no_other_members"].tolist() == [False, True, False]


def test_pass_through_rule_needs_both_directions():
    """Money only received or only sent scores 0; otherwise the smaller side."""
    frame = pd.DataFrame({"amt_in_7d": [100.0, 0.0, 300.0], "amt_out_7d": [0.0, 50.0, 200.0], "peers_in_7d": [1, 2, 3]})
    assert rule_score(frame, "min_amt_in_out_7d").tolist() == [0.0, 0.0, 200.0]
    assert rule_score(frame, "peers_in_7d").tolist() == [1, 2, 3]


def test_new_peers_rule_sums_both_directions():
    """New payers and new payees add up."""
    frame = pd.DataFrame({"new_peers_in_7d": [0, 2], "new_peers_out_7d": [1, 3]})
    assert rule_score(frame, "new_peers_7d").tolist() == [1, 5]
