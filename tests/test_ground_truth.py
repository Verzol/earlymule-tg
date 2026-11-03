"""Tests for ground-truth extraction: hand-built alerts, plus a check against the EDA notebook on the 10K run."""

import pandas as pd
import pytest

from earlymule.config import load_paths, repo_root
from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth


@pytest.fixture
def toy_gt():
    """Fan-in (3 sources -> 10) and a 3-account cycle with two same-day hops, plus one silent member."""
    accounts = pd.DataFrame({
        "alert_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "alert_type": ["fan_in"] * 4 + ["cycle"] * 4,
        "acct_id": [10, 11, 12, 13, 20, 21, 22, 23],
    })
    tx = pd.DataFrame({
        "alert_id": [1, 1, 1, 2, 2, 2],
        "alert_type": ["fan_in"] * 3 + ["cycle"] * 3,
        "tran_id": [100, 101, 102, 200, 201, 202],
        "orig_acct": [11, 12, 13, 20, 21, 22],
        "bene_acct": [10, 10, 10, 21, 22, 20],
        "base_amt": [500.0] * 6,
        "day": [2, 5, 5, 3, 3, 4],
    })
    members = pd.DataFrame({"alert_id": [1, 2], "acct_id": [10, 20], "is_main": [True, True]})
    return build_ground_truth(accounts, tx, members).set_index("acct_id")


def test_collector_milestones(toy_gt):
    """Collector gets inflow milestones; its missing outflow stays <NA>, not 0."""
    r = toy_gt.loc[10]
    assert (r.role, r.n_illicit_in, r.n_illicit_out) == ("collector", 3, 0)
    assert (r.onset_day, r.first_illicit_inflow_day, r.completion_day) == (2, 2, 5)
    assert pd.isna(r.first_illicit_outflow_day)
    assert r.upstream_accounts == [11, 12, 13] and r.downstream_accounts == []
    assert r.is_main and not r.is_mule_candidate


def test_fan_in_sources_are_not_candidates(toy_gt):
    """Accounts that only send once are sources, not mule candidates."""
    assert (toy_gt.loc[[11, 12, 13], "role"] == "source").all()
    assert not toy_gt.loc[[11, 12, 13], "is_mule_candidate"].any()


def test_cycle_relays_same_day(toy_gt):
    """Same-day inflow and outflow give equal days; no order is invented."""
    r = toy_gt.loc[21]
    assert r.role == "relay"
    assert r.is_mule_candidate
    assert (r.first_illicit_inflow_day, r.first_illicit_outflow_day, r.onset_day, r.completion_day) == (3, 3, 3, 3)
    main = toy_gt.loc[20]
    assert (main.onset_day, main.completion_day) == (3, 4)


def test_member_without_illicit_tx(toy_gt):
    """A member with no illicit events keeps a row with empty milestones."""
    r = toy_gt.loc[23]
    assert r.role == "no_illicit" and not r.is_mule_candidate
    assert pd.isna(r.onset_day) and pd.isna(r.completion_day)
    assert not r.is_main


def test_to_day_counts_from_base_date():
    """Day 0 is base_date; a time of day other than midnight is rejected."""
    base = pd.Timestamp("2017-01-01", tz="UTC")
    days = amlsim.to_day(pd.Series(["2017-01-01T00:00:00Z", "2017-01-05T00:00:00Z"]), base)
    assert days.tolist() == [0, 4]
    with pytest.raises(ValueError):
        amlsim.to_day(pd.Series(["2017-01-01T12:00:00Z"]), base)


def test_matches_eda_notebook_on_10k():
    """Keep the notebook's structural counts; only its 285 two-sided relays are now positive."""
    paths = load_paths()
    run, tmp = paths.amlsim_outputs / "10K", paths.amlsim_tmp / "10K"
    if not (run / "alert_transactions.csv").is_file() or not (tmp / "alert_members.csv").is_file():
        pytest.skip("10K run not generated on this machine")
    base = amlsim.read_base_date(repo_root() / "AMLSim" / "conf_10K.json")
    gt = build_ground_truth(amlsim.load_alert_accounts(run), amlsim.load_alert_transactions(run, base),
                            amlsim.load_alert_members(tmp))
    assert len(gt) == 742
    assert int(gt.role.isin(["relay", "collector", "disperser"]).sum()) == 345
    assert int(gt.is_mule_candidate.sum()) == 285
    assert int((gt.role == "relay").sum()) == 285


def test_directions_in_different_alerts_do_not_make_a_mule():
    """Receiving in one case and sending in another cannot establish an intermediary in either."""
    accounts = pd.DataFrame({"alert_id": [1, 1, 2, 2], "acct_id": [10, 11, 10, 12],
                             "alert_type": ["fan_in", "fan_in", "fan_out", "fan_out"]})
    tx = pd.DataFrame({"alert_id": [1, 2], "tran_id": [1, 2], "orig_acct": [11, 10],
                       "bene_acct": [10, 12], "day": [1, 2]})
    members = pd.DataFrame({"alert_id": [1, 2], "acct_id": [10, 10], "is_main": [True, True]})
    gt = build_ground_truth(accounts, tx, members)
    assert not gt.is_mule_candidate.any()


def test_one_sided_fan_hubs_are_not_mules():
    """Multiple inbound or outbound payments alone do not satisfy the approved two-sided rule."""
    accounts = pd.DataFrame({"alert_id": [1] * 6, "acct_id": [10, 11, 12, 20, 21, 22],
                             "alert_type": ["fan"] * 6})
    tx = pd.DataFrame({"alert_id": [1] * 4, "tran_id": [1, 2, 3, 4],
                       "orig_acct": [11, 12, 20, 20], "bene_acct": [10, 10, 21, 22],
                       "day": [1, 2, 1, 2]})
    members = pd.DataFrame({"alert_id": [1, 1], "acct_id": [10, 20], "is_main": [True, True]})
    gt = build_ground_truth(accounts, tx, members).set_index("acct_id")
    assert gt.loc[10, "role"] == "collector" and gt.loc[20, "role"] == "disperser"
    assert not gt.is_mule_candidate.any()


def test_main_and_day_order_do_not_replace_the_two_sided_rule(toy_gt):
    """A cycle main has both directions despite sending first; same-day order is unobserved."""
    main = toy_gt.loc[20]
    assert main.is_mule_candidate and main.first_illicit_outflow_day < main.first_illicit_inflow_day
    assert toy_gt.loc[21, "is_mule_candidate"]


def test_self_transfer_cannot_create_an_intermediary():
    """Reject a self-loop instead of inventing two independent sides from one payment."""
    accounts = pd.DataFrame({"alert_id": [1], "acct_id": [10], "alert_type": ["cycle"]})
    tx = pd.DataFrame({"alert_id": [1], "tran_id": [1], "orig_acct": [10],
                       "bene_acct": [10], "day": [1]})
    members = pd.DataFrame({"alert_id": [1], "acct_id": [10], "is_main": [True]})
    with pytest.raises(ValueError, match="self-transfers"):
        build_ground_truth(accounts, tx, members)
