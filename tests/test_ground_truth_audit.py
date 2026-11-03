"""Tests for the layered-typology checks of the label audit and for reading planned layers."""

import pandas as pd
import pytest

from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth
from earlymule.data.ground_truth_audit import layered_flow_checks, planned_edges

# Alert 1: M1, sources 10, 11 -> mule 12 -> exits 13, 14 (complete).
# Alert 2: M4, sources 20, 21 -> collector 22 -> relays 23, 24 -> exits 25, 26.
#   Relay 23 pays 26 (unplanned) before it receives; relay 24 never forwards.
# Alert 3: cycle without layers, ignored by the layered checks.
MEMBERS = pd.DataFrame({
    "alert_id": [1] * 5 + [2] * 7 + [3] * 3,
    "acct_id": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32],
    "is_main": [False, False, True, False, False, False, False, True, False, False, False, False, True, False, False],
    "planned_layer": pd.array([0, 0, 1, 2, 2, 0, 0, 1, 2, 2, 3, 3, None, None, None], dtype="Int64"),
})
TX = pd.DataFrame({
    "alert_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
    "tran_id": range(12),
    "orig_acct": [10, 11, 12, 12, 20, 21, 22, 22, 23, 30, 31, 32],
    "bene_acct": [12, 12, 13, 14, 22, 22, 23, 24, 26, 31, 32, 30],
    "base_amt": [100.0, 100.0, 90.0, 90.0, 100.0, 100.0, 90.0, 90.0, 50.0, 10.0, 9.0, 8.0],
    "day": [1, 2, 3, 3, 1, 1, 2, 2, 1, 5, 6, 7],
})


@pytest.fixture
def checks():
    """Run the layered checks on the toy alerts."""
    accounts = MEMBERS[["alert_id", "acct_id"]].assign(
        alert_type=MEMBERS.alert_id.map({1: "mule_fan_in_out", 2: "mule_collect_disperse", 3: "cycle"}))
    gt = build_ground_truth(accounts, TX, MEMBERS)
    return layered_flow_checks(gt, TX, MEMBERS)


def test_planned_edges_pair_equal_layers_by_file_order():
    """Equal-size layers pair by position, so reordering exits changes the plan."""
    edges = planned_edges(MEMBERS)
    pairs = set(map(tuple, edges[edges.from_layer.eq(2)][["orig_acct", "bene_acct"]].to_numpy()))
    assert pairs == {(23, 25), (24, 26)}
    swapped = MEMBERS.copy()
    swapped.loc[swapped.acct_id.isin([25, 26]), "acct_id"] = [26, 25]
    edges = planned_edges(swapped)
    assert set(map(tuple, edges[edges.from_layer.eq(2)][["orig_acct", "bene_acct"]].to_numpy())) == {(23, 26), (24, 25)}


def test_layered_edges_and_completeness(checks):
    """Missing and unplanned edges are counted; only the M1 alert is complete."""
    assert checks["layered_alerts"] == 2
    assert (checks["planned_edges"], checks["planned_edges_observed"], checks["unplanned_illicit_edges"]) == (10, 8, 1)
    assert checks["complete_alerts_by_typology"] == {"mule_collect_disperse": 0, "mule_fan_in_out": 1}


def test_layered_mule_checks(checks):
    """Relays count as planned mules; out-before-in and partial forwarding are reported."""
    assert (checks["planned_mules"], checks["planned_mules_two_sided"]) == (4, 3)
    assert checks["planned_ends_as_candidates"] == 0
    assert (checks["mules_out_before_in"], checks["mules_same_day"]) == (1, 0)
    assert checks["mule_forward_ratio"]["max"] == pytest.approx(0.9)
    assert checks["mule_forward_ratio"]["min"] == pytest.approx(50 / 90)


def test_run_without_layered_alerts():
    """Runs with only the original typologies give zero counts instead of failing."""
    members, tx = MEMBERS[MEMBERS.alert_id.eq(3)], TX[TX.alert_id.eq(3)]
    accounts = members[["alert_id", "acct_id"]].assign(alert_type="cycle")
    c = layered_flow_checks(build_ground_truth(accounts, tx, members), tx, members)
    assert (c["layered_alerts"], c["planned_edges"], c["planned_mules"]) == (0, 0, 0)
    assert c["complete_alerts_by_typology"] == {} and c["mule_forward_ratio"] == {}


def test_unequal_layers_cannot_be_paired():
    """Layers of different sizes > 1 have no defined pairing."""
    bad = pd.DataFrame({"alert_id": [1] * 5, "acct_id": [1, 2, 3, 4, 5], "is_main": [False] * 5,
                        "planned_layer": pd.array([0, 0, 1, 1, 1], dtype="Int64")})
    with pytest.raises(ValueError, match="cannot be paired"):
        planned_edges(bad)


@pytest.mark.parametrize("with_layer", [True, False])
def test_load_alert_members_planned_layer(tmp_path, with_layer):
    """Layer -1 and older files without the column both give <NA>; file order is kept."""
    df = pd.DataFrame({"alertID": [0, 0, 1], "reason": ["mule_fan_in_out"] * 2 + ["cycle"],
                       "accountID": [7, 3, 5], "isMain": ["true", "false", "true"]})
    if with_layer:
        df["layer"] = [1, 0, -1]
    df.to_csv(tmp_path / "alert_members.csv", index=False)
    m = amlsim.load_alert_members(tmp_path)
    assert m.acct_id.tolist() == [7, 3, 5]
    expected = [1, 0, None] if with_layer else [None, None, None]
    assert m.planned_layer.tolist() == [pd.NA if v is None else v for v in expected]
