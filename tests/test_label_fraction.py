"""Tests for label scarcity: whole cases keep or lose labels, nested sets per seed, relabel modes, left-out typology."""

import numpy as np
import pandas as pd
import pytest

from earlymule.data.label_fraction import kept_mules, leave_out, relabel, train_mule_cases


def cases(n_cases: int = 40) -> pd.DataFrame:
    """Case map with 1-3 mules per case."""
    rng = np.random.default_rng(0)
    alert = np.repeat(np.arange(n_cases), rng.integers(1, 4, n_cases))
    return pd.DataFrame({"acct_id": np.arange(len(alert)) + 100, "alert_id": alert})


def test_kept_mules_take_whole_cases_and_are_nested():
    """Kept mules form whole cases, sets grow with the fraction, and at least one case is kept."""
    c = cases()
    sets = {f: set(kept_mules(c, f, seed=1)) for f in [0.01, 0.05, 0.25, 1.0]}
    for f, kept in sets.items():
        alerts = set(c.alert_id[c.acct_id.isin(kept)])
        assert kept == set(c.acct_id[c.alert_id.isin(alerts)])
    assert len(set(c.alert_id[c.acct_id.isin(sets[0.01])])) == 1
    assert sets[0.01] <= sets[0.05] <= sets[0.25] <= sets[1.0] == set(c.acct_id)
    assert set(kept_mules(c, 0.25, seed=1)) == sets[0.25]
    assert set(kept_mules(c, 0.25, seed=2)) != sets[0.25]
    with pytest.raises(ValueError):
        kept_mules(c, 0.0, seed=1)


def test_relabel_negative_or_drop():
    """Positives of unkept mules become 0 or NaN; kept positives and negatives stay as they are."""
    label = np.array([1, 1, 0, 1, 0])
    acct = np.array([1, 2, 2, 3, 4])
    assert relabel(label, acct, np.array([1]), "negative").tolist() == [1, 0, 0, 0, 0]
    out = relabel(label, acct, np.array([1]), "drop")
    assert out[0] == 1 and np.isnan(out[[1, 3]]).all() and (out[[2, 4]] == 0).all()
    with pytest.raises(ValueError):
        relabel(label, acct, np.array([1]), "ignore")


def test_train_mule_cases_needs_one_case_per_mule():
    """Every training mule maps to its case; a mule without a case is rejected."""
    rows = pd.DataFrame({"acct_id": [1, 1, 2, 3], "label": [1, 0, 1, 0]})
    gt = pd.DataFrame({"acct_id": [1, 2, 9], "alert_id": [7, 8, 9], "typology": ["cycle", "relay", "cycle"]})
    assert train_mule_cases(rows, gt).to_dict("list") == {"acct_id": [1, 2], "alert_id": [7, 8],
                                                          "typology": ["cycle", "relay"]}
    with pytest.raises(ValueError):
        train_mule_cases(rows, gt[gt.acct_id != 2])


def test_leave_out_takes_one_typology_and_rejects_unknown_names():
    """Only the mules of the named typology leave; an unknown name or a map without typology is an error."""
    c = pd.DataFrame({"acct_id": [1, 2, 3], "alert_id": [7, 7, 8], "typology": ["cycle", "cycle", "relay"]})
    assert leave_out(c, "cycle").tolist() == [1, 2]
    with pytest.raises(ValueError):
        leave_out(c, "fan_in")
    with pytest.raises(ValueError):
        leave_out(c.drop(columns="typology"), "cycle")
