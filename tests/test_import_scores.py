"""Tests for evaluating scores of models trained elsewhere: labels attached locally, stale exports rejected."""

import numpy as np
import pandas as pd
import pytest

from earlymule.data.splits import active_labels
from earlymule.evaluation.daily import attach_labels
from earlymule.evaluation.import_scores import check_val_labels

MULES = pd.DataFrame({"onset_day": [3, 5], "completion_day": [4, 9], "typology": ["M1", "cycle"]},
                     index=pd.Index([1, 2], name="acct_id"))


def test_attach_labels_matches_active_labels():
    """Row labels equal the daily label of splits.active_labels on every day; typology only on positives."""
    rows = pd.DataFrame([(a, t) for t in range(12) for a in [1, 2, 3]], columns=["acct_id", "snapshot"])
    out = attach_labels(rows.assign(score=0.0), MULES)
    accts = pd.Index([1, 2, 3])
    for t in range(12):
        want = active_labels(MULES.onset_day.astype(float), MULES.completion_day.astype(float), accts, t)
        got = out[out.snapshot.eq(t)].set_index("acct_id").label
        pd.testing.assert_series_equal(got, want, check_names=False, check_index_type=False)
    assert out.typology.notna().eq(out.label.eq(1)).all()


def test_stale_validation_labels_are_rejected():
    """Validation labels shipped with the run must equal the local ones."""
    rows = pd.DataFrame({"acct_id": [1, 1, 3], "snapshot": [3, 5, 3], "score": 0.0})
    local = attach_labels(rows, MULES)
    check_val_labels(local.assign(label=local.label), local)
    with pytest.raises(ValueError):
        check_val_labels(local.assign(label=[0.0, np.nan, 0.0]), local)
