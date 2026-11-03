"""Tests for the walking skeleton parts: chronological split, snapshot labels, point-in-time features, metrics."""

import numpy as np
import pandas as pd
import pytest

from earlymule.data.splits import Split, chronological_splits, snapshot_days, snapshot_labels
from earlymule.evaluation.metrics import evaluate, recall_at_k
from earlymule.features.point_in_time import snapshot_features

TX = pd.DataFrame({
    "orig_acct": [1, 1, 2, 3, 1],
    "bene_acct": [2, 3, 1, 1, 2],
    "base_amt": [100.0, 50.0, 30.0, 20.0, 999.0],
    "day": [3, 9, 9, 10, 11],
})
ACCTS = pd.Index([1, 2, 3, 4], name="acct_id")


def test_chronological_splits_cover_all_days():
    """60/15/25 of 720 days gives contiguous blocks 0-431, 432-539, 540-719."""
    s = chronological_splits(720, (0.6, 0.15, 0.25))
    assert [(x.first_day, x.last_day) for x in s] == [(0, 431), (432, 539), (540, 719)]


def test_snapshot_days_keep_label_window_inside_split():
    """The first snapshot leaves a full label window and a full feature history."""
    assert snapshot_days(Split("val", 432, 539), horizon=7, min_history=30)[:2] == [438, 445]
    assert snapshot_days(Split("train", 0, 431), horizon=7, min_history=30)[0] == 30


def test_snapshot_labels():
    """Onset in (t-7, t] is positive, earlier onset is left out, later onset is negative."""
    onset = pd.Series([10.0, 3.0, 11.0], index=[1, 2, 3])
    label = snapshot_labels(onset, ACCTS, t=10, horizon=7)
    assert label[1] == 1.0 and np.isnan(label[2]) and label[3] == 0.0 and label[4] == 0.0


def test_window_boundaries():
    """Window (t-w, t]: a transfer on day t counts, one on day t-w does not."""
    f = snapshot_features(TX, ACCTS, t=10, windows=(1, 7))
    assert f.loc[1, "n_in_1d"] == 1          # day 10 from account 3
    assert f.loc[1, "n_out_7d"] == 1         # day 9 only; day 3 is outside (3, 10]
    assert f.loc[1, "out_in_ratio_7d"] == pytest.approx(50 / 50)
    assert f.loc[4, "n_tx_hist"] == 0 and np.isnan(f.loc[4, "days_since_last_tx"])


def test_features_ignore_transfers_after_t():
    """Adding transfers after day t leaves every feature at t unchanged."""
    before = snapshot_features(TX[TX.day.le(10)], ACCTS, t=10, windows=(1, 7, 30))
    after = snapshot_features(TX, ACCTS, t=10, windows=(1, 7, 30))
    pd.testing.assert_frame_equal(before, after)


def test_recall_at_k_ranks_within_each_snapshot():
    """Top-k is taken per snapshot, not over the pooled rows."""
    frame = pd.DataFrame({"snapshot": [1, 1, 1, 2, 2], "label": [1, 0, 0, 1, 0],
                          "score": [0.2, 0.9, 0.1, 0.05, 0.01], "typology": ["cycle", None, None, "m", None]})
    assert recall_at_k(frame, k=1) == 0.5
    m = evaluate(frame, k=1)
    assert m["positives"] == 2 and m["recall_at_1_by_typology"] == {"cycle": 0.0, "m": 1.0}
