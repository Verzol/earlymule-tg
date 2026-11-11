"""Tests for the Kaggle export: regular population, split rows and the exported tables."""

import numpy as np
import pandas as pd

from earlymule.data.export_events import build_export, regular_from_day
from earlymule.features.point_in_time import snapshot_features


def random_tx(n: int = 300, accounts: int = 12, days: int = 20, seed: int = 0) -> pd.DataFrame:
    """Random transfers sorted by day, self-transfers included."""
    rng = np.random.default_rng(seed)
    tx = pd.DataFrame({"orig_acct": rng.integers(1, accounts + 1, n), "bene_acct": rng.integers(1, accounts + 1, n),
                       "base_amt": rng.uniform(1, 1000, n), "day": rng.integers(0, days, n)})
    return tx.sort_values("day", kind="stable").reset_index(drop=True)


def test_regular_from_day_matches_n_tx_hist():
    """Regular at t by the export rule exactly when the n_tx_hist feature at t reaches the minimum."""
    tx, ids = random_tx(), pd.Index(range(1, 14))
    start = regular_from_day(tx, 20).reindex(ids)
    for t in range(20):
        hist = snapshot_features(tx, ids, t, windows=(1,)).n_tx_hist
        assert (hist.ge(20) == start.le(t)).all(), t


def test_export_tables():
    """Dense node index, all events, train without unlabelled rows, finished mules kept in val, no test labels."""
    tx = random_tx(accounts=6, days=20)
    accounts = pd.DataFrame({"initial_deposit": np.arange(6.0)}, index=pd.Index(range(1, 7), name="acct_id"))
    # Mule 1 is active on days 3-5 (train), mule 2 on days 12-14 (val) and finished from day 15
    mules = pd.DataFrame({"onset_day": [3, 12], "completion_day": [5, 14]}, index=pd.Index([1, 2], name="acct_id"))
    cfg = {"total_days": 20, "split_fractions": [0.6, 0.2, 0.2], "regular_min_tx": 2, "min_history_days": 1,
           "windows": [1, 7], "exclude_features": ["n_tx_hist"]}
    out = build_export({"tx": tx, "accounts": accounts, "mules": mules}, pd.Timestamp("2017-01-01", tz="UTC"), cfg)

    assert len(out["events"]) == len(tx) and out["nodes"].node.tolist() == list(range(6))
    assert out["events"].dow.iloc[0] == (6 + tx.day.iloc[0]) % 7  # 2017-01-01 was a Sunday
    train, val, test = out["rows_train"], out["rows_val"], out["rows_test"]
    assert train.snapshot.between(1, 11).all() and train.label.notna().all()
    assert set(train[train.label.eq(1)].snapshot[train.acct_id.eq(1)]) <= {3, 4, 5}
    assert val[val.acct_id.eq(2) & val.snapshot.eq(15)].label.isna().all()
    assert "label" not in test.columns and test.snapshot.between(16, 19).all()
    assert (out["nodes"].set_index("acct_id").node.reindex(test.acct_id).to_numpy() == test.node.to_numpy()).all()


def test_row_features_match_snapshot_features():
    """Each split's features are the point-in-time features of its rows, in row order, minus excluded columns."""
    tx = random_tx(accounts=6, days=20)
    accounts = pd.DataFrame({"initial_deposit": np.arange(6.0)}, index=pd.Index(range(1, 7), name="acct_id"))
    mules = pd.DataFrame({"onset_day": [3], "completion_day": [5]}, index=pd.Index([1], name="acct_id"))
    cfg = {"total_days": 20, "split_fractions": [0.6, 0.2, 0.2], "regular_min_tx": 2, "min_history_days": 1,
           "windows": [1, 7], "exclude_features": ["n_tx_hist"]}
    out = build_export({"tx": tx, "accounts": accounts, "mules": mules}, pd.Timestamp("2017-01-01", tz="UTC"), cfg)
    for split in ["train", "val", "test"]:
        rows, feats = out[f"rows_{split}"], out[f"features_{split}"]
        assert (feats[["acct_id", "snapshot"]].to_numpy() == rows[["acct_id", "snapshot"]].to_numpy()).all()
        assert "n_tx_hist" not in feats and "initial_deposit" not in feats and "label" not in feats
        # Spot-check one day against a direct snapshot
        t = int(rows.snapshot.iloc[-1])
        ids = pd.Index(rows.acct_id[rows.snapshot.eq(t)])
        direct = snapshot_features(tx, ids, t, (1, 7)).drop(columns="n_tx_hist")
        got = feats[feats.snapshot.eq(t)].set_index("acct_id")[direct.columns]
        pd.testing.assert_frame_equal(got, direct, check_names=False)
