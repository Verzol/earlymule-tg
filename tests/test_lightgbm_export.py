"""Tests for the feature-family ablation of the daily LightGBM trained from an export (models.lightgbm_export)."""

import numpy as np
import pandas as pd
import pytest

from earlymule.features.point_in_time import snapshot_features
from earlymule.models.lightgbm_export import feature_families, kept_features, load_split, train_mask
from earlymule.models.tabular import load_config

CFG = load_config("lightgbm_ablation.yaml")


def model_columns() -> list[str]:
    """The export's feature columns in order, from the real feature code on a tiny transfer table."""
    tx = pd.DataFrame({"orig_acct": [1, 2], "bene_acct": [2, 3], "base_amt": [10.0, 9.0], "day": [1, 2]})
    f = snapshot_features(tx, pd.Index([1, 2, 3], name="acct_id"), t=2, windows=tuple(CFG["windows"]))
    return [c for c in f.columns if c not in CFG["exclude_features"]] + ["initial_deposit"]


def test_every_feature_has_exactly_one_family():
    """All 73 model features map to one family; the 13 families are all used."""
    cols = model_columns()
    family = feature_families(cols, CFG["families"])
    assert len(cols) == 73 and set(family.values()) == set(CFG["families"])
    with pytest.raises(ValueError):
        feature_families(cols + ["unknown_feature"], CFG["families"])


def test_ablations_drop_exactly_their_columns():
    """Each family ablation drops that family only; window ablations drop the window and the 7d/30d ratios."""
    cols = model_columns()
    family = feature_families(cols, CFG["families"])
    assert kept_features(cols, CFG, "full") == cols
    kept = kept_features(cols, CFG, "no_churn")
    assert set(cols) - set(kept) == {c for c in cols if family[c] == "churn"}
    kept = kept_features(cols, CFG, "no_window_7d")
    assert not any(c.endswith("_7d") or c.endswith("_7d_over_30d") for c in kept)
    assert any(c.endswith("_30d") for c in kept) and any(c.endswith("_1d") for c in kept)
    dropped = set(cols) - set(kept_features(cols, CFG, "no_shortcuts"))
    assert {family[c] for c in dropped} == {"churn", "pass_through", "rapid_forward"}
    assert len(CFG["ablations"]) == 18


def test_train_mask_keeps_positives_and_repeats_per_seed():
    """Every positive is kept; the negative sample depends only on the seed."""
    label = pd.Series([1.0] * 50 + [0.0] * 20000)
    a = train_mask(label, 0.01, np.random.default_rng(0))
    b = train_mask(label, 0.01, np.random.default_rng(0))
    c = train_mask(label, 0.01, np.random.default_rng(1))
    assert a[:50].all() and (a == b).all() and not (a == c).all()
    assert 100 < a[50:].sum() < 320


def test_load_split_pairs_rows_and_features(tmp_path):
    """Rows and features are joined by position after an order check; deposit comes from nodes."""
    rows = pd.DataFrame({"acct_id": [5, 6], "snapshot": [3, 3], "label": [0.0, 1.0], "node": [0, 1]})
    feats = pd.DataFrame({"acct_id": [5, 6], "snapshot": [3, 3], "n_out_1d": [1.0, 2.0]})
    rows.to_parquet(tmp_path / "rows_val.parquet")
    feats.to_parquet(tmp_path / "features_val.parquet")
    out = load_split(tmp_path, "val", pd.Series([100.0, 200.0], index=[5, 6]))
    assert out.columns.tolist() == ["acct_id", "snapshot", "label", "n_out_1d", "initial_deposit"]
    assert out.initial_deposit.tolist() == [100.0, 200.0]
    feats.iloc[::-1].to_parquet(tmp_path / "features_val.parquet")
    with pytest.raises(ValueError):
        load_split(tmp_path, "val", pd.Series([100.0, 200.0], index=[5, 6]))
