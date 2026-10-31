"""Smoke tests for the conda env and the paths config. Run: pytest"""

import importlib

import pytest

from earlymule.config import load_paths, repo_root

CORE_LIBRARIES = ["numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "yaml", "matplotlib"]


@pytest.mark.parametrize("name", CORE_LIBRARIES)
def test_core_library_imports(name):
    """Each core library imports."""
    importlib.import_module(name)


def test_parquet_roundtrip(tmp_path):
    """pandas writes and reads parquet."""
    import pandas as pd

    df = pd.DataFrame({"acct_id": [1, 2], "onset_day": [3, 5]})
    df.to_parquet(tmp_path / "t.parquet", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "t.parquet"), df)


def test_lightgbm_fits_tiny_problem():
    """LightGBM trains and predicts on CPU."""
    import lightgbm as lgb
    import numpy as np

    X = np.random.default_rng(0).normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
    assert model.predict_proba(X).shape == (200, 2)


def test_paths_resolve_inside_repo():
    """Default relative paths resolve under the repo root."""
    paths = load_paths(local="configs/__none__.yaml")
    assert paths.processed == repo_root() / "data" / "processed"
    assert paths.dataset_dir.name == paths.dataset


def test_local_override_replaces_keys(tmp_path):
    """The local file overrides only the keys it defines."""
    local = tmp_path / "local.yaml"
    local.write_text("dataset: other-run\n", encoding="utf-8")
    paths = load_paths(local=local)
    assert paths.dataset == "other-run"
    assert paths.amlsim_outputs.name == "outputs"


def test_missing_key_is_reported(tmp_path):
    """A missing required key raises KeyError."""
    base = tmp_path / "paths.yaml"
    base.write_text("dataset: x\n", encoding="utf-8")
    with pytest.raises(KeyError, match="amlsim_outputs"):
        load_paths(base=base, local=tmp_path / "absent.yaml")


def test_dataset_folder_exists():
    """The selected AMLSim run has its main CSV files (skipped if not generated)."""
    d = load_paths().dataset_dir
    if not d.is_dir():
        pytest.skip(f"{d} not generated on this machine")
    for name in ["transactions.csv", "accounts.csv", "alert_accounts.csv", "alert_transactions.csv"]:
        assert (d / name).is_file(), name
