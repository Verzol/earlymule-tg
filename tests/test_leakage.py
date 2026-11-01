"""Tests for the input / evaluation-only column split. Run: pytest tests/test_leakage.py"""

import pandas as pd
import pytest

from earlymule.config import load_paths
from earlymule.data.leakage import (
    EVAL_ONLY_COLUMNS,
    INPUT_COLUMNS,
    LeakageError,
    check_feature_columns,
    input_columns,
)


def test_input_and_eval_only_do_not_overlap():
    """No input column is also marked evaluation-only."""
    for cols in INPUT_COLUMNS.values():
        assert not {c.lower() for c in cols} & EVAL_ONLY_COLUMNS


@pytest.mark.parametrize("col", ["is_sar", "alert_id", "prior_sar_count", "onset_day", "role", "IS_SAR"])
def test_eval_only_feature_is_rejected(col):
    """Label columns and ground-truth columns cannot be features, whatever the case."""
    with pytest.raises(LeakageError):
        check_feature_columns(["n_in_7d", col])


def test_key_feature_is_rejected():
    """acct_id is a join key, not a feature."""
    with pytest.raises(LeakageError, match="acct_id"):
        check_feature_columns(["acct_id"])


def test_behaviour_features_pass():
    """Ordinary feature names are accepted."""
    check_feature_columns(["n_in_7d", "sum_out_30d", "initial_deposit"])


def test_eval_only_file_is_rejected():
    """Feature code cannot load the alert files."""
    with pytest.raises(LeakageError):
        input_columns("alert_transactions.csv")


def test_input_columns_exist_in_run():
    """Every allowed column exists in the selected AMLSim run (skipped if not generated)."""
    d = load_paths().dataset_dir
    if not d.is_dir():
        pytest.skip(f"{d} not generated on this machine")
    for file_name, cols in INPUT_COLUMNS.items():
        header = pd.read_csv(d / file_name, nrows=0).columns
        assert set(cols) <= set(header), file_name
