"""Which AMLSim columns may be model input; reject features built from evaluation-only columns.

Full table with reasons: reports/column_classification.md.
"""

from __future__ import annotations

from collections.abc import Iterable

from earlymule.data.ground_truth import COLUMNS as GROUND_TRUTH_COLUMNS

# Columns feature code may read, per output file. Keys are kept for joins only.
INPUT_COLUMNS = {
    "accounts.csv": ["acct_id", "initial_deposit"],
    "transactions.csv": ["tran_id", "orig_acct", "bene_acct", "base_amt", "tran_timestamp"],
    "tx_log.csv": ["step", "amount", "nameOrig", "nameDest",
                   "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
}

# Join keys: allowed in a feature table index, never as a feature value.
# Low acct_id values are more active and more often alert members (ROC-AUC 0.64).
KEY_COLUMNS = {"acct_id", "tran_id", "orig_acct", "bene_acct", "nameorig", "namedest"}

# Simulator labels and anything derived from them
EVAL_ONLY_COLUMNS = {
    "is_sar", "issar", "alert_id", "alertid", "alert_type", "prior_sar_count",
    "is_main", "ismain", "model_id", "schedule_id", "sar",
} | {c.lower() for c in GROUND_TRUTH_COLUMNS if c != "acct_id"}

# Files with labels or generator internals; feature code must not open them
EVAL_ONLY_FILES = {
    "alert_accounts.csv", "alert_transactions.csv", "sar_accounts.csv", "tx_count.csv",
    "alert_members.csv", "normal_models.csv",
}


class LeakageError(ValueError):
    """A feature uses a column or file that the model must not see."""


def input_columns(file_name: str) -> list[str]:
    """Columns of an AMLSim output file that feature code may load."""
    if file_name in EVAL_ONLY_FILES:
        raise LeakageError(f"{file_name} is evaluation-only")
    if file_name not in INPUT_COLUMNS:
        raise LeakageError(f"{file_name} has no input columns")
    return list(INPUT_COLUMNS[file_name])


def check_feature_columns(columns: Iterable[str]) -> None:
    """Raise LeakageError if any feature column is a key or an evaluation-only column."""
    bad = sorted(c for c in columns if c.lower() in EVAL_ONLY_COLUMNS | KEY_COLUMNS)
    if bad:
        raise LeakageError(f"forbidden feature columns: {bad}")
