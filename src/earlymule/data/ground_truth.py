"""Per-account temporal ground truth (role, onset, completion) from AMLSim alert transactions.

Evaluation only: never join these columns into model features.
Run: python -m earlymule.data.ground_truth   (writes data/processed/ground_truth.parquet)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.data import amlsim

# Roles that receive and pass on money; provisional mule label until the advisor confirms it
MULE_ROLES = ("relay", "collector", "disperser")

COLUMNS = [
    "alert_id", "acct_id", "typology", "is_main", "role", "is_mule_candidate",
    "n_illicit_in", "n_illicit_out", "onset_day", "first_illicit_inflow_day",
    "first_illicit_outflow_day", "completion_day", "upstream_accounts", "downstream_accounts",
]


def assign_role(n_in: pd.Series, n_out: pd.Series) -> pd.Series:
    """Structural role from counts of illicit inflow and outflow events (checked in order)."""
    conditions = [
        (n_in > 0) & (n_out > 0),
        n_in > 1,
        n_out > 1,
        n_out == 1,
        n_in == 1,
    ]
    roles = ["relay", "collector", "disperser", "source", "destination"]
    return pd.Series(np.select(conditions, roles, default="no_illicit"), index=n_in.index)


def _side_stats(alert_tx: pd.DataFrame, acct_col: str, other_col: str, prefix: str) -> pd.DataFrame:
    """Per (alert_id, account): event count, first/last day and sorted counterparties on one side."""
    g = alert_tx.groupby(["alert_id", acct_col])
    out = g.agg(n=("tran_id", "size"), first=("day", "min"), last=("day", "max"))
    out["peers"] = g[other_col].agg(lambda s: sorted(set(s.astype(int))))
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out.rename_axis(["alert_id", "acct_id"])


def build_ground_truth(
    alert_accounts: pd.DataFrame,
    alert_tx: pd.DataFrame,
    alert_members: pd.DataFrame,
) -> pd.DataFrame:
    """One row per alert member with role and day milestones.

    Args:
        alert_accounts: from amlsim.load_alert_accounts.
        alert_tx: from amlsim.load_alert_transactions (needs `day`).
        alert_members: from amlsim.load_alert_members.
    Returns:
        DataFrame with COLUMNS. Days are Int64; a milestone that does not exist
        for the role (e.g. outflow of a collector) is <NA>, never 0.
    """
    # 1. Inflow and outflow sides; counterparties are restricted to the same alert
    inflow = _side_stats(alert_tx, "bene_acct", "orig_acct", "in")
    outflow = _side_stats(alert_tx, "orig_acct", "bene_acct", "out")

    # 2. Left join onto members so accounts without illicit events keep a row
    gt = (alert_accounts.rename(columns={"alert_type": "typology"})
          .set_index(["alert_id", "acct_id"]).join(inflow).join(outflow).reset_index())
    gt = gt.merge(alert_members, on=["alert_id", "acct_id"], how="left")
    gt["is_main"] = gt["is_main"].fillna(False).astype(bool)

    # 3. Counts and roles
    gt["n_illicit_in"] = gt["in_n"].fillna(0).astype("int64")
    gt["n_illicit_out"] = gt["out_n"].fillna(0).astype("int64")
    gt["role"] = assign_role(gt["n_illicit_in"], gt["n_illicit_out"])
    gt["is_mule_candidate"] = gt["role"].isin(MULE_ROLES)

    # 4. Milestones; same-day events have no known order, so only day-level min/max is used
    gt["first_illicit_inflow_day"] = gt["in_first"].astype("Int64")
    gt["first_illicit_outflow_day"] = gt["out_first"].astype("Int64")
    gt["onset_day"] = gt[["in_first", "out_first"]].min(axis=1).astype("Int64")
    # Completion = last illicit event in either direction; AMLSim has no cash-out to observe
    gt["completion_day"] = gt[["in_last", "out_last"]].max(axis=1).astype("Int64")

    # 5. Counterparty lists; empty list rather than missing
    for col, src in [("upstream_accounts", "in_peers"), ("downstream_accounts", "out_peers")]:
        gt[col] = [p if isinstance(p, list) else [] for p in gt[src]]

    return gt[COLUMNS].sort_values(["alert_id", "acct_id"]).reset_index(drop=True)


def summarize(gt: pd.DataFrame) -> pd.DataFrame:
    """Account counts by typology and role."""
    return pd.crosstab(gt["typology"], gt["role"], margins=True)


def main() -> None:
    """Build ground truth for the configured dataset, write parquet and print a summary."""
    # 1. Load the selected run
    paths = load_paths()
    base_date = amlsim.read_base_date(paths.amlsim_conf)
    gt = build_ground_truth(
        amlsim.load_alert_accounts(paths.dataset_dir),
        amlsim.load_alert_transactions(paths.dataset_dir, base_date),
        amlsim.load_alert_members(paths.dataset_tmp_dir),
    )

    # 2. Write and report
    paths.processed.mkdir(parents=True, exist_ok=True)
    out_file = paths.processed / "ground_truth.parquet"
    gt.to_parquet(out_file, index=False)
    cand = gt[gt["is_mule_candidate"]]
    print(f"{paths.dataset}: {len(gt)} alert accounts, {len(cand)} mule candidates -> {out_file}")
    print(summarize(gt).to_string())
    print("mule candidates with onset and completion:",
          int(cand[["onset_day", "completion_day"]].notna().all(axis=1).sum()))


if __name__ == "__main__":
    main()
