"""Point-in-time account features: at day t only transactions on or before day t are used.

Windows are in days, (t - w, t] with an end-of-day cut-off. Input columns come from
leakage.input_columns, so labels and simulator-only fields cannot enter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.data import amlsim
from earlymule.data.leakage import check_feature_columns, input_columns


def load_transactions(run_dir: Path, base_date: pd.Timestamp) -> pd.DataFrame:
    """All transfers (normal and illicit, as a bank would see them): orig_acct, bene_acct, base_amt, day.

    Sorted by day so snapshots can slice by position.
    """
    cols = [c for c in input_columns("transactions.csv") if c != "tran_id"]
    tx = pd.read_csv(Path(run_dir) / "transactions.csv", usecols=cols)
    tx["day"] = amlsim.to_day(tx.pop("tran_timestamp"), base_date)
    return tx.sort_values("day", kind="stable").reset_index(drop=True)


def _side(tx: pd.DataFrame, acct_col: str, peer_col: str, acct_ids: pd.Index, prefix: str) -> pd.DataFrame:
    """Count, amount and distinct peers per account on one side of the transfers."""
    g = tx.groupby(acct_col)
    out = pd.DataFrame({
        f"n_{prefix}": g.size(),
        f"amt_{prefix}": g.base_amt.sum(),
        f"peers_{prefix}": g[peer_col].nunique(),
    })
    return out.reindex(acct_ids, fill_value=0)


def snapshot_features(tx: pd.DataFrame, acct_ids: pd.Index, t: int, windows: tuple[int, ...]) -> pd.DataFrame:
    """Features of every account at day t.

    Args:
        tx: load_transactions output (sorted by day).
        acct_ids: accounts to score.
        t: snapshot day; transactions after t are never read.
        windows: window lengths in days.
    Returns:
        DataFrame indexed by acct_id, float32.
    """
    # 1. Everything up to and including day t
    days = tx.day.to_numpy()
    past = tx.iloc[: np.searchsorted(days, t, side="right")]
    parts = []

    # 2. Window counts, amounts, peers and the out/in amount ratio
    for w in windows:
        win = past.iloc[np.searchsorted(past.day.to_numpy(), t - w, side="right"):]
        f = _side(win, "orig_acct", "bene_acct", acct_ids, "out").join(
            _side(win, "bene_acct", "orig_acct", acct_ids, "in"))
        f["out_in_ratio"] = f.amt_out / f.amt_in.where(f.amt_in > 0)
        parts.append(f.add_suffix(f"_{w}d"))

    # 3. Whole history: activity level and recency
    stacked = pd.concat([past[["orig_acct", "day"]].set_axis(["acct", "day"], axis=1),
                         past[["bene_acct", "day"]].set_axis(["acct", "day"], axis=1)])
    g = stacked.groupby("acct").day
    hist = pd.DataFrame({"n_tx_hist": g.size(), "days_since_first_tx": t - g.min(),
                         "days_since_last_tx": t - g.max()}).reindex(acct_ids)
    hist["n_tx_hist"] = hist.n_tx_hist.fillna(0)
    parts.append(hist)

    features = pd.concat(parts, axis=1).astype("float32")
    check_feature_columns(features.columns)
    return features
