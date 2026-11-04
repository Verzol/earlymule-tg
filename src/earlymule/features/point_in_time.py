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


def pair_key(orig: pd.Series, bene: pd.Series) -> pd.Series:
    """One int64 key per directed account pair (account IDs are below 10^9)."""
    return orig.astype("int64") * 1_000_000_000 + bene.astype("int64")


def pair_first_day(tx: pd.DataFrame) -> pd.Series:
    """First day each directed pair transacted, indexed by pair_key.

    Safe to compute once over all days: it is only looked up for pairs seen on or before t,
    whose first day is then also on or before t.
    """
    return tx.groupby(pair_key(tx.orig_acct, tx.bene_acct)).day.min()


def snapshot_features(tx: pd.DataFrame, acct_ids: pd.Index, t: int, windows: tuple[int, ...],
                      first_day: pd.Series | None = None) -> pd.DataFrame:
    """Features of every account at day t.

    Args:
        tx: load_transactions output (sorted by day).
        acct_ids: accounts to score.
        t: snapshot day; transactions after t are never read.
        windows: window lengths in days.
        first_day: pair_first_day(tx); computed from transfers up to t when omitted.
    Returns:
        DataFrame indexed by acct_id, float32.
    """
    # 1. Everything up to and including day t
    days = tx.day.to_numpy()
    past = tx.iloc[: np.searchsorted(days, t, side="right")]
    if first_day is None:
        first_day = pair_first_day(past)
    parts = []

    # 2. Per window: counts, amounts, peers, out/in amount ratio, and peers never paid before the window
    for w in windows:
        win = past.iloc[np.searchsorted(past.day.to_numpy(), t - w, side="right"):]
        f = _side(win, "orig_acct", "bene_acct", acct_ids, "out").join(
            _side(win, "bene_acct", "orig_acct", acct_ids, "in"))
        f["out_in_ratio"] = f.amt_out / f.amt_in.where(f.amt_in > 0)
        new = win[pair_key(win.orig_acct, win.bene_acct).map(first_day).to_numpy() > t - w]
        f["new_peers_out"] = new.groupby("orig_acct").bene_acct.nunique().reindex(acct_ids, fill_value=0)
        f["new_peers_in"] = new.groupby("bene_acct").orig_acct.nunique().reindex(acct_ids, fill_value=0)
        f["new_share_out"] = f.new_peers_out / f.peers_out.where(f.peers_out > 0)
        f["new_share_in"] = f.new_peers_in / f.peers_in.where(f.peers_in > 0)
        parts.append(f.add_suffix(f"_{w}d"))

    # 3. Burst: share of the 30-day activity that fell in the last 7 days (7/30 = steady)
    if 7 in windows and 30 in windows:
        w7, w30 = parts[windows.index(7)], parts[windows.index(30)]
        for m in ["n_out", "n_in", "peers_out", "peers_in", "amt_out", "amt_in"]:
            parts.append((w7[f"{m}_7d"] / w30[f"{m}_30d"].where(w30[f"{m}_30d"] > 0)).rename(f"{m}_7d_over_30d"))

    # 4. Whole history: activity level and recency of each direction
    stacked = pd.concat([past[["orig_acct", "day"]].set_axis(["acct", "day"], axis=1).assign(out=True),
                         past[["bene_acct", "day"]].set_axis(["acct", "day"], axis=1).assign(out=False)])
    g = stacked.groupby("acct").day
    hist = pd.DataFrame({"n_tx_hist": g.size(), "days_since_first_tx": t - g.min(),
                         "days_since_last_tx": t - g.max(),
                         "days_since_last_out": t - stacked[stacked.out].groupby("acct").day.max(),
                         "days_since_last_in": t - stacked[~stacked.out].groupby("acct").day.max()}).reindex(acct_ids)
    hist["n_tx_hist"] = hist.n_tx_hist.fillna(0)
    parts.append(hist)

    features = pd.concat(parts, axis=1).astype("float32")
    check_feature_columns(features.columns)
    return features
