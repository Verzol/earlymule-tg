"""Transfer-time features for the second task: every outgoing transfer is scored just before it executes.

A transfer x (sender s, payee r, amount a, day d) sits at position p in the (day, seq) order of
load_transactions. Every value uses transfers before p only; x itself adds only its payee and amount. The
conservative variant cuts at the start of day d instead, so it needs no within-day order. Account context is
snapshot_features at the end of day d - 1, for the sender (prefix s_) and the payee (prefix r_).
Rows: transfers on days >= 31 whose sender has 20 transfers by the end of day d - 1 (the scored population).
Writes <processed>/transfer_features_<dataset>[_conservative].parquet.
Run: python -m earlymule.features.transfer_time [--dataset NAME] [--conservative]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.data import amlsim
from earlymule.data.export_events import regular_from_day
from earlymule.data.leakage import check_feature_columns, input_columns
from earlymule.features.point_in_time import load_transactions, pair_first_day, pair_key, snapshot_features

SHIFT = 2 ** 22  # positions and keys stay below this, so key * SHIFT + position is one sortable int64
KEYS = ["seq", "day", "orig_acct", "bene_acct"]
# Values that lean on simulator regularities (layered mules forward about 90% of what they receive, illicit
# amounts are small, regular accounts pay a few fixed partners): the "no shortcuts" variant drops them
SHORTCUTS = ["amt", "amt_over_in_since_last_out", "amt_over_last_in", "first_to_payee"]


class PriorIndex:
    """Count, sums, last position and running max of events per key, over events before a cut-off position."""

    def __init__(self, keys: np.ndarray, pos: np.ndarray, values: dict[str, np.ndarray]):
        order = np.lexsort((pos, keys))
        self.comp = keys[order].astype(np.int64) * SHIFT + pos[order]
        self.pos = pos[order]
        self.cum = {k: np.r_[0.0, np.cumsum(v[order], dtype=float)] for k, v in values.items()}
        self.runmax = {k: pd.Series(v[order]).groupby(keys[order]).cummax().to_numpy() for k, v in values.items()}

    def _range(self, qkeys: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index range of the events of each query key with position < cut."""
        base = qkeys.astype(np.int64) * SHIFT
        return np.searchsorted(self.comp, base), np.searchsorted(self.comp, base + cut)

    def count(self, qkeys, cut) -> np.ndarray:
        """Number of events before cut."""
        lo, hi = self._range(qkeys, cut)
        return (hi - lo).astype(float)

    def sum(self, name: str, qkeys, cut) -> np.ndarray:
        """Sum of one value over the events before cut."""
        lo, hi = self._range(qkeys, cut)
        return self.cum[name][hi] - self.cum[name][lo]

    def last_pos(self, qkeys, cut) -> np.ndarray:
        """Position of the last event before cut, -1 if none."""
        lo, hi = self._range(qkeys, cut)
        return np.where(hi > lo, self.pos[np.maximum(hi - 1, 0)], -1)

    def max(self, name: str, qkeys, cut) -> np.ndarray:
        """Largest value before cut, NaN if none."""
        lo, hi = self._range(qkeys, cut)
        return np.where(hi > lo, self.runmax[name][np.maximum(hi - 1, 0)], np.nan)


def load_sender_balance(run_dir: Path, seq: pd.Series) -> np.ndarray:
    """Sender's balance just before each transfer: oldbalanceOrig of its tx_log.csv row (seq is 1-based)."""
    log = pd.read_csv(Path(run_dir) / "tx_log.csv", usecols=["oldbalanceOrig"])
    return log.oldbalanceOrig.to_numpy()[seq.to_numpy() - 1]


def transfer_rows(tx: pd.DataFrame, regular_from: pd.Series, first_day: int = 31) -> np.ndarray:
    """Positions in tx of the scored transfers: day >= first_day and sender regular by the end of day d - 1."""
    reg = tx.orig_acct.map(regular_from).to_numpy(dtype=float)  # NaN (never regular) compares False
    return np.flatnonzero((tx.day.to_numpy() >= first_day) & (reg <= tx.day.to_numpy() - 1))


def _ratio(num, den) -> np.ndarray:
    """num / den with NaN where den is not positive."""
    den = np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def order_features(tx: pd.DataFrame, balance: np.ndarray, rows: np.ndarray, conservative: bool = False) -> pd.DataFrame:
    """Sender, order, pair and payee values of each row from transfers before it.

    Args:
        tx: load_transactions output; its positional index is the execution order.
        balance: sender's balance just before each transfer of tx (load_sender_balance), aligned with tx.
        rows: positions of the scored transfers.
        conservative: cut at the start of the transfer's day instead of at the transfer itself.
    Returns:
        DataFrame with one row per entry of rows (same order), float32 features.
    """
    # 1. Events: each transfer is an outflow of its sender and an inflow of its payee
    pos = np.arange(len(tx))
    s_all, r_all = tx.orig_acct.to_numpy(), tx.bene_acct.to_numpy()
    amt, day = tx.base_amt.to_numpy(dtype=float), tx.day.to_numpy()
    assert len(tx) < SHIFT and max(s_all.max(), r_all.max()) < SHIFT and (np.diff(day) >= 0).all()
    pk = pair_key(tx.orig_acct, tx.bene_acct)
    codes, uniq = pd.factorize(pk)
    # A pair's first transfer is "new" for the payee; known from earlier transfers only
    new_in = (pd.Series(pos).groupby(codes).transform("min").to_numpy() == pos).astype(float)
    one = np.ones(len(tx))
    out_ev = PriorIndex(s_all, pos, {"amt": amt, "one": one})
    in_ev = PriorIndex(r_all, pos, {"amt": amt, "one": one, "new_amt": amt * new_in})
    any_ev = PriorIndex(np.r_[s_all, r_all], np.r_[pos, pos], {"one": np.r_[one, one]})
    pair_ev = PriorIndex(codes, pos, {"amt": amt})

    # 2. Cut-offs: the transfer itself, or the start of its day
    s, r, a, d = s_all[rows], r_all[rows], amt[rows], day[rows]
    day_start = np.searchsorted(day, d)
    cut = day_start if conservative else rows
    f = {"amt": a}

    # 3. Sender, today before x (zero in the conservative variant by construction)
    for name, ev, col in [("s_in_amt_today", in_ev, "amt"), ("s_in_n_today", in_ev, "one"),
                          ("s_in_new_amt_today", in_ev, "new_amt"), ("s_out_n_today", out_ev, "one"),
                          ("s_out_amt_today", out_ev, "amt")]:
        f[name] = ev.sum(col, s, cut) - ev.sum(col, s, day_start)

    # 4. Order against the sender's own history
    last_out = out_ev.last_pos(s, cut)
    in_since_out = in_ev.sum("amt", s, cut) - np.where(last_out >= 0, in_ev.sum("amt", s, last_out + 1), 0.0)
    f["amt_over_in_since_last_out"] = _ratio(a, in_since_out)
    # Balance before x is what the bank sees; the conservative variant backs out today's earlier transfers,
    # giving the end-of-day d - 1 balance (also known at d - 1)
    net_today = (in_ev.sum("amt", s, rows) - in_ev.sum("amt", s, day_start)
                 - out_ev.sum("amt", s, rows) + out_ev.sum("amt", s, day_start))
    bal = balance[rows] - (net_today if conservative else 0.0)
    f["amt_over_balance"] = _ratio(a, bal)
    n_out = out_ev.count(s, cut)
    f["amt_over_mean_out"] = _ratio(a, _ratio(out_ev.sum("amt", s, cut), n_out))
    f["amt_over_max_out"] = _ratio(a, out_ev.max("amt", s, cut))
    last_in = in_ev.last_pos(s, cut)
    has_in = last_in >= 0
    f["amt_over_last_in"] = np.where(has_in, _ratio(a, amt[np.maximum(last_in, 0)]), np.nan)
    f["days_since_last_in"] = np.where(has_in, d - day[np.maximum(last_in, 0)], np.nan)
    f["tx_since_last_in"] = np.where(has_in, any_ev.count(s, cut) - any_ev.count(s, last_in + 1), np.nan)

    # 5. Pair: earlier transfers s -> r, and r -> s (codes of reverse pairs that never occur stay -1)
    pair = codes[rows]
    f["pair_n_before"] = pair_ev.count(pair, cut)
    f["pair_amt_before"] = pair_ev.sum("amt", pair, cut)
    f["first_to_payee"] = (f["pair_n_before"] == 0).astype(float)
    rev = pd.Index(uniq).get_indexer(pair_key(pd.Series(r), pd.Series(s)))
    f["reverse_n_before"] = np.where(rev >= 0, pair_ev.count(np.maximum(rev, 0), cut), 0.0)

    # 6. Payee before x: history counts and today's activity
    f["r_in_n_before"] = in_ev.count(r, cut)
    f["r_out_n_before"] = out_ev.count(r, cut)
    f["r_in_n_today"] = in_ev.count(r, cut) - in_ev.count(r, day_start)
    f["r_in_amt_today"] = in_ev.sum("amt", r, cut) - in_ev.sum("amt", r, day_start)
    f["r_out_n_today"] = out_ev.count(r, cut) - out_ev.count(r, day_start)
    return pd.DataFrame(f).astype("float32")


def daily_context(tx: pd.DataFrame, rows: np.ndarray, windows: tuple[int, ...],
                  first_day: pd.Series | None = None) -> pd.DataFrame:
    """Sender (s_) and payee (r_) point-in-time features at the end of day d - 1, one row per entry of rows."""
    sel = tx.iloc[rows][["orig_acct", "bene_acct", "day"]].reset_index(drop=True)
    first_day = pair_first_day(tx) if first_day is None else first_day
    parts, start = [], time.time()
    for i, (d, grp) in enumerate(sel.groupby("day", sort=True)):
        if i % 100 == 0:
            print(f"daily context: day {d}, {time.time() - start:.0f} s", flush=True)
        # snapshot_features reads transfers up to day d - 1 only
        ids = pd.Index(np.unique(np.r_[grp.orig_acct.to_numpy(), grp.bene_acct.to_numpy()]))
        f = snapshot_features(tx, ids, int(d) - 1, windows, first_day)
        both = pd.concat([f.reindex(grp.orig_acct).add_prefix("s_").reset_index(drop=True),
                          f.reindex(grp.bene_acct).add_prefix("r_").reset_index(drop=True)], axis=1)
        parts.append(both.set_index(grp.index))
    return pd.concat(parts).sort_index()


def transfer_features(tx: pd.DataFrame, balance: np.ndarray, rows: np.ndarray, windows=(1, 7, 30),
                      conservative: bool = False, first_day: pd.Series | None = None,
                      deposit: pd.Series | None = None) -> pd.DataFrame:
    """Keys (seq, day, orig_acct, bene_acct) and all features of the scored transfers."""
    keys = tx.iloc[rows][KEYS].reset_index(drop=True)
    parts = [keys, order_features(tx, balance, rows, conservative), daily_context(tx, rows, windows, first_day)]
    if deposit is not None:
        parts.append(pd.DataFrame({"s_initial_deposit": deposit.reindex(keys.orig_acct).to_numpy(dtype="float32")}))
    out = pd.concat(parts, axis=1)
    check_feature_columns([c for c in out.columns if c not in KEYS])
    return out


def risk_features(keys: pd.DataFrame, tx: pd.DataFrame, scores: pd.DataFrame, hot_share: float = 0.01) -> pd.DataFrame:
    """Payee risk score and risk propagation to the sender, both from daily account scores of day d - 1.

    Args:
        keys: seq, day, orig_acct, bene_acct of the scored transfers.
        scores: acct_id, snapshot, score of the daily model; on train days out-of-fold (oof_scores), so a
            score never comes from a model that saw the account's case.
        hot_share: accounts in this top share of a day's scores count as high-risk senders.
    Returns:
        r_risk (payee's score at d - 1) and hot_in_share_7d (share of the sender's inflow over days d - 7 to
        d - 1 that came from accounts high-risk at d - 1), aligned with keys.
    """
    # 1. Payee score of the previous day; NaN when the payee was not scored (fewer than 20 transfers)
    s = scores.set_index(["acct_id", "snapshot"]).score
    r_risk = s.reindex(pd.MultiIndex.from_arrays([keys.bene_acct, keys.day - 1])).to_numpy()

    # 2. High-risk accounts per day: top share of that day's scores (ranks, since score scales differ
    #    between the out-of-fold train days and the validation and test days)
    rank = scores.groupby("snapshot").score.rank(ascending=False, pct=True)
    hot = scores[rank.le(hot_share).to_numpy()].groupby("snapshot").acct_id.apply(set)

    # 3. Per day d: inflow of each account over days d - 7 .. d - 1, and the part from high-risk senders
    share = np.full(len(keys), np.nan)
    days = tx.day.to_numpy()
    for d, idx in keys.groupby("day").indices.items():
        win = tx.iloc[np.searchsorted(days, d - 7):np.searchsorted(days, d)]
        hot_d = hot.get(d - 1, set())
        total = win.groupby("bene_acct").base_amt.sum()
        from_hot = win[win.orig_acct.isin(hot_d)].groupby("bene_acct").base_amt.sum().reindex(total.index, fill_value=0)
        share[idx] = (from_hot / total).reindex(keys.orig_acct.to_numpy()[idx]).to_numpy()
    return pd.DataFrame({"r_risk": r_risk, "hot_in_share_7d": share}).astype("float32")


def main() -> None:
    """Build and save the transfer features of the configured dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--conservative", action="store_true")
    args = parser.parse_args()
    start = time.time()
    paths = load_paths()
    paths = paths.with_dataset(args.dataset) if args.dataset else paths

    # 1. Inputs a bank has: transfers in execution order, balances, opening deposits
    tx = load_transactions(paths.dataset_dir, amlsim.read_base_date(paths.amlsim_conf))
    balance = load_sender_balance(paths.dataset_dir, tx.seq)
    deposit = pd.read_csv(paths.dataset_dir / "accounts.csv",
                          usecols=input_columns("accounts.csv")).set_index("acct_id").initial_deposit
    # 2. Scored transfers and their features
    rows = transfer_rows(tx, regular_from_day(tx, 20))
    feats = transfer_features(tx, balance, rows, conservative=args.conservative, deposit=deposit)
    out = paths.processed / f"transfer_features_{paths.dataset}{'_conservative' if args.conservative else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"{len(feats):,} transfers x {feats.shape[1] - len(KEYS)} features -> {out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
