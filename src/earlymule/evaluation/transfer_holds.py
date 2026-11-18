"""Holding transfers at payment initiation: hold rule, threshold choice and mule detection (second task).

A transfer is held when its score reaches a fixed threshold θ and its sender has not been held in the last
30 days (an account held and reviewed is not held again, as with end-of-day alerts). θ is chosen on
validation so that on average H new accounts are held per day. A mule is caught when one of its transfers is
held at or before its first illicit outflow, on or after its onset; holding the first illicit outflow itself
stops that money. Order within a day is seq (execution order); the hold history starts empty on the first
day of each split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from earlymule.evaluation.uncertainty import _case_counts, _interval


def holds(rows: pd.DataFrame, theta: float, cooldown: int = 30) -> pd.DataFrame:
    """First holds: transfers with score >= theta whose sender was not held in the previous `cooldown` days.

    Args:
        rows: seq, day, orig_acct, score of one split.
    Returns:
        The held rows, in execution order.
    """
    cand = rows[rows.score.ge(theta)].sort_values("seq")
    if cooldown <= 0:
        return cand
    # Walk each sender's candidates in order: a candidate within the cooldown of the last hold is skipped
    keep = np.zeros(len(cand), dtype=bool)
    for idx in cand.groupby("orig_acct").indices.values():
        days, last = cand.day.to_numpy()[idx], -np.inf
        for j, d in zip(idx, days):
            if d - last >= cooldown:
                keep[j], last = True, d
    return cand[keep]


def threshold_for_budget(rows: pd.DataFrame, per_day: float, cooldown: int = 30) -> float:
    """Score threshold whose holds average `per_day` new accounts a day on these rows (validation).

    Bisection over the sorted scores: fewer holds as θ rises. Returns the lowest θ with at most per_day.
    """
    n_days = rows.day.nunique()
    s = np.sort(rows.score.to_numpy())[::-1]
    # Upper index bound: more candidates than holds can ever need (each hold is one row)
    lo, hi = 0, min(len(s) - 1, int(per_day * n_days * 50))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if len(holds(rows, s[mid], cooldown)) / n_days <= per_day:
            lo = mid
        else:
            hi = mid
    return float(s[lo])


def first_outflows(illicit: pd.DataFrame, mules: pd.DataFrame) -> pd.DataFrame:
    """Each mule's first illicit outflow: seq, day and amount, with the mule's case, typology and onset.

    Args:
        illicit: alert transactions with tran_id (= seq), orig_acct, base_amt, day.
        mules: ground truth rows of mule candidates (acct_id, alert_id, typology, onset_day).
    """
    out = illicit[illicit.orig_acct.isin(mules.acct_id)].sort_values("tran_id").groupby("orig_acct").head(1)
    out = out.rename(columns={"orig_acct": "acct_id", "tran_id": "first_out_seq", "day": "first_out_day",
                              "base_amt": "first_out_amt"})[["acct_id", "first_out_seq", "first_out_day", "first_out_amt"]]
    return mules[["acct_id", "alert_id", "typology", "onset_day"]].merge(out, on="acct_id")


def hold_table(held: pd.DataFrame, first_out: pd.DataFrame, scored_seq: pd.Index) -> pd.DataFrame:
    """One row per mule whose first illicit outflow is a scored transfer of this split.

    Returns:
        first_out with hold_seq (first hold on or after onset, NaN if none) and the flags before_forwarding
        (held at or before the first illicit outflow), first_out_held (that transfer itself held),
        can_precede_forwarding (a day lies between onset and the first outflow, the end-of-day denominator)
        and same_day (first outflow on the onset day).
    """
    t = first_out[first_out.first_out_seq.isin(scored_seq)].copy()
    h = held[held.orig_acct.isin(t.acct_id)].merge(t[["acct_id", "onset_day"]], left_on="orig_acct", right_on="acct_id")
    first = h[h.day.ge(h.onset_day)].groupby("acct_id").seq.min()
    t["hold_seq"] = t.acct_id.map(first)
    t["before_forwarding"] = t.hold_seq.le(t.first_out_seq)
    t["first_out_held"] = t.hold_seq.eq(t.first_out_seq)
    t["can_precede_forwarding"] = t.first_out_day.gt(t.onset_day)
    t["same_day"] = ~t.can_precede_forwarding
    return t.reset_index(drop=True)


def share_ci(table: pd.DataFrame, hit: str, counted: str | None, n: int, seed: int) -> dict:
    """Share of counted mules with `hit`, and its 95% case-bootstrap interval."""
    c = _case_counts(table, hit, counted)
    draws = np.random.default_rng(seed).integers(0, len(c), (n, len(c)))
    n_draw, h_draw = c.n.to_numpy()[draws].sum(1), c.h.to_numpy()[draws].sum(1)
    point = c.h.sum() / max(c.n.sum(), 1)
    return {"share": round(float(point), 4), "mules": int(c.n.sum()),
            "ci95": _interval(np.where(n_draw > 0, h_draw / np.maximum(n_draw, 1), np.nan))}


def hold_metrics(rows: pd.DataFrame, held: pd.DataFrame, table: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Detection, cost and money metrics of one hold rule on one split.

    Args:
        rows: seq, day, label (1 = illicit outflow of a mule), illicit (any illicit transfer) of every scored row.
        held: holds(...) of the same rows.
        table: hold_table(...) of the same split.
    """
    legit = ~rows.illicit.astype(bool)
    held_legit = held.seq.isin(rows.seq[legit])
    stopped = table.first_out_amt[table.before_forwarding].sum() / table.first_out_amt.sum()
    by_typ = table.groupby("typology").before_forwarding.mean().round(4).to_dict()
    return {
        "holds_per_day": round(len(held) / rows.day.nunique(), 2),
        "false_holds_per_1000_legit": round(1000 * held_legit.sum() / legit.sum(), 3),
        "hold_precision_mule_illicit": round(float(held.seq.isin(rows.seq[rows.label.eq(1)]).mean()), 4) if len(held) else np.nan,
        "hold_precision_any_illicit": round(float(held.seq.isin(rows.seq[~legit]).mean()), 4) if len(held) else np.nan,
        "caught_all": share_ci(table, "before_forwarding", None, n_boot, seed),
        "caught_can_precede": share_ci(table, "before_forwarding", "can_precede_forwarding", n_boot, seed),
        "caught_same_day": share_ci(table, "before_forwarding", "same_day", n_boot, seed),
        "first_outflow_itself_held": round(float(table.first_out_held.mean()), 4),
        "first_outflow_amount_stopped": round(float(stopped), 4),
        "caught_by_typology": by_typ,
    }
