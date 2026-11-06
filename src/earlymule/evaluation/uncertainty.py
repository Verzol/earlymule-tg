"""95% intervals by case bootstrap: whole laundering cases are resampled, because mules of one case move together.

Scores and alerts are fixed; only the mules, grouped by case, are resampled. Negatives, including the
non-mule members of a case (hard negatives), stay fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from earlymule.evaluation.metrics import rank_by_score


def average_precision_fast(pos: np.ndarray, neg_sorted: np.ndarray) -> float:
    """Average precision from positive scores and ascending negative scores, as sklearn computes it.

    Tied scores form one threshold: precision there counts every row scored at or above it.
    LightGBM scores are piecewise constant, so ties are common.
    """
    # 1. Distinct positive scores, highest first, with the positives at or above each
    values, counts = np.unique(pos, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    tp = np.cumsum(counts)
    # 2. Negatives at or above each threshold, then recall steps times precision
    fp = len(neg_sorted) - np.searchsorted(neg_sorted, values, side="left")
    return float(np.sum(counts / len(pos) * tp / (tp + fp)))


def _interval(values: np.ndarray) -> list[float]:
    """2.5th and 97.5th percentiles, rounded; NaN draws are skipped."""
    return [round(float(v), 4) for v in np.nanpercentile(values, [2.5, 97.5])]


def bootstrap_ranking(frame: pd.DataFrame, case_of: pd.Series, k: int, n: int, seed: int) -> dict:
    """Intervals for PR-AUC and Recall@K (top k within each snapshot).

    Args:
        frame: rows with acct_id, snapshot, label, score.
        case_of: alert_id per mule acct_id.
    """
    # 1. Fixed parts: sorted negatives and, per positive row, whether it made its snapshot's top k
    ranked = rank_by_score(frame)
    in_top = ranked.groupby("snapshot").cumcount().lt(k)
    pos = ranked[ranked.label.eq(1)].assign(top=in_top[ranked.label.eq(1)], case=lambda d: d.acct_id.map(case_of))
    neg_sorted = np.sort(ranked.score[ranked.label.eq(0)].to_numpy())
    groups = [g for _, g in pos.groupby("case")]

    # 2. Resample cases with replacement
    rng = np.random.default_rng(seed)
    ap, rec = [], []
    for _ in range(n):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        ap.append(average_precision_fast(sample.score.to_numpy(), neg_sorted))
        rec.append(sample.top.mean())
    return {"pr_auc_ci95": _interval(np.array(ap)), f"recall_at_{k}_ci95": _interval(np.array(rec))}


# Detection shares of detection_table: (output name, hit column, column marking the mules counted)
DETECTION_SHARES = [("detected_before_forwarding", "before_forwarding", "can_precede_forwarding"),
                    ("detected_by_completion", "by_completion", None),
                    ("detected_before_completion", "before_completion", None)]


def _case_counts(table: pd.DataFrame, hit: str, counted: str | None) -> pd.DataFrame:
    """Per case: mules counted and mules hit, so a resample is a weighted sum over cases."""
    t = table.assign(n=table[counted].astype(int) if counted else 1)
    t["h"] = (t[hit] & t.n.astype(bool)).astype(int)
    return t.groupby("alert_id")[["n", "h"]].sum()


def bootstrap_detection(table: pd.DataFrame, n: int, seed: int) -> dict:
    """Intervals for the detection shares of detection_table, resampling whole cases.

    Alerts are fixed (computed once from the scores); only the set of mule cases is resampled.
    """
    rng = np.random.default_rng(seed)
    cases = table.alert_id.unique()
    draws = rng.integers(0, len(cases), (n, len(cases)))
    out = {}
    for name, hit, counted in DETECTION_SHARES:
        c = _case_counts(table, hit, counted).reindex(cases)
        n_draw, h_draw = c.n.to_numpy()[draws].sum(1), c.h.to_numpy()[draws].sum(1)
        # A draw without any counted mule has no share; it is skipped
        out[f"{name}_ci95"] = _interval(np.where(n_draw > 0, h_draw / np.maximum(n_draw, 1), np.nan))
    return out


def paired_bootstrap_difference(a: pd.DataFrame, b: pd.DataFrame, n: int, seed: int,
                                hit: str = "before_forwarding", counted: str | None = "can_precede_forwarding") -> dict:
    """Difference of one detection share between two models, with a 95% paired case-bootstrap interval.

    Args:
        a, b: detection_table of each model on the same split, population, K and cooldown.
    Returns:
        difference a - b and its interval; a beats b on this run when the interval lies above 0.
    """
    # 1. Same mules on both sides; each draw uses the same cases for both models
    if set(a.acct_id) != set(b.acct_id):
        raise ValueError("detection tables cover different mules")
    ca = _case_counts(a, hit, counted)
    cb = _case_counts(b, hit, counted).reindex(ca.index)
    draws = np.random.default_rng(seed).integers(0, len(ca), (n, len(ca)))
    # 2. Shares per draw, then the spread of their difference
    n_draw = ca.n.to_numpy()[draws].sum(1)
    diff = np.where(n_draw > 0, (ca.h.to_numpy()[draws].sum(1) - cb.h.to_numpy()[draws].sum(1)) / np.maximum(n_draw, 1),
                    np.nan)
    point = (ca.h.sum() - cb.h.sum()) / max(ca.n.sum(), 1)
    return {"difference": round(float(point), 4), "difference_ci95": _interval(diff)}
