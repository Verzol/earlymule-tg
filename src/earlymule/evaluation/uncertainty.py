"""95% intervals by case bootstrap: whole laundering cases are resampled, because mules of one case move together.

Negatives are kept fixed; only the positive rows (grouped by case) are resampled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    """2.5th and 97.5th percentiles, rounded."""
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


def bootstrap_ranking(frame: pd.DataFrame, case_of: pd.Series, k: int, n: int, seed: int) -> dict:
    """Intervals for PR-AUC and Recall@K (top k within each snapshot).

    Args:
        frame: rows with acct_id, snapshot, label, score.
        case_of: alert_id per mule acct_id.
    """
    # 1. Fixed parts: sorted negatives and, per positive row, whether it made its snapshot's top k
    ranked = frame.sort_values("score", ascending=False, kind="stable")
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


def bootstrap_detection(table: pd.DataFrame, n: int, seed: int) -> dict:
    """Intervals for the per-mule detection shares of detection_table, resampling cases."""
    groups = [g for _, g in table.groupby("alert_id")]
    rng = np.random.default_rng(seed)
    by_c, early = [], []
    for _ in range(n):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        by_c.append(sample.by_completion.mean())
        early.append(sample.early.mean())
    return {"detected_by_completion_ci95": _interval(np.array(by_c)), "detected_early_ci95": _interval(np.array(early))}
