"""Ranking metrics for snapshot scoring: PR-AUC, Recall@K per snapshot, mule rate as the random baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k(frame: pd.DataFrame, k: int) -> float:
    """Share of positives ranked in the top k of their own snapshot.

    Args:
        frame: rows with snapshot, label (0/1, NaN = unlabelled but still ranked) and score.
        k: accounts an analyst reviews per snapshot.
    """
    top = frame.sort_values("score", ascending=False, kind="stable").groupby("snapshot").head(k)
    total = frame.label.sum()
    return float(top.label.sum() / total) if total else np.nan


def evaluate(frame: pd.DataFrame, k: int) -> dict:
    """Metrics over all snapshots of one split and population.

    Args:
        frame: rows with snapshot, label, score and typology (NaN for negatives). NaN-label rows count
            for the top-k ranking only.
    Returns:
        counts, mule rate, PR-AUC, ROC-AUC, Recall@K and Recall@K per typology.
    """
    # 1. Pooled ranking quality over labelled rows; the mule rate is the PR-AUC of a random ranking
    lab = frame[frame.label.notna()]
    n_pos = int(lab.label.sum())
    out = {"rows": len(lab), "unlabelled_rows": len(frame) - len(lab), "positives": n_pos,
           "mule_rate": n_pos / len(lab),
           "pr_auc": float(average_precision_score(lab.label, lab.score)) if n_pos else np.nan,
           "roc_auc": float(roc_auc_score(lab.label, lab.score)) if 0 < n_pos < len(lab) else np.nan,
           f"recall_at_{k}": recall_at_k(frame, k)}

    # 2. Recall@K by typology, to see whether only some patterns are found
    top = frame.sort_values("score", ascending=False, kind="stable").groupby("snapshot").head(k)
    pos = frame[frame.label.eq(1)]
    found = top[top.label.eq(1)].typology.value_counts()
    out[f"recall_at_{k}_by_typology"] = {
        t: round(float(found.get(t, 0) / n), 4) for t, n in pos.typology.value_counts().items()}
    return out
