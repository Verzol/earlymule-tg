"""Ranking metrics for snapshot scoring: PR-AUC, Recall@K per snapshot, mule rate as the random baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def rank_by_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in ranking order: score descending, ties broken by acct_id ascending (same rule for every model)."""
    if "acct_id" not in frame:
        return frame.sort_values("score", ascending=False, kind="stable")
    return frame.sort_values(["score", "acct_id"], ascending=[False, True], kind="stable")


def recall_at_k(frame: pd.DataFrame, k: int) -> float:
    """Share of positives ranked in the top k of their own snapshot.

    Args:
        frame: rows with snapshot, label (0/1, NaN = unlabelled but still ranked) and score.
        k: accounts an analyst reviews per snapshot.
    """
    top = rank_by_score(frame).groupby("snapshot").head(k)
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
    top = rank_by_score(frame).groupby("snapshot").head(k)
    pos = frame[frame.label.eq(1)]
    found = top[top.label.eq(1)].typology.value_counts()
    out[f"recall_at_{k}_by_typology"] = {
        t: round(float(found.get(t, 0) / n), 4) for t, n in pos.typology.value_counts().items()}
    return out


def threshold_classification(frame: pd.DataFrame, theta: float) -> dict:
    """Secondary metrics of the research plan (23.2) for the rule score >= theta, over labelled rows.

    Each labelled account-day is one prediction, so these describe daily classification, not detection timing.
    """
    lab = frame[frame.label.notna()]
    y, pred = lab.label.astype(int), lab.score.ge(theta).astype(int)
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4)}
