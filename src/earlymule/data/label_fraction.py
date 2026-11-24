"""Label scarcity (research plan section 19): keep the labels of a share of the training mules, chosen by case.

A bank learns of mules case by case, so whole laundering cases keep or lose their labels together. Cases are
shuffled once per seed and the first max(1, round(fraction x cases)) keep their labels, so the sets are nested
(1% inside 5% inside 10% ...). The other training mules are either relabelled negative (default: the bank does not
know them, so they sit among the negatives) or dropped from the loss. Validation and test labels are untouched.
Run: python -m earlymule.data.label_fraction --export <export folder> --out <csv>
Writes the case map acct_id, alert_id of the training mules, which the Kaggle runs read with --label-cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.config import load_paths

UNLABELED = ("negative", "drop")


def train_mule_cases(rows_train: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """acct_id, alert_id of every account with a positive train row; raise if one has no case."""
    mules = pd.DataFrame({"acct_id": np.unique(rows_train.acct_id[rows_train.label == 1])})
    cases = mules.merge(ground_truth[["acct_id", "alert_id"]], on="acct_id", how="left")
    if cases.alert_id.isna().any() or cases.acct_id.duplicated().any():
        raise ValueError("every training mule must belong to exactly one case")
    return cases.astype({"alert_id": int})


def kept_mules(cases: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    """acct_id of the training mules whose case keeps its label at this fraction and seed."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    order = np.random.default_rng(seed).permutation(np.sort(cases.alert_id.unique()))
    keep = order[:max(1, round(fraction * len(order)))]
    return cases.acct_id[cases.alert_id.isin(keep)].to_numpy()


def relabel(label: np.ndarray, acct_id: np.ndarray, kept: np.ndarray, unlabeled: str) -> np.ndarray:
    """Train labels with the positives of unkept mules set to 0 (negative) or NaN (drop: left out of the loss)."""
    if unlabeled not in UNLABELED:
        raise ValueError(f"unlabeled must be one of {UNLABELED}")
    out = label.astype("float64").copy()
    lost = (label == 1) & ~np.isin(acct_id, kept)
    out[lost] = 0.0 if unlabeled == "negative" else np.nan
    return out


def main() -> None:
    """Write the case map of the training mules of an export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, help="export folder with rows_train.parquet")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = pd.read_parquet(Path(args.export) / "rows_train.parquet", columns=["acct_id", "label"])
    # Ground truth of the export's own dataset (its manifest names it)
    dataset = json.loads((Path(args.export) / "manifest.json").read_text(encoding="utf-8"))["dataset"]
    gt = pd.read_parquet(load_paths().with_dataset(dataset).dataset_processed / "ground_truth.parquet",
                         columns=["acct_id", "alert_id"])
    cases = train_mule_cases(rows, gt)
    cases.to_csv(args.out, index=False)
    print(f"{len(cases)} training mules in {cases.alert_id.nunique()} cases -> {args.out}")


if __name__ == "__main__":
    main()
