"""Out-of-fold daily LightGBM scores of the train days, used as the payee's risk score in the transfer-time task.

A transfer on day d reads the payee's end-of-day d - 1 score. On validation and test days that is the score
of a model trained on the train split; on train days an in-sample score would carry the payee's own label,
so every train account-day is scored by a model that never saw the account's laundering case.

Two steps:
  folds (local, reads ground truth): case members get the fold of their case, other accounts a random fold.
    Writes <processed>/oof_folds_<dataset>.csv (acct_id, fold); no account belongs to two cases.
  train (Kaggle CPU, reads the export and the folds file): for fold k, train on the train rows of the other
    folds (every positive, 1% of negatives, as lightgbm_export) with early stopping on validation PR-AUC,
    then score every train row of fold k. Writes <out>/oof_train_fold<k>.parquet (acct_id, snapshot, score)
    and oof_fold<k>.json.
Run: python -m earlymule.models.oof_scores folds [--k 5]
     python -m earlymule.models.oof_scores train --data <export> --folds <csv> --fold K [--seed 0] --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.experiment_log import git_state, set_seed
from earlymule.models.lightgbm_export import KEYS, load_split, train_mask
from earlymule.models.tabular import fit, load_config


def assign_folds(accounts: pd.Index, case_of: pd.Series, k: int, seed: int = 0) -> pd.Series:
    """Fold per account: all members of a case share one fold; other accounts are spread at random.

    Args:
        accounts: every account id.
        case_of: case id per case member (indexed by acct_id).
    Returns:
        Series acct_id -> fold in 0..k-1.
    """
    rng = np.random.default_rng(seed)
    # 1. Cases shuffled, then dealt round-robin so folds hold about the same number of cases
    cases = rng.permutation(np.sort(case_of.unique()))
    case_fold = pd.Series(np.arange(len(cases)) % k, index=cases)
    # 2. Everyone else at random; case members overwrite with their case's fold
    fold = pd.Series(rng.integers(0, k, len(accounts)), index=accounts, name="fold")
    fold.loc[case_of.index] = case_fold.reindex(case_of.to_numpy()).to_numpy()
    return fold.rename_axis("acct_id")


def write_folds(k: int) -> Path:
    """Folds of the configured dataset from its ground truth; returns the CSV path."""
    paths = load_paths()
    gt = pd.read_parquet(paths.dataset_processed / "ground_truth.parquet")
    if gt.acct_id.duplicated().any():
        raise ValueError("an account belongs to two cases; folds by case would split it")
    accounts = pd.read_csv(paths.dataset_dir / "accounts.csv", usecols=["acct_id"]).acct_id
    folds = assign_folds(pd.Index(accounts), gt.set_index("acct_id").alert_id, k)
    out = paths.processed / f"oof_folds_{paths.dataset}.csv"
    folds.to_csv(out)
    print(f"{len(folds):,} accounts, {gt.alert_id.nunique()} cases in {k} folds -> {out}")
    return out


def train_fold(data: Path, folds_csv: Path, fold: int, seed: int, out: Path) -> dict:
    """Train on the other folds, score fold `fold`'s train rows, write scores and a run record."""
    start = time.time()
    cfg = load_config("daily.yaml") | {"seed": seed}
    set_seed(seed)
    fold_of = pd.read_csv(folds_csv).set_index("acct_id").fold

    # 1. Fold of every train row; the 1% negative sample is drawn exactly as lightgbm_export does
    rows = pd.read_parquet(data / "rows_train.parquet", columns=["acct_id", "label"])
    in_fold = (fold_of.reindex(rows.acct_id).to_numpy() == fold)
    sample = train_mask(rows.label, cfg["train_negative_rate"], np.random.default_rng(seed))
    deposit = pd.read_parquet(data / "nodes.parquet").set_index("acct_id").initial_deposit
    train = load_split(data, "train", deposit, sample & ~in_fold)
    val = load_split(data, "val", deposit)
    features = [c for c in train.columns if c not in (*KEYS, "label")]

    # 2. Fit, then score every train row of this fold (all of them, not only the sample)
    model = fit(train, val, features, cfg)
    held = load_split(data, "train", deposit, in_fold)
    scores = held[KEYS].assign(score=model.predict(held[features], raw_score=True))
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / f"oof_train_fold{fold}.parquet", index=False)

    # 3. Run record
    commit, dirty = git_state()
    meta = {"fold": fold, "seed": seed, "commit": commit, "dirty": dirty, "lightgbm_version": lgb.__version__,
            "train_rows": len(train), "scored_rows": len(scores), "features": len(features),
            "best_iteration": int(model.best_iteration_),
            "best_val_pr_auc": round(model.best_score_["valid_0"]["average_precision"], 4),
            "seconds": round(time.time() - start, 1)}
    (out / f"oof_fold{fold}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


def main() -> None:
    """Dispatch the folds and train steps."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--k", type=int, default=5)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--folds", required=True)
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.step == "folds":
        write_folds(args.k)
    else:
        train_fold(Path(args.data), Path(args.folds), args.fold, args.seed, Path(args.out))


if __name__ == "__main__":
    main()
