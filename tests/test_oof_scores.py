"""Out-of-fold scores for the payee risk score: folds keep cases together, a fold's model never sees that fold."""

import json

import numpy as np
import pandas as pd

from earlymule.models import oof_scores
from earlymule.models.oof_scores import assign_folds, train_fold


def test_case_members_share_a_fold_and_every_account_has_one():
    """Members of one case get the same fold; cases spread evenly; non-members get a fold too."""
    accounts = pd.Index(range(1000))
    case_of = pd.Series(np.repeat(np.arange(50), 4), index=range(0, 200))
    fold = assign_folds(accounts, case_of, k=5)
    assert fold.index.equals(accounts) and set(fold.unique()) == set(range(5))
    assert fold.loc[case_of.index].groupby(case_of.to_numpy()).nunique().eq(1).all()
    assert fold.loc[case_of.index].groupby(case_of.to_numpy()).first().value_counts().eq(10).all()
    assert (fold == assign_folds(accounts, case_of, k=5)).all()


def test_fold_model_trains_without_its_fold_and_scores_only_it(tmp_path, monkeypatch):
    """Train rows of fold k never reach the model; every train row of fold k, sampled or not, gets a score."""
    rng = np.random.default_rng(0)
    accts = np.arange(40)
    for split, days in [("train", range(5)), ("val", range(5, 7))]:
        rows = pd.DataFrame([(a, d) for d in days for a in accts], columns=["acct_id", "snapshot"])
        rows["label"] = (rows.acct_id < 4).astype(float)
        rows.to_parquet(tmp_path / f"rows_{split}.parquet")
        rows[["acct_id", "snapshot"]].assign(n_out_1d=rng.random(len(rows))).to_parquet(
            tmp_path / f"features_{split}.parquet")
    pd.DataFrame({"acct_id": accts, "initial_deposit": 1.0}).to_parquet(tmp_path / "nodes.parquet")
    folds = tmp_path / "folds.csv"
    pd.DataFrame({"acct_id": accts, "fold": accts % 5}).to_csv(folds, index=False)

    seen = {}

    def fake_fit(train, val, features, cfg):
        """Record the train rows and return a stand-in model."""
        seen["accts"] = set(train.acct_id)

        class Model:
            best_iteration_ = 1
            best_score_ = {"valid_0": {"average_precision": 0.5}}

            def predict(self, x, raw_score=True):
                return np.zeros(len(x))
        return Model()

    monkeypatch.setattr(oof_scores, "fit", fake_fit)
    train_fold(tmp_path, folds, fold=2, seed=0, out=tmp_path / "out")
    scores = pd.read_parquet(tmp_path / "out" / "oof_train_fold2.parquet")
    assert not seen["accts"] & set(accts[accts % 5 == 2])
    assert set(scores.acct_id) == set(accts[accts % 5 == 2]) and len(scores) == 8 * 5
    assert json.loads((tmp_path / "out" / "oof_fold2.json").read_text())["scored_rows"] == 40
