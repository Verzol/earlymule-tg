"""Walking skeleton: AMLSim run -> weekly snapshots -> point-in-time features -> LightGBM -> PR-AUC, Recall@K.

Reads the configured dataset (configs/paths.yaml) and configs/lightgbm.yaml, scores the validation
and test snapshots, and appends one row to experiments/experiments.csv.
Run: python -m earlymule.models.tabular
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from earlymule.config import load_paths, repo_root
from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth
from earlymule.data.leakage import input_columns
from earlymule.data.splits import chronological_splits, snapshot_days, snapshot_labels
from earlymule.evaluation.metrics import evaluate
from earlymule.experiment_log import log_run, set_seed
from earlymule.features.point_in_time import load_transactions, snapshot_features


def build_split(tx, accounts, onset, typology, days, cfg, negative_rate=1.0, rng=None) -> pd.DataFrame:
    """Stack the snapshots of one split into rows (acct_id, snapshot, features, label, typology).

    Args:
        accounts: accounts.csv input columns, indexed by acct_id.
        onset, typology: per mule candidate, indexed by acct_id.
        negative_rate: share of negatives kept (train only); positives are always kept.
    """
    frames = []
    for t in days:
        # 1. Label at t; mules that started before the window are left out
        label = snapshot_labels(onset, accounts.index, t, cfg["horizon_days"])
        keep = label.notna()
        if negative_rate < 1.0:
            keep &= label.eq(1) | pd.Series(rng.random(len(label)) < negative_rate, index=label.index)
        ids = label.index[keep]
        # 2. Features from transactions up to t, plus the static input
        f = snapshot_features(tx, ids, t, tuple(cfg["windows"]))
        f["initial_deposit"] = accounts.initial_deposit.reindex(ids).astype("float32")
        f["label"] = label[ids].to_numpy()
        f["snapshot"] = t
        f["typology"] = typology.reindex(ids).where(f.label.eq(1)).to_numpy()
        frames.append(f)
    return pd.concat(frames).rename_axis("acct_id").reset_index()


def main() -> None:
    """Build the splits, train LightGBM, evaluate on validation and test, log the run."""
    start = time.time()
    cfg = yaml.safe_load((repo_root() / "configs" / "lightgbm.yaml").read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # 1. Inputs: transfers and static attributes; labels from the ground truth
    paths = load_paths()
    base = amlsim.read_base_date(paths.amlsim_conf)
    tx = load_transactions(paths.dataset_dir, base)
    accounts = pd.read_csv(paths.dataset_dir / "accounts.csv",
                           usecols=input_columns("accounts.csv")).set_index("acct_id")
    gt = build_ground_truth(amlsim.load_alert_accounts(paths.dataset_dir),
                            amlsim.load_alert_transactions(paths.dataset_dir, base),
                            amlsim.load_alert_members(paths.dataset_tmp_dir))
    mules = gt[gt.is_mule_candidate].set_index("acct_id")
    onset, typology = mules.onset_day.astype(float), mules.typology

    # 2. Weekly snapshots per split; train negatives are subsampled to keep memory small
    splits = {s.name: s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    days = {n: snapshot_days(s, cfg["horizon_days"], cfg["min_history_days"]) for n, s in splits.items()}
    data = {n: build_split(tx, accounts, onset, typology, days[n], cfg,
                           cfg["train_negative_rate"] if n == "train" else 1.0, rng) for n in splits}
    meta = ["acct_id", "snapshot", "label", "typology"]
    features = [c for c in data["train"].columns if c not in meta]

    # 3. Train with early stopping on validation PR-AUC
    model = lgb.LGBMClassifier(random_state=cfg["seed"], **cfg["lightgbm"])
    model.fit(data["train"][features], data["train"].label, eval_set=[(data["val"][features], data["val"].label)],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])

    # 4. Evaluate on all accounts and on regular accounts (enough history at the snapshot day)
    metrics = {"best_iteration": int(model.best_iteration_), "features": len(features)}
    for name in ["val", "test"]:
        frame = data[name].assign(score=model.predict_proba(data[name][features])[:, 1])
        regular = frame[frame.n_tx_hist.ge(cfg["regular_min_tx"])]
        metrics[name] = {"all": evaluate(frame, cfg["recall_k"]), "regular": evaluate(regular, cfg["recall_k"])}
    metrics["seconds"] = round(time.time() - start, 1)

    # 5. Log and report
    config = {**cfg, "splits": {n: [s.first_day, s.last_day] for n, s in splits.items()},
              "snapshots": {n: len(d) for n, d in days.items()}, "train_rows": len(data["train"])}
    log_run("lightgbm_walking_skeleton", config=config, seed=cfg["seed"], metrics=metrics)
    print(json.dumps(metrics, indent=1))
    top = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
    print("top features (split count):\n" + top.to_string())


if __name__ == "__main__":
    main()
