"""Walking skeleton: AMLSim run -> weekly snapshots -> point-in-time features -> LightGBM -> PR-AUC, Recall@K.

Reads the configured dataset (configs/paths.yaml) and configs/lightgbm.yaml, scores the validation
and test snapshots, and appends one row to experiments/experiments.csv.
Run: python -m earlymule.models.tabular [--seed N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

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
from earlymule.evaluation.uncertainty import bootstrap_ranking
from earlymule.experiment_log import log_run, set_seed
from earlymule.features.point_in_time import load_transactions, pair_first_day, snapshot_features

META = ["acct_id", "snapshot", "label", "typology"]


def load_config(name: str) -> dict:
    """Read configs/<name>; a `base` key names another config whose values it overrides."""
    cfg = yaml.safe_load((repo_root() / "configs" / name).read_text(encoding="utf-8"))
    base = cfg.pop("base", None)
    return {**load_config(base), **cfg} if base else cfg


def load_inputs() -> dict:
    """Transfers, static attributes and mule ground truth of the configured dataset."""
    paths = load_paths()
    base = amlsim.read_base_date(paths.amlsim_conf)
    tx = load_transactions(paths.dataset_dir, base)
    accounts = pd.read_csv(paths.dataset_dir / "accounts.csv",
                           usecols=input_columns("accounts.csv")).set_index("acct_id")
    gt = build_ground_truth(amlsim.load_alert_accounts(paths.dataset_dir),
                            amlsim.load_alert_transactions(paths.dataset_dir, base),
                            amlsim.load_alert_members(paths.dataset_tmp_dir))
    mules = gt[gt.is_mule_candidate].set_index("acct_id")
    # Hard negatives: alert members without both illicit directions (sources, exits, one-sided hubs)
    others = pd.Index(gt.acct_id[~gt.is_mule_candidate].unique())
    return {"tx": tx, "accounts": accounts, "mules": mules, "other_members": others,
            "first_day": pair_first_day(tx)}


def build_split(inputs: dict, label_fn: Callable[[int], pd.Series], days, cfg, negative_rate=1.0,
                rng=None, keep: Callable[[pd.DataFrame], pd.Series] | None = None,
                keep_unlabeled: bool = False) -> pd.DataFrame:
    """Stack the snapshots of one split into rows (acct_id, snapshot, features, label, typology).

    Args:
        label_fn: day t -> label per account (NaN = no label, e.g. a mule after completion).
        negative_rate: share of negatives kept (train only); positives are always kept.
        keep: optional row filter on the feature table (e.g. regular accounts only).
        keep_unlabeled: keep NaN-label rows (scoring splits). A bank cannot tell a finished mule apart,
            so it must stay in the ranking and can take alert slots; metrics skip its label.
    """
    accounts, frames = inputs["accounts"], []
    for t in days:
        # 1. Labels at t, then negative subsampling
        label = label_fn(t)
        mask = pd.Series(True, index=label.index) if keep_unlabeled else label.notna()
        if negative_rate < 1.0:
            mask &= label.eq(1) | pd.Series(rng.random(len(label)) < negative_rate, index=label.index)
        ids = label.index[mask]
        # 2. Features from transactions up to t, plus the static input
        f = snapshot_features(inputs["tx"], ids, t, tuple(cfg["windows"]), inputs["first_day"])
        f["initial_deposit"] = accounts.initial_deposit.reindex(ids).astype("float32")
        f["label"] = label[ids].to_numpy()
        f["snapshot"] = t
        f["typology"] = inputs["mules"].typology.reindex(ids).where(f.label.eq(1)).to_numpy()
        frames.append(f[keep(f)] if keep else f)
    return pd.concat(frames).rename_axis("acct_id").reset_index()


def model_features(frame: pd.DataFrame, cfg: dict) -> list[str]:
    """Feature columns: everything except metadata and the configured exclusions."""
    return [c for c in frame.columns if c not in META and c not in cfg.get("exclude_features", [])]


def fit(train: pd.DataFrame, val: pd.DataFrame, features: list[str], cfg: dict) -> lgb.LGBMClassifier:
    """LightGBM with early stopping on validation PR-AUC (labelled validation rows only)."""
    model = lgb.LGBMClassifier(random_state=cfg["seed"], **cfg["lightgbm"])
    val = val[val.label.notna()]
    model.fit(train[features], train.label, eval_set=[(val[features], val.label)],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    return model


def seed_from_cli(cfg: dict) -> dict:
    """Override the config seed with --seed, for repeated runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    return {**cfg, "seed": parser.parse_args().seed}


def main() -> None:
    """Build the splits, train LightGBM, evaluate on validation and test, log the run."""
    start = time.time()
    cfg = seed_from_cli(load_config("lightgbm.yaml"))
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # 1. Inputs and weekly snapshots per split; train negatives are subsampled to keep memory small
    inputs = load_inputs()
    onset = inputs["mules"].onset_day.astype(float)
    accts = inputs["accounts"].index
    splits = {s.name: s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    days = {n: snapshot_days(s, cfg["horizon_days"], cfg["min_history_days"]) for n, s in splits.items()}
    data = {n: build_split(inputs, lambda t: snapshot_labels(onset, accts, t, cfg["horizon_days"]), days[n], cfg,
                           cfg["train_negative_rate"] if n == "train" else 1.0, rng, keep_unlabeled=n != "train")
            for n in splits}
    features = model_features(data["train"], cfg)

    # 2. Train, then evaluate on all accounts and on regular accounts (enough history at the snapshot day)
    model = fit(data["train"], data["val"], features, cfg)
    metrics = {"best_iteration": int(model.best_iteration_), "features": len(features)}
    for name in ["val", "test"]:
        # Log-odds, not probabilities: probabilities saturate at 1.0 and tie at the top of the ranking
        frame = data[name].assign(score=model.predict(data[name][features], raw_score=True))
        regular = frame[frame.n_tx_hist.ge(cfg["regular_min_tx"])]
        metrics[name] = {"all": evaluate(frame, cfg["recall_k"]), "regular": evaluate(regular, cfg["recall_k"])}
    # 3. Case-bootstrap intervals on test
    case_of = inputs["mules"].alert_id
    for pop, f in [("all", frame), ("regular", regular)]:
        metrics["test"][pop].update(bootstrap_ranking(f, case_of, cfg["recall_k"], cfg["bootstrap_samples"], cfg["seed"]))
    metrics["seconds"] = round(time.time() - start, 1)

    # 4. Log and report
    config = {**cfg, "splits": {n: [s.first_day, s.last_day] for n, s in splits.items()},
              "snapshots": {n: len(d) for n, d in days.items()}, "train_rows": len(data["train"]),
              "feature_names": features}
    log_run(cfg.get("run_name", "lightgbm_weekly"), config=config, seed=cfg["seed"], metrics=metrics)
    print(json.dumps(metrics, indent=1))
    gain = pd.Series(model.booster_.feature_importance("gain"), index=features)
    print("top features (gain share):\n" + (gain / gain.sum()).sort_values(ascending=False).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
