"""Difficulty and leakage check: random ranking, unlearned rules and LightGBM on feature groups, same weekly split.

Each configuration is scored on three test populations (all accounts, regular accounts, regular accounts
without non-mule alert members) and logged as its own row (run_name difficulty_<name>) in
experiments/experiments.csv. Also writes data/processed/difficulty_table.csv.
Run: python -m earlymule.models.difficulty [--seed N]
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.data.splits import chronological_splits, snapshot_days, snapshot_labels
from earlymule.evaluation.metrics import evaluate
from earlymule.evaluation.uncertainty import bootstrap_ranking
from earlymule.experiment_log import log_run, set_seed
from earlymule.models.tabular import build_split, fit, load_config, load_inputs, model_features, seed_from_cli


def populations(frame: pd.DataFrame, other_members: pd.Index, regular_min_tx: int) -> dict[str, pd.Series]:
    """Row masks of the evaluation populations.

    Args:
        frame: snapshot rows with acct_id and n_tx_hist (transfers up to the snapshot day).
        other_members: alert members that are not mules (sources, exits, one-sided hubs).
    """
    regular = frame.n_tx_hist.ge(regular_min_tx)
    return {"all": pd.Series(True, index=frame.index), "regular": regular,
            "regular_no_other_members": regular & ~frame.acct_id.isin(other_members)}


def rule_score(frame: pd.DataFrame, feature: str) -> np.ndarray:
    """Unlearned score per row: a feature column, or min_amt_in_out_7d (money both received and sent)."""
    if feature == "min_amt_in_out_7d":
        return np.minimum(frame.amt_in_7d, frame.amt_out_7d).to_numpy()
    return frame[feature].to_numpy()


def score_populations(frame: pd.DataFrame, score: np.ndarray, masks: dict, cfg: dict,
                      case_of: pd.Series | None = None) -> dict:
    """Metrics per population; case-bootstrap intervals when case_of is given (test only)."""
    scored = frame.assign(score=score)
    out = {}
    for name, mask in masks.items():
        f = scored[mask.to_numpy()]
        out[name] = evaluate(f, cfg["recall_k"])
        if case_of is not None:
            out[name].update(bootstrap_ranking(f, case_of, cfg["recall_k"], cfg["bootstrap_samples"], cfg["seed"]))
    return out


def main() -> None:
    """Build the weekly split once, score every configuration, log one row each."""
    start = time.time()
    cfg = seed_from_cli(load_config("difficulty.yaml"))
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # 1. Same inputs, split and train subsampling as the weekly LightGBM run
    inputs = load_inputs()
    onset = inputs["mules"].onset_day.astype(float)
    accts = inputs["accounts"].index
    splits = {s.name: s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    days = {n: snapshot_days(s, cfg["horizon_days"], cfg["min_history_days"]) for n, s in splits.items()}
    data = {n: build_split(inputs, lambda t: snapshot_labels(onset, accts, t, cfg["horizon_days"]), days[n], cfg,
                           cfg["train_negative_rate"] if n == "train" else 1.0, rng) for n in splits}
    masks = {n: populations(data[n], inputs["other_members"], cfg["regular_min_tx"]) for n in ["val", "test"]}
    case_of = inputs["mules"].alert_id

    # 2. Scores per configuration: random, rules (no training), LightGBM per feature group
    configs: dict[str, tuple[dict, dict]] = {}
    configs["random"] = ({"kind": "random"}, {n: rng.random(len(data[n])) for n in ["val", "test"]})
    for name, feature in cfg["rules"].items():
        configs[f"rule_{name}"] = ({"kind": "rule", "feature": feature},
                                   {n: rule_score(data[n], feature) for n in ["val", "test"]})
    for name, features in cfg["models"].items():
        features = features or model_features(data["train"], cfg)
        model = fit(data["train"], data["val"], features, cfg)
        # Log-odds rank without the ties that saturated probabilities create
        scores = {n: model.predict(data[n][features], raw_score=True) for n in ["val", "test"]}
        configs[f"lgbm_{name}"] = ({"kind": "lightgbm", "features": features,
                                    "best_iteration": int(model.best_iteration_)}, scores)

    # 3. Evaluate, log each configuration, collect the comparison table
    rows = []
    shared = {k: cfg[k] for k in ["seed", "total_days", "split_fractions", "horizon_days", "min_history_days",
                                  "windows", "train_negative_rate", "regular_min_tx", "recall_k", "bootstrap_samples"]}
    for name, (detail, scores) in configs.items():
        metrics = {"val": score_populations(data["val"], scores["val"], masks["val"], cfg),
                   "test": score_populations(data["test"], scores["test"], masks["test"], cfg, case_of)}
        config = {**shared, **detail, "lightgbm": cfg["lightgbm"] if detail["kind"] == "lightgbm" else None}
        log_run(f"{cfg['run_name']}_{name}", config=config, seed=cfg["seed"], metrics=metrics)
        for pop, m in metrics["test"].items():
            rows.append({"config": name, "population": pop, "rows": m["rows"], "positives": m["positives"],
                         "mule_rate": m["mule_rate"], "pr_auc": m["pr_auc"], "pr_auc_ci95": m["pr_auc_ci95"],
                         "roc_auc": m["roc_auc"], f"recall_at_{cfg['recall_k']}": m[f"recall_at_{cfg['recall_k']}"],
                         "by_typology": json.dumps(m[f"recall_at_{cfg['recall_k']}_by_typology"], sort_keys=True)})

    # 4. Write and report
    table = pd.DataFrame(rows)
    out_file = load_paths().processed / "difficulty_table.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_file, index=False)
    print(table.drop(columns="by_typology").round(4).to_string(index=False))
    print(f"-> {out_file} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
