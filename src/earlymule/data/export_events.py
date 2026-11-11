"""Export the transfer stream and the rows to score as a Kaggle Dataset folder for the temporal graph models.

Kaggle only trains and scores; evaluation runs locally on the returned scores, so test labels are not exported.
Rows follow the daily scoring run (configs/daily.yaml): regular accounts, train from day min_history_days,
validation and test every day with finished mules kept unlabelled. features_<split> holds the LightGBM
point-in-time features of each row, in the same row order.
Run: python -m earlymule.data.export_events [--dataset 80K-earlymule-mule]
Writes data/processed/kaggle/<dataset>/: events, nodes, rows_{train,val,test}.parquet,
features_{train,val,test}.parquet, manifest.json, dataset-metadata.json. Upload: kaggle datasets create -p <folder>; later: kaggle datasets version -p <folder> -m "..."
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from earlymule.config import load_paths, repo_root
from earlymule.data import amlsim
from earlymule.data.leakage import check_feature_columns
from earlymule.data.splits import active_labels, chronological_splits
from earlymule.experiment_log import git_state
from earlymule.features.point_in_time import snapshot_features
from earlymule.models.tabular import load_config, load_inputs


def regular_from_day(tx: pd.DataFrame, min_tx: int) -> pd.Series:
    """First day each account has at least min_tx transfers (sent or received) up to that day.

    Same count as the n_tx_hist feature, so an account is regular at day t exactly when this day is <= t.
    Accounts that never reach min_tx are absent.
    """
    # 1. One row per account side of a transfer; a self-transfer counts twice, as in n_tx_hist
    sides = pd.DataFrame({"acct": np.r_[tx.orig_acct.to_numpy(), tx.bene_acct.to_numpy()],
                          "day": np.r_[tx.day.to_numpy(), tx.day.to_numpy()]})
    # 2. Day of the min_tx-th side per account
    sides = sides.sort_values(["acct", "day"], kind="stable")
    nth = sides.groupby("acct").cumcount().add(1).eq(min_tx)
    return sides[nth.to_numpy()].set_index("acct").day.rename("regular_from_day")


def split_rows(label_fn, regular_from: pd.Series, days, keep_unlabeled: bool) -> pd.DataFrame:
    """(acct_id, snapshot, label) of the regular accounts on each day; NaN labels dropped unless keep_unlabeled."""
    frames = []
    for t in days:
        ids = regular_from.index[regular_from.le(t).to_numpy()]
        label = label_fn(t).reindex(ids)
        if not keep_unlabeled:
            label = label.dropna()
        frames.append(pd.DataFrame({"acct_id": label.index, "snapshot": t, "label": label.to_numpy()}))
    return pd.concat(frames, ignore_index=True).astype({"acct_id": "int64", "snapshot": "int16", "label": "float32"})


def row_features(inputs: dict, rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """LightGBM features of each (acct_id, snapshot) row at the end of its day, in the order of rows.

    Same columns as the daily LightGBM run minus initial_deposit, which the graph models read from nodes.
    """
    # 1. One snapshot per day, on that day's accounts only
    frames = []
    for t, ids in rows.groupby("snapshot", sort=False).acct_id:
        f = snapshot_features(inputs["tx"], pd.Index(ids), int(t), tuple(cfg["windows"]), inputs.get("first_day"))
        frames.append(f.drop(columns=cfg.get("exclude_features", [])).rename_axis("acct_id")
                      .reset_index().assign(snapshot=t))
    out = pd.concat(frames, ignore_index=True)
    # 2. Row order must match rows exactly: the models pair them by position
    if not (out.acct_id.to_numpy() == rows.acct_id.to_numpy()).all() or             not (out.snapshot.to_numpy() == rows.snapshot.to_numpy()).all():
        raise ValueError("feature rows are not in the order of the rows to score")
    cols = [c for c in out.columns if c not in ("acct_id", "snapshot")]
    return out[["acct_id", "snapshot", *cols]].astype({"snapshot": "int16"})


def build_export(inputs: dict, base_date: pd.Timestamp, cfg: dict) -> dict[str, pd.DataFrame]:
    """Tables of the export: events, nodes and the rows of each split (test without labels).

    Args:
        inputs: load_inputs output.
        base_date: day 0 of the simulation, for the day of the week.
        cfg: daily scoring config (split fractions, regular_min_tx, min_history_days, windows, exclude_features).
    """
    # 1. Nodes: every account, with a dense index the graph models use
    accounts = inputs["accounts"].sort_index()
    node_of = pd.Series(np.arange(len(accounts), dtype="int32"), index=accounts.index)
    nodes = pd.DataFrame({"node": node_of.to_numpy(), "acct_id": accounts.index,
                          "initial_deposit": accounts.initial_deposit.astype("float32").to_numpy()})

    # 2. Events in day order; only input columns (Data Spec section 6), the amount on a log scale
    tx = inputs["tx"]
    events = pd.DataFrame({"src": node_of.reindex(tx.orig_acct).to_numpy(),
                           "dst": node_of.reindex(tx.bene_acct).to_numpy(),
                           "day": tx.day.astype("int16").to_numpy(),
                           "log_amount": np.log1p(tx.base_amt.to_numpy()).astype("float32"),
                           "dow": ((base_date.day_of_week + tx.day) % 7).astype("int8").to_numpy()})
    if events[["src", "dst"]].isna().any().any():
        raise ValueError("a transfer names an account missing from accounts.csv")
    events = events.astype({"src": "int32", "dst": "int32"})

    # 3. Rows per split, same population and labels as the daily scoring run
    mules = inputs["mules"]
    onset, completion = mules.onset_day.astype(float), mules.completion_day.astype(float)
    label_fn = lambda t: active_labels(onset, completion, accounts.index, t)  # noqa: E731
    regular = regular_from_day(tx, cfg["regular_min_tx"])
    s = {x.name: x for x in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    rows = {"train": split_rows(label_fn, regular, range(cfg["min_history_days"], s["train"].last_day + 1), False),
            "val": split_rows(label_fn, regular, range(s["val"].first_day, s["val"].last_day + 1), True),
            # Test labels stay local: Kaggle cannot use them even by mistake
            "test": split_rows(label_fn, regular, range(s["test"].first_day, s["test"].last_day + 1),
                               True).drop(columns="label")}

    # 4. Node index on every row, and the row features; model inputs must not carry evaluation-only columns
    tables = {"events": events, "nodes": nodes}
    for name, r in rows.items():
        tables[f"rows_{name}"] = r.assign(node=node_of.reindex(r.acct_id).to_numpy())
        tables[f"features_{name}"] = row_features(inputs, r, cfg)
    for t in tables.values():
        check_feature_columns([c for c in t.columns if c != "acct_id"])
    return tables


def main() -> None:
    """Export one dataset to data/processed/kaggle/<dataset>/ with its manifest and Kaggle metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="AMLSim run folder; default from configs/paths.yaml")
    args = parser.parse_args()
    kaggle = yaml.safe_load((repo_root() / "configs" / "kaggle.yaml").read_text(encoding="utf-8"))
    cfg = load_config(kaggle["scoring_config"])

    # 1. Inputs of the chosen dataset
    paths = load_paths()
    if args.dataset:
        paths = paths.with_dataset(args.dataset)
    base_date = amlsim.read_base_date(paths.amlsim_conf)
    tables = build_export(load_inputs(paths), base_date, cfg)

    # 2. Parquet files
    out = paths.processed / "kaggle" / paths.dataset
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_parquet(out / f"{name}.parquet", index=False)

    # 3. Manifest: the notebook checks the commit and split days against it
    commit, dirty = git_state()
    splits = {x.name: [x.first_day, x.last_day]
              for x in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"]))}
    manifest = {"dataset": paths.dataset, "commit": commit, "dirty": dirty,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_date": str(base_date.date()), "splits": splits,
                "train_first_day": cfg["min_history_days"], "regular_min_tx": cfg["regular_min_tx"],
                "rows": {n: len(t) for n, t in tables.items()},
                "columns": {n: list(t.columns) for n, t in tables.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # 4. Kaggle metadata; private unless made public on the website
    slug = (kaggle["slug_prefix"] + paths.dataset).lower()
    meta = {"title": f"EarlyMule {paths.dataset} events", "id": f"{kaggle['owner']}/{slug}",
            "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"folder": str(out), **{k: manifest[k] for k in ["commit", "dirty", "rows"]}}, indent=1))


if __name__ == "__main__":
    main()
