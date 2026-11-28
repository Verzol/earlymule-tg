"""Why the Circular 17 rule (code 3) catches few mules even by completion: which condition fails, and why.

For every test mule, on each day of its case (onset to completion), the rule's two conditions are checked on
all of the mule's transfers (what the rule sees) and on its illicit transfers only (the case money alone).
Comparing the two separates two causes: too few senders in the case, or forwarding diluted by the mule's
normal transfers. Evaluation-only: the illicit flag comes from the ground truth.
Input: transfers and ground truth of one AMLSim run; thresholds from configs/reference_rules.yaml.
Output: reports/figures/rule_diagnosis/<dataset>_mules.csv (one row per mule) and <dataset>_summary.csv.
Run: python -m earlymule.analysis.rule_diagnosis [--dataset <AMLSim run>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from earlymule.config import load_paths
from earlymule.data import amlsim
from earlymule.data.splits import chronological_splits
from earlymule.features.point_in_time import pair_first_day, snapshot_features
from earlymule.models.tabular import load_config, load_inputs

OUT = Path("reports/figures/rule_diagnosis")


def day_conditions(tx: pd.DataFrame, first_day: pd.Series, ids: pd.Index, t: int, w: int, rule: dict) -> pd.DataFrame:
    """Senders, forwarded share and both rule conditions of `ids` at the end of day t, from transfers `tx`."""
    f = snapshot_features(tx, ids, t, (w,), first_day)
    senders, share = f[f"peers_in_{w}d"], f[f"fwd_1d_{w}d"]
    return pd.DataFrame({"senders": senders, "share": share, "n_tx_hist": f.n_tx_hist,
                         "senders_ok": senders.ge(rule["min_senders"]),
                         "share_ok": share.ge(rule["min_forward_share"])}, index=ids)


def mule_days(inputs: dict, illicit: pd.DataFrame, days: range, w: int, rule: dict, min_tx: int) -> pd.DataFrame:
    """One row per (mule, case day) in `days`: rule conditions on all transfers and on illicit transfers only."""
    mules = inputs["mules"]
    first_all, first_ill = pair_first_day(inputs["tx"]), pair_first_day(illicit)
    frames = []
    for t in days:
        # 1. Mules whose case is running on day t
        ids = mules.index[mules.onset_day.le(t) & mules.completion_day.ge(t)]
        if len(ids) == 0:
            continue
        # 2. Same conditions, once on every transfer and once on the case money alone
        a = day_conditions(inputs["tx"], first_all, ids, t, w, rule)
        i = day_conditions(illicit, first_ill, ids, t, w, rule).drop(columns="n_tx_hist").add_prefix("ill_")
        frames.append(a.join(i).assign(snapshot=t))
    rows = pd.concat(frames).rename_axis("acct_id").reset_index()
    # 3. Only days on which the mule is in the scored (regular) population, as for the rule itself
    return rows[rows.n_tx_hist.ge(min_tx)]


def per_mule(rows: pd.DataFrame, mules: pd.DataFrame) -> pd.DataFrame:
    """Whether each condition holds on any scored case day, on all transfers and on illicit ones."""
    rows = rows.assign(both=rows.senders_ok & rows.share_ok, ill_both=rows.ill_senders_ok & rows.ill_share_ok)
    flags = ["senders_ok", "share_ok", "both", "ill_senders_ok", "ill_share_ok", "ill_both"]
    out = rows.groupby("acct_id")[flags].any()
    out["max_ill_senders"] = rows.groupby("acct_id").ill_senders.max()
    out["max_share"] = rows.groupby("acct_id").share.max()
    out["max_ill_share"] = rows.groupby("acct_id").ill_share.max()
    return out.join(mules[["typology", "onset_day", "completion_day"]])


def summary(m: pd.DataFrame) -> pd.DataFrame:
    """Share of mules meeting each condition, overall and per typology."""
    flags = ["senders_ok", "share_ok", "both", "ill_senders_ok", "ill_share_ok", "ill_both"]
    by = m.groupby("typology")[flags].mean()
    by.loc["all"] = m[flags].mean()
    by["mules"] = m.groupby("typology").size().reindex(by.index).fillna(len(m)).astype(int)
    by["median_max_ill_senders"] = m.groupby("typology").max_ill_senders.median().reindex(by.index)
    by.loc["all", "median_max_ill_senders"] = m.max_ill_senders.median()
    return by.round(3)


def main() -> None:
    """Rule conditions per test mule of one dataset, written as CSV tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="AMLSim run folder; default from configs/paths.yaml")
    args = parser.parse_args()
    cfg = load_config("reference_rules.yaml")
    rule, w = cfg["rule"], cfg["rule"]["window_days"]

    # 1. Inputs and the illicit transfers, sorted by day as snapshot_features expects
    paths = load_paths()
    paths = paths.with_dataset(args.dataset) if args.dataset else paths
    inputs = load_inputs(paths)
    base = amlsim.read_base_date(paths.amlsim_conf)
    illicit = amlsim.load_alert_transactions(paths.dataset_dir, base).sort_values("day", kind="stable")
    illicit = illicit[["orig_acct", "bene_acct", "base_amt", "day"]].reset_index(drop=True)

    # 2. Test period, mules whose case starts in it, case days only
    test = next(s for s in chronological_splits(cfg["total_days"], tuple(cfg["split_fractions"])) if s.name == "test")
    inputs["mules"] = inputs["mules"][inputs["mules"].onset_day.between(test.first_day, test.last_day)]
    rows = mule_days(inputs, illicit, range(test.first_day, test.last_day + 1), w, rule, cfg["regular_min_tx"])

    # 3. Tables
    m = per_mule(rows, inputs["mules"])
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / f"{paths.dataset}_mules.csv")
    s = summary(m)
    s.to_csv(OUT / f"{paths.dataset}_summary.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()
