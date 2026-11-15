"""Post-run EDA of the 80K dataset: mule timelines, features at the decision point, trajectories, model misses.

Used by notebooks/02_post_run_eda.ipynb (EMTG-69). Day indices follow load_transactions (end-of-day cut-off);
seq is the execution order within a day (data.intraday_order). Labels and alert ids are read here for
analysis only; no function feeds them into features.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from earlymule.data import amlsim
from earlymule.data.export_events import regular_from_day
from earlymule.evaluation.early_detection import daily_alerts
from earlymule.features.point_in_time import snapshot_features

# Feature families of configs/lightgbm_ablation.yaml, by column-name pattern (first match wins here)
FAMILY_PATTERNS = [("cross_window", "_7d_over_30d$"), ("activity", r"^(n|amt)_(in|out)_\d+d$"),
                   ("fan", r"^peers_"), ("pass_through", "^out_in_ratio_"), ("imbalance", "^imbalance_"),
                   ("churn", "^new_(peers|share)_"), ("burstiness", "^burst_"), ("rapid_forward", "^fwd_"),
                   ("dwell", "^dwell_"), ("two_hop", "^(up_amt_in|down_amt_out)_"), ("entropy", "^ent_"),
                   ("recency", "^days_since_"), ("static", "^initial_deposit$")]


def family_of(col: str) -> str:
    """Feature family of one column name."""
    return next(f for f, p in FAMILY_PATTERNS if re.search(p, col))


def labelled_transfers(run_dir: Path, base_date: pd.Timestamp) -> pd.DataFrame:
    """Every transfer with day, seq and its illicit flag and alert id (evaluation columns, analysis only)."""
    tx = pd.read_csv(run_dir / "transactions.csv",
                     usecols=["tran_id", "orig_acct", "bene_acct", "base_amt", "tran_timestamp", "is_sar", "alert_id"])
    tx["day"] = amlsim.to_day(tx.pop("tran_timestamp"), base_date)
    return tx.rename(columns={"tran_id": "seq"}).sort_values(["day", "seq"]).reset_index(drop=True)


def mule_timeline(mules: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Per mule: milestones, gaps in days, within-day order of first inflow and outflow, share forwarded.

    Args:
        mules: load_inputs()["mules"] (index acct_id; onset_day, first_illicit_*_day, completion_day, typology).
        tx: labelled_transfers output.
    """
    # 1. First illicit inflow and outflow of each mule in execution order, and illicit totals per direction
    ill = tx[tx.is_sar]
    first_in = ill.groupby("bene_acct").seq.min().rename("first_in_seq")
    first_out = ill.groupby("orig_acct").seq.min().rename("first_out_seq")
    amt_in = ill.groupby("bene_acct").base_amt.sum().rename("illicit_in")
    amt_out = ill.groupby("orig_acct").base_amt.sum().rename("illicit_out")
    t = mules[["typology", "onset_day", "first_illicit_inflow_day", "first_illicit_outflow_day", "completion_day"]]
    t = t.join([first_in, first_out, amt_in, amt_out]).astype({"onset_day": float})
    # 2. Gaps; same-day mules split by which comes first in execution order
    t["days_to_first_out"] = t.first_illicit_outflow_day - t.onset_day
    t["days_first_out_to_completion"] = t.completion_day - t.first_illicit_outflow_day
    t["case_days"] = t.completion_day - t.onset_day
    same = t.days_to_first_out.eq(0)
    t["same_day_order"] = np.where(~same, "different days", np.where(t.first_in_seq < t.first_out_seq,
                                                                     "receives first", "forwards first"))
    t["forwarded_share"] = t.illicit_out / t.illicit_in
    return t


def regular_accounts(tx: pd.DataFrame, min_tx: int) -> pd.Series:
    """First day each account has at least min_tx transfers (the scored population from that day on)."""
    return regular_from_day(tx, min_tx)


def sample_controls(days: np.ndarray, regular_from: pd.Series, exclude: pd.Index, per_day: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Random regular accounts outside every case on each given day: rows (acct_id, day)."""
    pool = regular_from.drop(exclude, errors="ignore")
    rows = []
    for d in np.unique(days):
        cand = pool.index[pool.to_numpy() <= d]
        rows.append(pd.DataFrame({"acct_id": rng.choice(cand, min(per_day, len(cand)), replace=False), "day": d}))
    return pd.concat(rows, ignore_index=True)


def matched_controls(timeline: pd.DataFrame, regular_from: pd.Series, n_tx_at, exclude: pd.Index,
                     per_mule: int, rng: np.random.Generator) -> pd.DataFrame:
    """For each mule, per_mule regular non-member accounts of the same activity decile on its onset day.

    Args:
        n_tx_at: n_tx_up_to(tx), transfers per account up to a day; called once per onset day.
    Returns:
        DataFrame acct_id, mule (the mule it is matched to).
    """
    out = []
    pool = regular_from.drop(exclude, errors="ignore")
    for d, group in timeline.groupby("onset_day"):
        # 1. Deciles of activity among regular non-members of that day, and each mule's decile
        cand = pool.index[pool.to_numpy() <= d]
        if len(cand) < 10 * per_mule:
            continue
        n = n_tx_at(int(d), cand.union(group.index))
        edges = np.quantile(n[cand], np.linspace(0, 1, 11))
        cand_bin = np.clip(np.searchsorted(edges, n[cand], side="right") - 1, 0, 9)
        # 2. Controls drawn from the mule's decile
        for mule in group.index:
            b = np.clip(np.searchsorted(edges, n[mule], side="right") - 1, 0, 9)
            same = cand[cand_bin == b]
            same = same if len(same) >= per_mule else cand
            out.append(pd.DataFrame({"acct_id": rng.choice(same, per_mule, replace=False), "mule": mule}))
    return pd.concat(out, ignore_index=True)


def n_tx_up_to(tx: pd.DataFrame):
    """Return f(day, accounts) -> number of transfers (either side) up to and including that day."""
    days = tx.day.to_numpy()

    def f(d: int, accts: pd.Index) -> pd.Series:
        past = tx.iloc[: np.searchsorted(days, d, side="right")]
        n = past.orig_acct.value_counts().add(past.bene_acct.value_counts(), fill_value=0)
        return n.reindex(accts, fill_value=0)
    return f


def features_for_rows(inputs: dict, rows: pd.DataFrame, windows: tuple[int, ...], exclude: list[str]) -> pd.DataFrame:
    """Point-in-time features (end of each row's day) for rows (acct_id, day, ...); extra columns kept.

    Rows sharing a day are computed in one snapshot call; duplicates are computed once.
    """
    frames = []
    for d, ids in rows.drop_duplicates(["acct_id", "day"]).groupby("day").acct_id:
        f = snapshot_features(inputs["tx"], pd.Index(ids.unique(), name="acct_id"), int(d), windows,
                              inputs["first_day"])
        f["initial_deposit"] = inputs["accounts"].initial_deposit.reindex(f.index).astype("float32")
        frames.append(f.drop(columns=[c for c in exclude if c in f]).reset_index().assign(day=d))
    feats = pd.concat(frames, ignore_index=True)
    return rows.merge(feats, on=["acct_id", "day"], how="left")


def univariate_table(frame: pd.DataFrame, features: list[str], label: str = "is_mule") -> pd.DataFrame:
    """Per feature: medians by class, standardised mean difference, ROC-AUC, KS and PR-AUC on one feature.

    Missing values are filled with the column median for the AUCs; ROC-AUC below 0.5 means lower in mules,
    so `separation` = |ROC-AUC - 0.5| ranks features regardless of direction.
    """
    y = frame[label].to_numpy().astype(bool)
    rows = []
    for c in features:
        x = frame[c].astype(float)
        a, b = x[y].dropna(), x[~y].dropna()
        filled = x.fillna(x.median())
        auc = roc_auc_score(y, filled)
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"feature": c, "family": family_of(c), "median_mule": a.median(), "median_other": b.median(),
                     "missing_mule": 1 - len(a) / y.sum(), "missing_other": 1 - len(b) / (~y).sum(),
                     "smd": (a.mean() - b.mean()) / sd if sd > 0 else 0.0, "roc_auc": auc,
                     "ks": ks_2samp(a, b).statistic if len(a) and len(b) else np.nan,
                     "pr_auc": average_precision_score(y, filled if auc >= 0.5 else -filled)})
    out = pd.DataFrame(rows)
    out["separation"] = (out.roc_auc - 0.5).abs()
    return out.sort_values("separation", ascending=False).reset_index(drop=True)


def correlation_clusters(frame: pd.DataFrame, features: list[str], threshold: float = 0.9) -> pd.Series:
    """Cluster id per feature: features whose |Spearman rho| exceeds threshold end up together (average link)."""
    rho = frame[features].rank().corr().abs().fillna(0).to_numpy(copy=True)  # writable under copy-on-write
    np.fill_diagonal(rho, 1.0)
    dist = squareform(1 - rho, checks=False)
    ids = fcluster(linkage(dist, "average"), t=1 - threshold, criterion="distance")
    return pd.Series(ids, index=features, name="cluster")


def alert_outcomes(scores: pd.DataFrame, mules: pd.DataFrame, others: pd.Index, k: int, cooldown: int) -> pd.DataFrame:
    """Daily top-k alerts with cooldown, each tagged: active mule, mule outside its case days, case member, other.

    Args:
        scores: scores_test.parquet of a run (acct_id, snapshot, score, label).
    """
    alerts = daily_alerts(scores[["acct_id", "snapshot", "score"]], k, cooldown)
    m = mules.reindex(alerts.acct_id)
    active = (alerts.snapshot.to_numpy() >= m.onset_day.to_numpy()) & (alerts.snapshot.to_numpy() <= m.completion_day.to_numpy())
    kind = np.select([active, m.onset_day.notna().to_numpy(), alerts.acct_id.isin(others).to_numpy()],
                     ["active mule", "mule outside case days", "case member, not mule"], "other account")
    return alerts.assign(kind=kind)
