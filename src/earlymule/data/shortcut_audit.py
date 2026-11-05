"""Screen simple account features for label shortcuts created by the simulator's member selection.

Each feature ranks accounts alone against the mule-candidate label (ROC-AUC 0.5 = no signal,
far from 0.5 in either direction = shortcut). Features cover the whole simulation, so this is a
screening of the dataset, not a point-in-time model; never reuse them as model inputs.
Run: python -m earlymule.data.shortcut_audit   (writes data/processed/shortcut_audit.csv, shortcut_members.csv)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from earlymule.config import load_paths
from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth

# Accounts with at least this many normal transfers form the "regular" population
REGULAR_MIN_NORMAL_TX = 20


def account_activity(accounts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Whole-period normal activity per account.

    Args:
        accounts: accounts.csv with acct_id, initial_deposit.
        tx: transactions.csv with orig_acct, bene_acct, base_amt, is_sar.
    Returns:
        DataFrame indexed by acct_id; counts are 0 for accounts without normal transfers.
    """
    normal = tx[~tx.is_sar.astype(str).str.lower().eq("true")]
    f = pd.DataFrame(index=pd.Index(accounts.acct_id, name="acct_id"))
    out, inn = normal.groupby("orig_acct"), normal.groupby("bene_acct")
    f["normal_out"] = out.size().reindex(f.index, fill_value=0)
    f["normal_in"] = inn.size().reindex(f.index, fill_value=0)
    f["normal_total"] = f.normal_out + f.normal_in
    f["n_out_peers"] = out.bene_acct.nunique().reindex(f.index, fill_value=0)
    f["n_in_peers"] = inn.orig_acct.nunique().reindex(f.index, fill_value=0)
    f["mean_out_amt"] = out.base_amt.mean().reindex(f.index)
    f["initial_deposit"] = accounts.set_index("acct_id").initial_deposit.reindex(f.index)
    f["acct_id"] = f.index
    return f


def single_feature_auc(features: pd.DataFrame, positive: pd.Series,
                       populations: dict[str, pd.Series]) -> pd.DataFrame:
    """ROC-AUC of each feature alone, within each population.

    Args:
        features: one row per account.
        positive: bool label aligned with features.
        populations: name -> bool mask aligned with features.
    Returns:
        DataFrame (feature x population); NaN where a population lacks one class.
    """
    table = {}
    for name, mask in populations.items():
        col = {}
        for feature in features.columns:
            # Missing values (e.g. mean amount without outflow) are left out, not imputed
            keep = mask & features[feature].notna()
            y = positive[keep]
            col[feature] = roc_auc_score(y, features.loc[keep, feature]) if 0 < y.sum() < len(y) else np.nan
        table[name] = col
    return pd.DataFrame(table)


def member_activity(f: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Normal activity of alert members by role and main flag, against non-member accounts.

    Shows whether the selection of alert members (not hubs in particular) drives the activity shortcut.
    Args:
        f: account_activity output.
        gt: ground truth, one row per (alert_id, acct_id).
    Returns:
        One row per group: accounts, median normal transfers, share regular, median percentile among active accounts.
    """
    # 1. One group per member row; an account in two alerts counts once per role it plays
    groups = gt.assign(group=np.where(gt.is_mule_candidate, "mule", "other_" + gt.role.astype(str))
                       + np.where(gt.is_main, " (main)", ""))[["acct_id", "group"]].drop_duplicates()
    non_member = f.index[~f.index.isin(gt.acct_id)]
    groups = pd.concat([groups, pd.DataFrame({"acct_id": non_member[f.loc[non_member, "normal_total"].gt(0)],
                                              "group": "non_member_active"}),
                        pd.DataFrame({"acct_id": non_member[f.loc[non_member, "normal_total"].ge(REGULAR_MIN_NORMAL_TX)],
                                      "group": "non_member_regular"})])
    # 2. Percentile of normal activity among every account with at least one normal transfer
    active = f.normal_total[f.normal_total.gt(0)]
    pct = active.rank(pct=True)
    g = groups.assign(n=groups.acct_id.map(f.normal_total), pct=groups.acct_id.map(pct)).groupby("group")
    return pd.DataFrame({"accounts": g.size(), "median_normal_tx": g.n.median(),
                         "share_regular": g.n.apply(lambda s: s.ge(REGULAR_MIN_NORMAL_TX).mean()),
                         "median_pct_among_active": g.pct.median()}).sort_values("median_normal_tx")


def main() -> None:
    """Screen the configured dataset and write the AUC table."""
    # 1. Labels and whole-period activity of the selected run
    paths = load_paths()
    base = amlsim.read_base_date(paths.amlsim_conf)
    gt = build_ground_truth(amlsim.load_alert_accounts(paths.dataset_dir),
                            amlsim.load_alert_transactions(paths.dataset_dir, base),
                            amlsim.load_alert_members(paths.dataset_tmp_dir))
    accounts = pd.read_csv(paths.dataset_dir / "accounts.csv", usecols=["acct_id", "initial_deposit"])
    tx = pd.read_csv(paths.dataset_dir / "transactions.csv",
                     usecols=["orig_acct", "bene_acct", "base_amt", "is_sar"])
    f = account_activity(accounts, tx)
    positive = pd.Series(f.index.isin(gt.acct_id[gt.is_mule_candidate]), index=f.index)

    # 2. Populations: other alert members are left out of the last one, as neither clean negatives nor mules
    other_member = pd.Series(f.index.isin(gt.acct_id), index=f.index) & ~positive
    populations = {
        "all": pd.Series(True, index=f.index),
        "active": f.normal_total.gt(0),
        "regular": f.normal_total.ge(REGULAR_MIN_NORMAL_TX),
        "regular_without_other_members": f.normal_total.ge(REGULAR_MIN_NORMAL_TX) & ~other_member,
    }
    table = single_feature_auc(f, positive, populations)

    # 3. Write and report
    paths.processed.mkdir(parents=True, exist_ok=True)
    out_file = paths.processed / "shortcut_audit.csv"
    table.rename_axis("feature").to_csv(out_file)
    sizes = {k: f"{int(m.sum())} ({int(positive[m].sum())} pos)" for k, m in populations.items()}
    print(f"{paths.dataset}: populations {sizes} -> {out_file}")
    print(table.round(3).to_string())
    members = member_activity(f, gt)
    members.to_csv(paths.processed / "shortcut_members.csv")
    print(members.round(3).to_string())


if __name__ == "__main__":
    main()
