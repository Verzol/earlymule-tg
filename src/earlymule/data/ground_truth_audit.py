"""Audit labels against both AMLSim transaction exports and write per-member evidence.

Run: python -m earlymule.data.ground_truth_audit
Inputs: configured AMLSim outputs and generator members. Outputs: ground_truth_label_audit.csv
and ground_truth_label_audit.json under data/processed; no feature or training input is produced.
"""

from __future__ import annotations

import json

import pandas as pd

from earlymule.config import load_paths
from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth


def audit_labels() -> tuple[pd.DataFrame, dict]:
    """Cross-check illicit edges; return same-alert evidence and counts at day resolution.

    Normal transfers inside each alert's observed first-to-last illicit day are counted
    only to show why nearby activity cannot prove that an edge belongs to the case.
    """
    # 1. Compare every illicit transaction with the complete transaction export.
    paths = load_paths()
    base = amlsim.read_base_date(paths.amlsim_conf)
    illicit = amlsim.load_alert_transactions(paths.dataset_dir, base)
    tx = pd.read_csv(paths.dataset_dir / "transactions.csv")
    tx["day"] = amlsim.to_day(tx.pop("tran_timestamp"), base)
    marked = tx[tx["is_sar"].astype(str).str.lower().eq("true")]
    cols = ["tran_id", "alert_id", "orig_acct", "bene_acct", "base_amt", "day"]
    if tx.tran_id.duplicated().any() or illicit.tran_id.duplicated().any():
        raise ValueError("duplicate transaction IDs prevent an unambiguous audit")
    pd.testing.assert_frame_equal(
        illicit[cols].sort_values("tran_id").reset_index(drop=True),
        marked[cols].sort_values("tran_id").reset_index(drop=True), check_dtype=False,
    )
    members = amlsim.load_alert_accounts(paths.dataset_dir)
    if members.duplicated(["alert_id", "acct_id"]).any():
        raise ValueError("duplicate alert members")
    member_keys = set(map(tuple, members[["alert_id", "acct_id"]].to_numpy()))
    for side in ["orig_acct", "bene_acct"]:
        if not set(map(tuple, illicit[["alert_id", side]].to_numpy())) <= member_keys:
            raise ValueError("illicit endpoint missing from alert membership")
    gt = build_ground_truth(members, illicit, amlsim.load_alert_members(paths.dataset_tmp_dir))

    # 2. Build independent inbound/outbound evidence from transactions.csv, per case.
    evidence = gt.copy()
    evidence["legacy_positive"] = evidence.role.isin(["relay", "collector", "disperser"])
    evidence["first_direction"] = "missing_side"
    both = evidence.is_mule_candidate
    incoming, outgoing = evidence.first_illicit_inflow_day, evidence.first_illicit_outflow_day
    evidence.loc[both & incoming.lt(outgoing), "first_direction"] = "in_before_out"
    evidence.loc[both & incoming.eq(outgoing), "first_direction"] = "same_day_unknown_order"
    evidence.loc[both & incoming.gt(outgoing), "first_direction"] = "out_before_in"
    evidence["normal_in_during_alert"] = 0
    evidence["normal_out_during_alert"] = 0
    for alert_id, group in evidence.groupby("alert_id"):
        edges = marked[marked.alert_id.eq(alert_id)]
        in_counts, out_counts = edges.bene_acct.value_counts(), edges.orig_acct.value_counts()
        expected_in = group.acct_id.map(in_counts).fillna(0).astype(int)
        expected_out = group.acct_id.map(out_counts).fillna(0).astype(int)
        if not (group.n_illicit_in.eq(expected_in).all() and group.n_illicit_out.eq(expected_out).all()):
            raise ValueError(f"ground truth disagrees with full export for alert {alert_id}")
        if edges.empty:
            continue
        # Nearby normal edges are not evidence of illicit forwarding or funding.
        normal = tx[tx.alert_id.eq(-1) & tx.day.between(edges.day.min(), edges.day.max())]
        evidence.loc[group.index, "normal_in_during_alert"] = group.acct_id.map(
            normal.bene_acct.value_counts()).fillna(0).astype(int)
        evidence.loc[group.index, "normal_out_during_alert"] = group.acct_id.map(
            normal.orig_acct.value_counts()).fillna(0).astype(int)

    # 3. Summarize coverage and retain every changed row for review, without fitting a model.
    selected = evidence[evidence.is_mule_candidate]
    changed = evidence[evidence.legacy_positive & ~evidence.is_mule_candidate]
    duration = selected.completion_day - selected.onset_day
    account_count = len(pd.read_csv(paths.dataset_dir / "accounts.csv", usecols=["acct_id"]))
    summary = {
        "dataset": paths.dataset,
        "rule": "n_illicit_in > 0 AND n_illicit_out > 0 within the same alert_id",
        "accounts": account_count, "alert_members": len(evidence),
        "alerts": int(evidence.alert_id.nunique()), "illicit_edges_verified": len(illicit),
        "legacy_positive_rows": int(evidence.legacy_positive.sum()),
        "positive_rows": len(selected), "positive_accounts": int(selected.acct_id.nunique()),
        "positive_prevalence_all_accounts": selected.acct_id.nunique() / account_count,
        "changed_rows": len(changed), "changed_by_role": changed.role.value_counts().to_dict(),
        "positive_by_typology": selected.typology.value_counts().to_dict(),
        "positive_first_direction": selected.first_direction.value_counts().to_dict(),
        "positive_main_accounts": int(selected.is_main.sum()),
        "changed_collectors_with_nearby_normal_out": int(
            (changed.role.eq("collector") & changed.normal_out_during_alert.gt(0)).sum()),
        "changed_dispersers_with_nearby_normal_in": int(
            (changed.role.eq("disperser") & changed.normal_in_during_alert.gt(0)).sum()),
        "positive_duration_days": {"median": float(duration.median()), "max": int(duration.max())},
        "positive_missing_milestones": int(selected[["onset_day", "completion_day",
            "first_illicit_inflow_day", "first_illicit_outflow_day"]].isna().any(axis=1).sum()),
        "multi_alert_accounts": int(evidence.groupby("acct_id").alert_id.nunique().gt(1).sum()),
        "silent_members": int(evidence.role.eq("no_illicit").sum()),
    }
    return evidence, summary


def main() -> None:
    """Write reproducible row evidence and a JSON summary to the configured processed directory."""
    evidence, summary = audit_labels()
    target = load_paths().processed
    target.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(target / "ground_truth_label_audit.csv", index=False)
    (target / "ground_truth_label_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
