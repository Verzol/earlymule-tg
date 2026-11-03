"""Audit labels against both AMLSim transaction exports and write per-member evidence.

Run: python -m earlymule.data.ground_truth_audit
Inputs: configured AMLSim outputs and generator members. Outputs: ground_truth_label_audit.csv
and ground_truth_label_audit.json under data/processed; no feature or training input is produced.
Layered mule typologies are also checked against their generator plan (layered_flow_checks).
"""

from __future__ import annotations

import json

import pandas as pd

from earlymule.config import load_paths
from earlymule.data import amlsim
from earlymule.data.ground_truth import build_ground_truth


def planned_edges(members: pd.DataFrame) -> pd.DataFrame:
    """Planned illicit edges of layered alerts, by the simulator's rule.

    Consecutive layers are linked all-to-all when either has one account, otherwise
    one-to-one by file order (LayeredMuleTypology.java).
    Args:
        members: amlsim.load_alert_members output, in file order.
    Returns:
        DataFrame with alert_id, orig_acct, bene_acct, from_layer.
    """
    rows = []
    layered = members[members.planned_layer.notna()]
    for alert_id, group in layered.groupby("alert_id", sort=False):
        layers = [group.acct_id[group.planned_layer.eq(l)].tolist()
                  for l in range(int(group.planned_layer.max()) + 1)]
        for l, (orig, bene) in enumerate(zip(layers, layers[1:])):
            if len(orig) == 1 or len(bene) == 1:
                pairs = [(o, b) for o in orig for b in bene]
            elif len(orig) == len(bene):
                pairs = list(zip(orig, bene))
            else:
                raise ValueError(f"alert {alert_id}: layers {l} and {l + 1} cannot be paired")
            rows += [(alert_id, o, b, l) for o, b in pairs]
    return pd.DataFrame(rows, columns=["alert_id", "orig_acct", "bene_acct", "from_layer"])


def layered_flow_checks(gt: pd.DataFrame, illicit: pd.DataFrame, members: pd.DataFrame) -> dict:
    """Compare observed illicit flows of layered alerts with the generator plan.

    A planned mule is any account between the first (sources) and last (exits) layer.
    Args:
        gt: build_ground_truth output (with planned_layer).
        illicit: alert transactions with alert_id, orig_acct, bene_acct, base_amt.
        members: amlsim.load_alert_members output, in file order.
    Returns:
        Counts per check; empty layered runs give zeros.
    """
    # 1. Planned edges versus observed illicit edges of layered alerts
    plan = planned_edges(members)
    keys = ["alert_id", "orig_acct", "bene_acct"]
    observed = illicit[illicit.alert_id.isin(plan.alert_id)][keys + ["base_amt"]]
    joined = plan.merge(observed.drop_duplicates(keys), on=keys, how="left", indicator=True)
    unplanned = observed.merge(plan[keys], on=keys, how="left", indicator=True)

    # 2. Planned roles from layers; label still comes from observed flows
    lay = gt[gt.planned_layer.notna()].copy()
    last = lay.groupby("alert_id").planned_layer.transform("max")
    mules = lay[lay.planned_layer.gt(0) & lay.planned_layer.lt(last)]
    ends = lay[lay.planned_layer.eq(0) | lay.planned_layer.eq(last)]
    two_sided = mules[mules.is_mule_candidate]

    # 3. Each mule should pass on what it received minus the simulator margin
    amt_in = observed.groupby(["alert_id", "bene_acct"]).base_amt.sum()
    amt_out = observed.groupby(["alert_id", "orig_acct"]).base_amt.sum()
    idx = pd.MultiIndex.from_frame(two_sided[["alert_id", "acct_id"]])
    ratio = amt_out.reindex(idx).to_numpy() / amt_in.reindex(idx).to_numpy()
    # Plain bool before grouping: the categorical _merge column fails to aggregate when empty
    complete = joined._merge.eq("both").groupby(joined.alert_id).all()
    typology = lay.drop_duplicates("alert_id").set_index("alert_id").typology
    return {
        "layered_alerts": int(lay.alert_id.nunique()),
        "complete_alerts_by_typology": {k: int(v) for k, v in complete.groupby(typology).sum().items()},
        "planned_edges": len(plan),
        "planned_edges_observed": int(joined._merge.eq("both").sum()),
        "unplanned_illicit_edges": int(unplanned._merge.eq("left_only").sum()),
        "planned_mules": len(mules),
        "planned_mules_two_sided": len(two_sided),
        "planned_ends_as_candidates": int(ends.is_mule_candidate.sum()),
        "mules_out_before_in": int(two_sided.first_illicit_outflow_day.lt(two_sided.first_illicit_inflow_day).sum()),
        "mules_same_day": int(two_sided.first_illicit_outflow_day.eq(two_sided.first_illicit_inflow_day).sum()),
        "mule_forward_ratio": ({"min": float(ratio.min()), "median": float(pd.Series(ratio).median()),
                                "max": float(ratio.max())} if len(ratio) else {}),
    }


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
    generator_members = amlsim.load_alert_members(paths.dataset_tmp_dir)
    gt = build_ground_truth(members, illicit, generator_members)

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
        "layered": layered_flow_checks(gt, illicit, generator_members),
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
