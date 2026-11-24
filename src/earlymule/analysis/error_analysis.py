"""Error analysis of two daily runs on test: which mules are missed before forwarding, and where the alerts go.

Works on saved outputs only (no retraining): each run's scores_test.parquet and detection_test.parquet, the
ground truth and the 80K export's point-in-time features. Alerts are rebuilt with the protocol's settings
(K new accounts per day, 30-day cooldown).
Run: python -m earlymule.analysis.error_analysis [--runs lightgbm_daily@7bfcfb4 tgn_transition@7bfcfb4] [--k 10]
Writes reports/figures/error_analysis/*.csv and summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from earlymule.config import load_paths
from earlymule.evaluation.early_detection import daily_alerts
from earlymule.experiment_log import run_folder

COOLDOWN = 30
OUT = Path("reports/figures/error_analysis")
FEATURES = ["new_share_in_7d", "new_peers_in_7d", "n_in_7d_over_30d", "peers_in_30d", "n_in_30d", "n_out_30d"]


def missed_rank(scores: pd.DataFrame, alerts: pd.DataFrame, mules: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    """Best rank of each eligible mule among the accounts that could still be alerted, over its days before forwarding.

    Rank 1 = top of the day; a mule is alerted when this rank is within K on some day. Accounts in cooldown are
    left out of the ranking, as daily_alerts does.
    Args:
        mules: detection rows (acct_id, onset_day, first_illicit_outflow_day) of mules that can precede forwarding.
    Returns:
        acct_id, best_rank (NaN if never scored in its gap), best_day, in_cooldown (the mule itself was alerted in
        the cooldown days before best_day, i.e. before its onset, so it could not be alerted again).
    """
    # 1. Days to look at: each mule's days in [onset, first outflow)
    gap = scores.merge(mules[["acct_id", "onset_day", "first_illicit_outflow_day"]], on="acct_id")
    gap = gap[gap.snapshot.ge(gap.onset_day) & gap.snapshot.lt(gap.first_illicit_outflow_day)]
    out = []
    for day, rows in gap.groupby("snapshot"):
        # 2. Accounts alerted in the previous cooldown days are not candidates on this day
        cool = alerts.acct_id[alerts.snapshot.between(day - cooldown + 1, day - 1)]
        today = scores[scores.snapshot.eq(day) & ~scores.acct_id.isin(cool)].score.to_numpy()
        today = np.sort(today)[::-1]
        for acct, s in zip(rows.acct_id, rows.score):
            out.append((acct, day, int(np.searchsorted(-today, -s, side="left")) + 1, bool(cool.eq(acct).any())))
    r = pd.DataFrame(out, columns=["acct_id", "best_day", "best_rank", "in_cooldown"])
    r = r.sort_values(["acct_id", "best_rank"]).drop_duplicates("acct_id")
    return mules[["acct_id"]].merge(r, on="acct_id", how="left")


def alert_kind(alerts: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Label each alert: active mule, mule outside its case days, non-mule case member (by role, during or outside
    its case), or ordinary account.

    The case window of a member is [first onset, last completion] over all members of the case.
    """
    # 1. Case window of every (case, account) pair
    win = gt.groupby("alert_id").agg(start=("onset_day", "min"), end=("completion_day", "max"))
    m = gt[["alert_id", "acct_id", "role", "is_mule_candidate", "onset_day", "completion_day"]].join(win, on="alert_id")
    a = alerts[["acct_id", "snapshot"]].reset_index(drop=True).reset_index().merge(m, on="acct_id", how="left")
    a = a.astype({c: float for c in ["onset_day", "completion_day", "start", "end"]})
    # 2. One label per alert; an account in several cases takes the most specific match
    mule_active = a.is_mule_candidate.eq(True) & a.snapshot.between(a.onset_day, a.completion_day)
    in_case = a.snapshot.between(a.start, a.end)
    kind = np.select([mule_active, a.is_mule_candidate.eq(True), a.role.notna() & in_case, a.role.notna()],
                     ["active mule", "mule outside its case", "member " + a.role.fillna("") + ", during case",
                      "member " + a.role.fillna("") + ", outside case"], "ordinary account")
    order = {"active mule": 0, "mule outside its case": 1}
    a = a.assign(kind=kind, rank=[order.get(k, 2 if "during" in k else 3 if "member" in k else 4) for k in kind])
    return a.sort_values("rank").drop_duplicates("index").set_index("index").sort_index().kind


def known_upstream(gt: pd.DataFrame, export: Path) -> pd.Series:
    """Per mule: True if one of its upstream case members had already sent it money before its onset.

    Such a mule's illicit inflow comes from a known partner, so the counterparty-churn signal stays quiet.
    """
    # 1. Transfers in account ids, receiver side only
    nodes = pd.read_parquet(export / "nodes.parquet")
    ev = pd.read_parquet(export / "events.parquet", columns=["src", "dst", "day"])
    n2a = pd.Series(nodes.acct_id.to_numpy(), index=nodes.node.to_numpy())
    ev = pd.DataFrame({"s": n2a[ev.src].to_numpy(), "d": n2a[ev.dst].to_numpy(), "day": ev.day.to_numpy()})
    # 2. Upstream pairs of each mule, matched against transfers before onset
    up = gt[gt.is_mule_candidate][["acct_id", "onset_day", "upstream_accounts"]].explode("upstream_accounts")
    up = up.rename(columns={"upstream_accounts": "s", "acct_id": "d"}).dropna()
    hit = up.merge(ev, on=["s", "d"])
    known = set(hit.d[hit.day.lt(hit.onset_day.astype(float))])
    return pd.Series({a: a in known for a in gt.acct_id[gt.is_mule_candidate].unique()}, name="upstream_known")


def mule_groups(det: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Eligible mules with the groups used to split misses: typology, days to first outflow, activity quartile.

    Features are those of the decision point, the end of day (first outflow - 1), as in reports/post_run_eda.md.
    """
    m = det[det.can_precede_forwarding].drop_duplicates("acct_id").copy()
    m["decision_day"] = m.first_illicit_outflow_day - 1
    m = m.merge(feats, left_on=["acct_id", "decision_day"], right_on=["acct_id", "snapshot"], how="left")
    m["days_to_outflow"] = (m.first_illicit_outflow_day - m.onset_day).clip(upper=3).map({1: "1", 2: "2", 3: "3+"})
    m["activity_30d"] = m.n_in_30d + m.n_out_30d
    m["activity_quartile"] = pd.qcut(m.activity_30d.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    return m


def main() -> None:
    """Build the tables of both runs and write them with a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["lightgbm_daily@7bfcfb4", "tgn_transition@7bfcfb4"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth and decision-point features of the test rows
    gt = pd.read_parquet(paths.processed / "ground_truth.parquet")
    feats = pd.read_parquet(paths.processed / "kaggle" / paths.dataset / "features_test.parquet",
                            columns=["acct_id", "snapshot", *FEATURES])

    # 2. Per run and seed: per-mule outcome, best rank before forwarding, and the kind of every alert
    outcomes, kinds, ordinary = [], [], []
    for spec in args.runs:
        name, commit = spec.split("@")
        for seed in args.seeds:
            folder = run_folder(name, commit, seed)
            scores = pd.read_parquet(folder / "scores_test.parquet", columns=["acct_id", "snapshot", "score"])
            det = pd.read_parquet(folder / "detection_test.parquet")
            det = det[det.k.eq(args.k)]
            alerts = daily_alerts(scores, args.k, COOLDOWN)
            elig = det[det.can_precede_forwarding]
            rank = missed_rank(scores, alerts, elig, COOLDOWN)
            outcomes.append(elig[["acct_id", "before_forwarding"]].merge(rank, on="acct_id").assign(run=name, seed=seed))
            kind = alert_kind(alerts, gt)
            kinds.append(kind.value_counts().rename("alerts").reset_index().assign(run=name, seed=seed))
            ordinary.append(alerts[kind.eq("ordinary account").to_numpy()].assign(run=name, seed=seed))
        groups = mule_groups(det, feats)
    groups = groups.join(known_upstream(gt, paths.processed / "kaggle" / paths.dataset), on="acct_id")

    # 3. Missed mules by group (mean share caught over seeds, per run)
    cols = ["acct_id", "typology", "days_to_outflow", "activity_quartile", "upstream_known", *FEATURES]
    o = pd.concat(outcomes).merge(groups[cols], on="acct_id")
    o.to_csv(OUT / "mule_outcomes.csv", index=False)
    by = {g: o.groupby(["run", g], observed=True).before_forwarding.mean().unstack(0).round(3)
          for g in ["typology", "days_to_outflow", "activity_quartile", "upstream_known"]}
    for g, t in by.items():
        t.assign(mules=o.drop_duplicates("acct_id").groupby(g, observed=True).size()).to_csv(OUT / f"caught_by_{g}.csv")

    # 4. How far the missed mules were from the budget: best rank bands
    missed = o[~o.before_forwarding]
    # Rank 1-10 and still missed: alerted shortly before onset, so the cooldown blocked the alert in its gap
    bands = pd.cut(missed.best_rank, [0, 10, 20, 50, 100, 1e9],
                   labels=["1-10, in cooldown", "11-20", "21-50", "51-100", ">100"])
    assert missed.in_cooldown[missed.best_rank.le(10)].all(), "a top-10 mule was missed without a cooldown"
    near = missed.assign(band=bands.astype(str).where(missed.best_rank.notna(), "not ranked")) \
        .groupby(["run", "band"]).size().unstack(0).fillna(0).astype(int)
    near.to_csv(OUT / "missed_rank_bands.csv")

    # 5. Mules never caught by either run in any seed, and caught by all
    hits = o.groupby("acct_id").before_forwarding.agg(["sum", "count"])
    never, always = hits.index[hits["sum"].eq(0)], hits.index[hits["sum"].eq(hits["count"])]
    profile = groups.assign(group=np.select([groups.acct_id.isin(never), groups.acct_id.isin(always)],
                                            ["never caught", "always caught"], "sometimes")) \
        .groupby("group")[["activity_30d", *FEATURES]].median().round(3)
    profile = profile.assign(mules=groups.assign(g=np.select([groups.acct_id.isin(never), groups.acct_id.isin(always)],
                                                             ["never caught", "always caught"], "sometimes"))
                             .groupby("g").size())
    profile.to_csv(OUT / "never_vs_always_profile.csv")
    groups[groups.acct_id.isin(never)][["acct_id", "alert_id", "typology", "onset_day", "first_illicit_outflow_day",
                                        "activity_30d", *FEATURES]].to_csv(OUT / "never_caught.csv", index=False)

    # 6. Where the alerts go, and the ordinary accounts that take them
    k = pd.concat(kinds).groupby(["run", "kind"]).alerts.sum().unstack(0).fillna(0)
    (k / k.sum()).round(4).assign(**{f"{c} alerts": k[c].astype(int) for c in k.columns}).to_csv(OUT / "alert_kinds.csv")
    ordn = pd.concat(ordinary).merge(feats, on=["acct_id", "snapshot"], how="left")
    rand = feats[~feats.acct_id.isin(gt.acct_id)].sample(20000, random_state=0)
    prof = pd.concat({"ordinary alerted": ordn[FEATURES].median(), "ordinary random": rand[FEATURES].median()}, axis=1)
    prof.round(3).to_csv(OUT / "ordinary_alert_profile.csv")
    per_acct = ordn.groupby("acct_id").agg(alerts=("snapshot", "size"), runs=("run", "nunique"), seeds=("seed", "nunique"))
    repeat = {"ordinary_alerts": int(len(ordn)), "distinct_accounts": int(len(per_acct)),
              "accounts_in_both_runs": int(per_acct.runs.eq(2).sum()),
              "alerts_on_accounts_in_both_runs": int(per_acct.alerts[per_acct.runs.eq(2)].sum()),
              "accounts_alerted_5_plus": int(per_acct.alerts.ge(5).sum()),
              "share_new_partner_7d": round(float(ordn.new_peers_in_7d.gt(0).mean()), 3),
              "share_new_partner_7d_random": round(float(rand.new_peers_in_7d.gt(0).mean()), 3)}

    # 7. Summary
    summary = {"runs": args.runs, "k": args.k, "seeds": args.seeds, "eligible_mules": int(o.acct_id.nunique()),
               "never_caught": len(never), "always_caught": len(always),
               "caught_by_typology": by["typology"].to_dict(), "caught_by_days_to_outflow": by["days_to_outflow"].to_dict(),
               "caught_by_activity": {r: {str(q): v for q, v in d.items()} for r, d in by["activity_quartile"].to_dict().items()},
               "caught_by_upstream_known": {r: {str(q): v for q, v in d.items()} for r, d in by["upstream_known"].to_dict().items()},
               "never_caught_upstream_known": int(groups.upstream_known[groups.acct_id.isin(never)].sum()),
               "missed_rank_bands": near.to_dict(), "alert_kinds": (k / k.sum()).round(4).to_dict(), "ordinary": repeat}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
