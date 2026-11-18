"""Summary of the transfer-time task (EMTG-72): variants against the full set and against end-of-day alerts.

Reads the logged runs transfer_lgbm_<variant> (experiments.csv) and their per-mule hold tables, and the
end-of-day detection tables of lightgbm_daily (72 features). Comparisons are paired per seed with a case
bootstrap: variants against the full set at the same budget; holds of H accounts a day against end-of-day
alerts of K = H, on the mules both tables cover; holds and end-of-day alerts together (H = K = 10) against
end-of-day K = 20, the same review load.
Writes reports/figures/transfer/summary.csv and comparisons.csv.
Run: python -m earlymule.analysis.transfer_summary --runs <folder with transfer_lgbm_*/seed*/> [--commit d6128f0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from earlymule.config import repo_root
from earlymule.evaluation.uncertainty import paired_bootstrap_difference
from earlymule.experiment_log import read_log, run_folder

SEEDS = (0, 1, 2)
VARIANTS = ["sender", "order", "pair", "payee", "full", "external_payee", "no_shortcuts", "no_churn",
            "conservative", "wide_label"]
DAILY = ("lightgbm_daily", "7bfcfb4")


def summary(log: pd.DataFrame, commit: str) -> pd.DataFrame:
    """Mean and sd over seeds of the main metrics of every variant."""
    rows = []
    for v in VARIANTS:
        runs = log[(log.run_name == f"transfer_lgbm_{v}") & (log.commit == commit)].drop_duplicates("seed", keep="last")
        for _, r in runs.iterrows():
            m = r.metrics if isinstance(r.metrics, dict) else json.loads(r.metrics)  # read_log parses JSON
            row = {"variant": v, "seed": r.seed, "val_pr_auc": m["val_pr_auc"], "test_pr_auc": m["test_pr_auc"],
                   "best_iteration": m["best_iteration"], "features": m["features"]}
            for h in (10, 20, 50):
                t = m[f"h{h}"]["test"]
                row |= {f"h{h}_caught_can_precede": t["caught_can_precede"]["share"],
                        f"h{h}_caught_all": t["caught_all"]["share"], f"h{h}_caught_same_day": t["caught_same_day"]["share"],
                        f"h{h}_holds_per_day": t["holds_per_day"], f"h{h}_false_per_1000": t["false_holds_per_1000_legit"]}
            t = m["h10"]["test"]
            row |= {"h10_first_out_held": t["first_outflow_itself_held"], "h10_amount_stopped": t["first_outflow_amount_stopped"]}
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    agg = per_seed.drop(columns="seed").groupby("variant", sort=False).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reindex(VARIANTS).round(4)


def _holds(runs: Path, variant: str, seed: int, h: int) -> pd.DataFrame:
    """Per-mule hold table on test of one run."""
    return pd.read_parquet(runs / f"transfer_lgbm_{variant}" / f"seed{seed}" / f"holds_test_h{h}.parquet")


def _daily(seed: int, k: int) -> pd.DataFrame:
    """End-of-day detection table on test of the daily LightGBM at budget k."""
    t = pd.read_parquet(run_folder(*DAILY, seed) / "detection_test.parquet")
    return t[t.k.eq(k)]


def comparisons(runs: Path, n: int) -> pd.DataFrame:
    """Paired case-bootstrap differences per seed (points of the share of mules caught before forwarding)."""
    out = []

    def add(name, a, b, hit="before_forwarding", counted="can_precede_forwarding", seed=0):
        d = paired_bootstrap_difference(a, b, n, seed, hit=hit, counted=counted)
        out.append({"comparison": name, "seed": seed, "mules": int(a[counted].sum()) if counted else len(a),
                     "difference": d["difference"], "ci95": d["difference_ci95"]})

    for s in SEEDS:
        # 1. Variants against the full set, same seed and budget, on mules with a day before forwarding
        full = _holds(runs, "full", s, 10)
        for v in VARIANTS:
            if v != "full":
                add(f"{v} - full (H=10)", _holds(runs, v, s, 10), full, seed=s)
        # 2. Holds against end-of-day alerts at the same budget, on the mules both cover
        for h in (10, 20):
            hold, day = _holds(runs, "full", s, h), _daily(s, h)
            both = sorted(set(hold.acct_id) & set(day.acct_id))
            a = hold.set_index("acct_id").loc[both].reset_index()
            b = day.set_index("acct_id").loc[both].reset_index()
            add(f"holds H={h} - end of day K={h}", a, b, seed=s)
        # 3. Both rules at H = K = 10 against end of day K = 20, the same review load
        hold, d10, d20 = _holds(runs, "full", s, 10), _daily(s, 10), _daily(s, 20)
        both = sorted(set(hold.acct_id) & set(d10.acct_id))
        a = d10.set_index("acct_id").loc[both]
        a["before_forwarding"] = a.before_forwarding | hold.set_index("acct_id").loc[both, "before_forwarding"]
        add("holds H=10 + end of day K=10 - end of day K=20", a.reset_index(),
            d20.set_index("acct_id").loc[both].reset_index(), seed=s)
    return pd.DataFrame(out)


def main() -> None:
    """Write the summary and comparison tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="folder holding transfer_lgbm_<variant>/seed<N>/")
    parser.add_argument("--commit", default="d6128f0")
    parser.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    args = parser.parse_args()
    out = repo_root() / "reports" / "figures" / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    s = summary(read_log(), args.commit)
    s.to_csv(out / "summary.csv")
    c = comparisons(Path(args.runs), args.n)
    c.to_csv(out / "comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[[c for c in s.columns if c.startswith(("h10_caught", "val_pr_auc_mean", "test_pr_auc_mean", "h10_false_per_1000_mean"))]])
    print(c.to_string())


if __name__ == "__main__":
    main()
