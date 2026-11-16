"""Summary of the LightGBM feature ablation (EMTG-70): each ablation against the full feature set, paired per seed.

Reads the logged runs lgbm_abl_<ablation> (imported with earlymule.evaluation.import_scores) and their
per-mule tables. For every ablation of configs/lightgbm_ablation.yaml and every seed, the difference in
detection before forwarding at K = 10 against lgbm_abl_full of the same seed, which was trained on the same
negative sample, with a paired case-bootstrap interval; plus validation and test PR-AUC and the change per
typology. An ablation "hurts" when every seed's interval lies below 0.
Writes reports/figures/features/ablation_summary.csv and ablation_by_typology.csv.
Run: python -m earlymule.analysis.ablation_summary [--commit 956b8a1] [--k 10] [--n 1000]
"""

from __future__ import annotations

import argparse

import pandas as pd

from earlymule.config import repo_root
from earlymule.evaluation.compare import compare_tables, load_tables
from earlymule.experiment_log import read_log, select_runs
from earlymule.models.tabular import load_config


def pr_auc(log, run: str, commit: str) -> dict[str, float]:
    """Mean validation and test PR-AUC (scored population) over the logged seeds of one run."""
    m = select_runs(log, run, commit).metrics  # read_log parses the JSON columns
    return {"val_pr_auc": round(float(m.map(lambda x: x["val"]["ranking"]["pr_auc"]).mean()), 4),
            "test_pr_auc": round(float(m.map(lambda x: x["test"]["ranking"]["pr_auc"]).mean()), 4)}


def by_typology(table: pd.DataFrame, k: int) -> pd.Series:
    """Share detected before forwarding per typology, mules that leave a day for it."""
    t = table[table.k.eq(k) & table.can_precede_forwarding]
    return t.groupby("typology").before_forwarding.mean()


def main() -> None:
    """Compare every ablation with the full set and write the two tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="956b8a1")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    cfg, log = load_config("lightgbm_ablation.yaml"), read_log()
    out = repo_root() / "reports" / "figures" / "features"
    out.mkdir(parents=True, exist_ok=True)

    # 1. The full feature set: per-mule tables per seed and its typology shares
    _, full = load_tables("lgbm_abl_full", args.commit)
    full_typ = pd.concat({s: by_typology(t, args.k) for s, t in full.items()}, axis=1).mean(axis=1)
    rows, typ = [], {}
    for ablation, spec in cfg["ablations"].items():
        run = f"lgbm_abl_{ablation}"
        try:
            _, tables = load_tables(run, args.commit)
        except ValueError:
            print(f"{run}: not logged yet")
            continue
        # 2. Paired difference per seed against the full set on the same negative sample
        cmp = compare_tables(tables, full, args.k, args.n) if ablation != "full" else None
        row = {"ablation": ablation, "dropped": ", ".join(spec.get("families", []) + spec.get("windows", [])) or "-",
               "seeds": len(tables), **pr_auc(log, run, args.commit),
               "before_forwarding": round(float(pd.Series({s: t[t.k.eq(args.k) & t.can_precede_forwarding]
                                                            .before_forwarding.mean() for s, t in tables.items()}).mean()), 4)}
        if cmp is not None:
            row |= {"diff_mean": round(float(cmp.difference.mean()), 4),
                    "diff_per_seed": " / ".join(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                                for d, lo, hi in zip(cmp.difference, cmp.ci_low, cmp.ci_high)),
                    "hurts_all_seeds": bool((cmp.ci_high < 0).all()), "helps_all_seeds": bool((cmp.ci_low > 0).all())}
        rows.append(row)
        # 3. Per typology: change of the share against the full set, mean over seeds
        typ[ablation] = pd.concat({s: by_typology(t, args.k) for s, t in tables.items()}, axis=1).mean(axis=1) - full_typ

    summary = pd.DataFrame(rows).sort_values("diff_mean", na_position="first")
    summary.to_csv(out / "ablation_summary.csv", index=False)
    pd.DataFrame(typ).T.round(4).to_csv(out / "ablation_by_typology.csv")
    print(summary.drop(columns=["diff_per_seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
