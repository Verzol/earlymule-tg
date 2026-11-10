"""Paired comparison of two daily runs on the primary metric (evaluation protocol section 6).

For every seed both runs have, A's and B's per-mule tables at K are compared with a paired case bootstrap.
A is better than B only if the 95% interval of A - B lies above 0 for every seed.
Run: python -m earlymule.evaluation.compare A B [--commit-a C] [--commit-b C] [--k 10] [--n 1000]
Writes experiments/runs/comparisons/<A>_vs_<B>.csv.
"""

from __future__ import annotations

import argparse

import pandas as pd

from earlymule.config import load_paths
from earlymule.evaluation.uncertainty import paired_bootstrap_difference
from earlymule.experiment_log import read_log, run_folder, select_runs


def compare_tables(a: dict[int, pd.DataFrame], b: dict[int, pd.DataFrame], k: int, n: int) -> pd.DataFrame:
    """Paired difference A - B in detection before forwarding, one row per shared seed.

    Args:
        a, b: seed -> per-mule detection table with a column k (all budgets of one run).
    Returns:
        seed, share_a, share_b, difference, ci_low, ci_high, a_better (interval above 0).
    """
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise ValueError("the two runs share no seed")
    rows = []
    for s in seeds:
        ta, tb = a[s][a[s].k.eq(k)], b[s][b[s].k.eq(k)]
        r = paired_bootstrap_difference(ta, tb, n, s)
        share = lambda t: round(float(t.before_forwarding[t.can_precede_forwarding].mean()), 4)  # noqa: E731
        rows.append({"seed": s, "share_a": share(ta), "share_b": share(tb), "difference": r["difference"],
                     "ci_low": r["difference_ci95"][0], "ci_high": r["difference_ci95"][1],
                     "a_better": r["difference_ci95"][0] > 0})
    return pd.DataFrame(rows)


def load_tables(run_name: str, commit: str | None) -> tuple[str, dict[int, pd.DataFrame]]:
    """Commit and seed -> saved per-mule table of the selected logged runs of run_name."""
    runs = select_runs(read_log(), run_name, commit)
    commit = runs.commit.iloc[0]
    return commit, {int(s): pd.read_parquet(run_folder(run_name, commit, int(s)) / "detection_test.parquet")
                    for s in runs.seed}


def main() -> None:
    """Compare two logged daily runs and print the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--commit-a")
    parser.add_argument("--commit-b")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    # 1. Per-mule tables of both runs, then the paired differences per seed
    commit_a, a = load_tables(args.a, args.commit_a)
    commit_b, b = load_tables(args.b, args.commit_b)
    result = compare_tables(a, b, args.k, args.n).assign(a=f"{args.a}@{commit_a}", b=f"{args.b}@{commit_b}", k=args.k)

    # 2. Save and report; the verdict needs every seed's interval above 0
    out = load_paths().experiments / "runs" / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / f"{args.a}_vs_{args.b}.csv", index=False)
    print(result.drop(columns=["a", "b", "k"]).to_string(index=False))
    verdict = "better than" if result.a_better.all() else "not shown better than"
    print(f"{args.a}@{commit_a} is {verdict} {args.b}@{commit_b} at K = {args.k} ({len(result)} seeds)")


if __name__ == "__main__":
    main()
