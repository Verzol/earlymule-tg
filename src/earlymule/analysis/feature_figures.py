"""Figures of the feature analysis (EMTG-70): SHAP by family and group, SHAP beeswarm at the decision point,
ablation differences against the full feature set.

Reads the SHAP values written by earlymule.analysis.shap_values (<source>/<run>/seed<N>/shap_values.parquet and
shap_rows.parquet) and reports/figures/features/ablation_summary.csv from earlymule.analysis.ablation_summary.
Writes PNGs to reports/figures/features/.
Run: python -m earlymule.analysis.feature_figures --source <run outputs> [--run lgbm_abl_full] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from earlymule.config import repo_root  # noqa: E402

META = ["acct_id", "snapshot", "group", "typology", "kind", "seed"]


def family_bars(table: pd.DataFrame, path: Path) -> None:
    """Horizontal bars of mean |SHAP| per family, one bar per row group."""
    order = table["mule, decision point"].sort_values().index
    ax = table.loc[order].plot.barh(figsize=(7, 4.5), width=0.8)
    ax.set(xlabel="mean |SHAP| of the family (log-odds)", title="Feature families driving the LightGBM score")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def beeswarm(values: pd.DataFrame, rows: pd.DataFrame, path: Path, top: int = 15) -> None:
    """SHAP beeswarm of the decision-point mules and random rows together, top features by mean |SHAP|."""
    feats = [c for c in values.columns if c not in META]
    keep = values.group.isin(["mule, decision point", "random row"]).to_numpy()
    expl = shap.Explanation(values.loc[keep, feats].to_numpy(), data=rows.loc[keep, feats].to_numpy(),
                            feature_names=feats)
    shap.plots.beeswarm(expl, max_display=top, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.suptitle("SHAP values, decision-point mules and random rows (3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_chart(summary: pd.DataFrame, path: Path) -> None:
    """Change in detection before forwarding against the full set, per ablation: seed differences and their CIs."""
    s = summary[summary.ablation != "full"].copy()
    parsed = s.diff_per_seed.map(lambda t: [tuple(map(float, re.findall(r"[-+]?\d*\.\d+", part)))
                                            for part in t.split(" / ")])
    s = s.assign(parsed=parsed).sort_values("diff_mean")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    for i, (_, r) in enumerate(s.iterrows()):
        for j, (d, lo, hi) in enumerate(r.parsed):
            y = i + (j - 1) * 0.22
            ax.plot([lo * 100, hi * 100], [y, y], color="grey", lw=1)
            ax.plot(d * 100, y, "o", color="C3" if hi < 0 else "C0", ms=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(s)), s.ablation)
    ax.set(xlabel="difference in mules detected before forwarding (points), per seed with 95% CI",
           title="Leaving features out, against the full set (same negative sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Draw every figure whose inputs exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="folder holding <run>/seed<N>/ with the SHAP parquet files")
    parser.add_argument("--run", default="lgbm_abl_full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    out = repo_root() / "reports" / "figures" / "features"

    # 1. SHAP figures from the per-seed parquet files
    dirs = [Path(args.source) / args.run / f"seed{s}" for s in args.seeds]
    values = pd.concat([pd.read_parquet(d / "shap_values.parquet") for d in dirs], ignore_index=True)
    rows = pd.concat([pd.read_parquet(d / "shap_rows.parquet") for d in dirs], ignore_index=True)
    family_bars(pd.read_csv(out / "shap_family_by_group.csv", index_col=0), out / "shap_family_by_group.png")
    beeswarm(values, rows, out / "shap_beeswarm_decision_point.png")
    # 2. Ablation chart once the summary has more than the full set
    summary = out / "ablation_summary.csv"
    if summary.exists() and len(pd.read_csv(summary)) > 1:
        ablation_chart(pd.read_csv(summary), out / "ablation_differences.png")


if __name__ == "__main__":
    main()
