"""TreeSHAP of the daily LightGBM trained from the export (EMTG-70): which features drive the score, and where.

For each seed of a run trained by models.lightgbm_export (model.txt and scores_test.parquet as written by
the run, e.g. the downloaded Kaggle output <source>/<run>/seed<N>/), explains three groups of test rows:
  mules at their decision point (end of the day before the first illicit outflow, scored population),
  false alerts (daily top 10 with the 30-day cooldown that are not an active mule), and random rows.
Interventional TreeSHAP against a background of training negatives is the main estimate; LightGBM's
path-dependent pred_contrib is kept as a cross-check, since it can shift credit between correlated features.
Writes per-seed SHAP values next to the model (parquet, git-ignored) and summary tables to
reports/figures/features/.
Run: python -m earlymule.analysis.shap_values --data <export folder> --source <run outputs> [--run lgbm_abl_full]
     [--seeds 0 1 2] [--background 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from earlymule.analysis.eda import alert_outcomes, family_of
from earlymule.config import repo_root
from earlymule.models.lightgbm_export import load_split, train_mask
from earlymule.models.tabular import load_config, load_inputs

KEYS = ["acct_id", "snapshot"]


def explained_rows(test: pd.DataFrame, scores: pd.DataFrame, mules: pd.DataFrame, others: pd.Index,
                   n_false: int, n_random: int, rng: np.random.Generator) -> pd.DataFrame:
    """Test rows to explain, tagged by group: decision-point mules, false alerts, random rows.

    Args:
        test: export test rows with features (load_split), in export order.
        scores: the run's scores_test.parquet (acct_id, snapshot, score).
    """
    # 1. Mules on the day before their first illicit outflow, when they are in the scored rows; a mule that
    #    forwards on its onset day has no such day while active, so it is left out
    waits = mules[mules.first_illicit_outflow_day > mules.onset_day]
    decision = (waits.first_illicit_outflow_day - 1).rename("snapshot").reset_index()
    mule_rows = test.merge(decision, on=KEYS).assign(group="mule, decision point")
    mule_rows["typology"] = mule_rows.acct_id.map(mules.typology)
    # 2. False alerts of the run: top 10 per day with cooldown, landing on anything but an active mule
    alerts = alert_outcomes(scores, mules, others, 10, 30)
    false = alerts[alerts.kind != "active mule"][KEYS + ["kind"]]
    false = false.iloc[rng.permutation(len(false))[:n_false]]
    false_rows = test.merge(false, on=KEYS).assign(group="false alert")
    # 3. Random scored rows as the reference population
    rand_rows = test.iloc[rng.permutation(len(test))[:n_random]].assign(group="random row")
    return pd.concat([mule_rows, false_rows, rand_rows], ignore_index=True)


def family_table(values: pd.DataFrame, features: list[str], by: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature family and group: SHAP values of one family's columns are summed per row first."""
    fam = pd.Series({f: family_of(f) for f in features})
    per_family = values[features].T.groupby(fam).sum().T.abs()
    return per_family.join(values[by]).groupby(by).mean().T


def main() -> None:
    """Explain every seed of one run and write the summary tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="export folder the run was trained on")
    parser.add_argument("--run", default="lgbm_abl_full")
    parser.add_argument("--source", required=True, help="folder holding <run>/seed<N>/ with model.txt and scores")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--background", type=int, default=500)
    parser.add_argument("--false-alerts", type=int, default=1000)
    parser.add_argument("--random-rows", type=int, default=3000)
    args = parser.parse_args()

    # 1. Shared inputs: ground truth, export test rows, training negatives for the background
    data = Path(args.data)
    cfg = load_config("lightgbm_ablation.yaml")
    inputs = load_inputs()
    deposit = pd.read_parquet(data / "nodes.parquet").set_index("acct_id").initial_deposit
    test = load_split(data, "test", deposit)
    train_label = pd.read_parquet(data / "rows_train.parquet", columns=["label"]).label
    out_tables = repo_root() / "reports" / "figures" / "features"
    out_tables.mkdir(parents=True, exist_ok=True)
    tables, ranks = [], []

    for seed in args.seeds:
        # 2. Model, its training sample (same mask as training) and a background of its negatives
        run_dir = Path(args.source) / args.run / f"seed{seed}"
        booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
        features = booster.feature_name()
        rng = np.random.default_rng(seed)
        mask = train_mask(train_label, cfg["train_negative_rate"], np.random.default_rng(seed))
        train = load_split(data, "train", deposit, mask)
        background = train[train.label.eq(0)].sample(args.background, random_state=seed)[features]
        # 3. Rows to explain and their SHAP values (interventional) and pred_contrib (path-dependent)
        rows = explained_rows(test, pd.read_parquet(run_dir / "scores_test.parquet"), inputs["mules"],
                              inputs["other_members"], args.false_alerts, args.random_rows, rng)
        explainer = shap.TreeExplainer(booster, shap.maskers.Independent(background, max_samples=args.background),
                                       feature_perturbation="interventional", model_output="raw")
        sv = pd.DataFrame(explainer.shap_values(rows[features]), columns=features)
        pc = pd.DataFrame(booster.predict(rows[features], pred_contrib=True)[:, :-1], columns=features)
        meta = rows[KEYS + ["group", "typology", "kind"]].reset_index(drop=True).assign(seed=seed)
        pd.concat([meta, sv], axis=1).to_parquet(run_dir / "shap_values.parquet", index=False)
        pd.concat([meta, rows[features].reset_index(drop=True)], axis=1).to_parquet(run_dir / "shap_rows.parquet", index=False)
        # 4. Per-seed summaries: mean |SHAP| by feature and group, by family; agreement with pred_contrib
        both = pd.concat([meta, sv], axis=1)
        tables.append(both.assign(seed=seed))
        a, b = sv.abs().mean(), pc.abs().mean()
        ranks.append({"seed": seed, "spearman_mean_abs": float(a.rank().corr(b.rank())),
                      "top10_overlap": len(set(a.nlargest(10).index) & set(b.nlargest(10).index))})
        print(json.dumps({"seed": seed, "rows": rows.group.value_counts().to_dict(), **ranks[-1]}), flush=True)

    # 5. Tables over all seeds
    allv = pd.concat(tables, ignore_index=True)
    feats = [c for c in allv.columns if c not in ("acct_id", "snapshot", "group", "typology", "kind", "seed")]
    by_feature = allv.groupby("group")[feats].apply(lambda g: g.abs().mean()).T
    by_feature.assign(family=[family_of(f) for f in by_feature.index]).sort_values(
        "mule, decision point", ascending=False).to_csv(out_tables / "shap_mean_abs_by_feature.csv")
    family_table(allv, feats, ["group"]).to_csv(out_tables / "shap_family_by_group.csv")
    family_table(allv[allv.group == "mule, decision point"], feats, ["typology"]).to_csv(
        out_tables / "shap_family_by_typology.csv")
    pd.DataFrame(ranks).to_csv(out_tables / "shap_vs_pred_contrib.csv", index=False)


if __name__ == "__main__":
    main()
