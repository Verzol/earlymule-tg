# Evaluation protocol v0.3

Version 0.3, 2026-11-10. Fixes how every model in the thesis is scored, so that LightGBM, the graph baselines and EarlyMule-TG are compared under the same rules. The dataset, labels and allowed columns are in `docs/data_spec.md`. Choices that stay within the research plan are decisions, each with its reason; the one change to a definition in the plan, the primary early-detection metric, was sent to the advisor on 2026-11-09 and, with no reply by 2026-11-12, is applied as proposed (section 8). Section 9 lists what is not implemented yet.

v0.2 answered two reviews of v0.1: finished mules stay in the ranking, the alert budget counts new accounts with a 30-day cooldown, the primary metric is detection before the first forwarding, model comparisons use a paired case bootstrap, and ties and the start of the alert history follow fixed rules. v0.3 adds the research plan's threshold rule, its secondary metrics and per-typology table, results over all accounts for the daily task, and early stopping on validation PR-AUC only (section 7). All logged runs were redone under these rules (commit `c62fa61`).

## 1. Task

At the end of each day t the model gives every scored account a suspicion score, using only transfers dated on or before t. Accounts are ranked by score and an analyst reviews K new accounts of the day.

- **Main task: daily early detection.** Score every day; the question is whether a mule is alerted before it forwards the illicit money.
- **Secondary task: weekly ranking.** Score every 7 days; positive if the mule's onset fell in the last 7 days. Kept as a sanity check only: LightGBM reaches Recall@100 near 0.97 there, so it cannot separate stronger models (`reports/difficulty_check.md`).

## 2. Data split

| Split | Days | Use |
|---|---|---|
| Train | 0–431 (60%) | Fitting; daily snapshots from day 30, when the 30-day window is full |
| Validation | 432–539 (15%) | Early stopping, hyperparameters, feature and design choices |
| Test | 540–719 (25%) | Final numbers only; never used to choose anything |

One fixed split, three seeds, instead of three rolling time folds: the research plan (section 21.4) allows either, and one split keeps the full 180-day test period with its 219 mules. Features of a day never read later transfers (unit-tested). The simulation ends on day 719 and the last mule completes on day 717, so no test case is cut off by the end of the data.

## 3. Ranking population and labels

Two populations are kept apart:

- **Ranked population:** every regular account on day t (section 4). This is all a bank could rank: nothing in it depends on ground truth.
- **Labelled rows:** the subset that enters label-based metrics. A row without a label is still ranked and can take an alert slot.

**Daily labels.** A mule is positive from its onset day to its completion day (first and last illicit transfer), both included. Before onset it is a negative. After completion it has no label: it stays in the ranking, an alert on it counts as a miss for alert precision, and its rows are left out of PR-AUC.

**Weekly labels.** Positive if onset falls in (t − 7, t]; negative if onset is later; no label if onset is earlier (it was due at an earlier snapshot), again still ranked.

**Mules across the start of the test period** (3 mules start before day 540 and are still active on it): their active test days are positives for PR-AUC and alert precision, but they are not in the detection denominators, because their onset falls outside the test period.

## 4. Scored population

**Regular accounts:** at least 20 transfers (any kind) up to day t. Results on all accounts are reported next to them, never alone.

Reason: case members were drawn from active accounts, so over all accounts activity alone separates mules (ROC-AUC 0.91 for a model on transfer counts); on regular accounts it does not (ROC-AUC 0.55, PR-AUC 0.0008 against a mule rate of 0.0006). The model is trained on the same population: on validation this raised daily PR-AUC from 0.26 to 0.55.

**Hard negatives (kept).** Sources, exits and one-sided fan hubs of laundering cases stay as negatives in the main results, because a bank does not know which accounts are in a case. They take about 30% of daily alerts at K = 10, and removing them raises weekly PR-AUC from 0.80 to 0.89.

## 5. Metrics

Day resolution throughout: an alert on day d uses transfers up to the end of day d, and within a day the order of transfers is unknown.

**Alerts.** Each day t the K highest-ranked accounts that are out of cooldown are alerted (budget unit: new accounts per day). An account last alerted on day a is out of cooldown on day t when t − a ≥ 30: alerted on day d, it is skipped on days d + 1 to d + 29 and can be alerted again from day d + 30; its slot goes to the next account. 30 days equals the longest feature window. A mule alerted shortly before its onset is therefore suppressed during its case and can be missed, as in practice.

**Ranking order.** Score descending, ties broken by `acct_id` ascending, the same for every model and every top-K metric (`rank_by_score` in `src/earlymule/evaluation/metrics.py`). In the current LightGBM test scores no day has a tie between ranks 10 and 11, so the rule does not change any reported number.

**Start of the alert history.** The cooldown history starts empty on the first day of each split (day 432 for validation, day 540 for test), so every split is scored on its own and every model under the same condition. Sensitivity (seed 0, K = 10): carrying over the validation alerts of the same model changes 50 of the 1,800 test alerts; detected before forwarding stays 0.763, before completion moves from 0.653 to 0.648, alert precision from 0.119 to 0.117.

**Which mules count.** Mules whose onset falls in the test period and who are in the ranked population on at least one active day (219 in the current test set).

| Metric | Definition | Mules counted | Unit |
|---|---|---|---|
| **Detected before forwarding** (primary) | First alert d on or after onset with d < first illicit outflow day | Mules whose first outflow is after their onset day (173); also reported: how many of them are in the ranked population on at least one day before the first outflow (173) | share of mules |
| Detected before completion | Same, d < completion day | All counted mules; also multi-day mules only (179) | share of mules |
| Detected by completion | Same, d ≤ completion day | All counted mules | share of mules |
| Time to detect (TTD) | d − onset for the first alert | Mules detected by completion; median | days |
| Lead time | completion − d | Same; median | days |
| Alert precision | Alerts on a mule during its active days ÷ all alerts | — | share of alerts |
| Alerts on hard negatives | Alerts on non-mule case members ÷ all alerts | — | share of alerts |
| Distinct accounts alerted | Number of different accounts among all alerts | — | accounts |
| PR-AUC | Average precision over all labelled (account, day) rows, pooled | — | 0–1; random = mule rate |
| Recall@K (weekly) | Share of positives in the top K = 100 of their own snapshot; no cooldown | — | share of positives |
| Per typology | Before forwarding, by completion, median TTD, median lead time, and the share of all alerts that hit an active mule of the typology (research plan section 24) | As above, per typology | — |

**Why "before forwarding" is primary.** A mule caught before its first illicit outflow can still be stopped before the money leaves; one caught later, but before completion, may already have passed most of it on. The metric needs at least one day between onset and first outflow; mules that receive and forward on the same day, and the 40 cycle initiators (they send first and receive the loop's return later, so onset is their first outflow), cannot be caught before forwarding at day resolution. They still count in the other detection metrics; detection before forwarding and by completion are also broken down by typology. A mule can also miss the chance because it only joins the ranked population (20 transfers) after its first outflow; each run reports this count so that a population limit is not read as a model error. In the current test set all 173 are rankable before their first outflow.

**Alert budget.** K = 5, 10, 20 and 50 new accounts per day are all reported; **K = 10 is the headline**, as an assumed daily review capacity. About 1.2 mules start per test day (219 in 180 days), and a case lasts at most 19 days, so under the cooldown each mule can be hit by at most one alert: alert precision is capped near 222 / 1,800 = 0.12 at K = 10. Precision is therefore reported but not used to compare models.

**Research plan's threshold rule.** The plan detects a mule on the first day its score reaches a threshold θ chosen on validation (plan section 24). θ is the score with the best F1 over labelled validation rows: the plan names no criterion, and F1 weighs missed mules and false alerts equally without a cost ratio. Every account-day at or above θ is an alert, with no daily budget and no cooldown. The same detection metrics are reported for it, plus alerts per day and the plan's secondary metrics (section 23.2) over labelled test rows: precision, recall, F1, F1-macro, MCC and balanced accuracy.

**Why TTD does not rank models here.** At day resolution the median TTD is 0 days for every typology: most detected mules are alerted on their onset day. The plan's KPI of a 10% lower median TTD therefore cannot be met by any model; detection before forwarding and lead time carry the timing comparison, and TTD is still reported.

**Sanity check.** Mules alerted in the 7 days before their onset are counted; a high count would point to leakage.

## 6. Uncertainty and comparisons

- Three seeds (0, 1, 2) per model; report mean ± standard deviation.
- 95% intervals by case bootstrap on test (1,000 resamples). Scores and alerts are computed once and kept fixed; only the mules are resampled, whole laundering cases at a time, because mules of one case move together. Negatives, including the non-mule members of a case, are not resampled.
- **Comparing two models (A against B).** On the primary metric at K = 10, for each seed s compare A's run with seed s to B's run with seed s: draw 1,000 resamples of the cases, compute both models' shares on the same draw, and take the 2.5th and 97.5th percentiles of A − B (`paired_bootstrap_difference` in `src/earlymule/evaluation/uncertainty.py`, on the per-mule tables that each daily run saves). A is called better than B only if this interval lies above 0 for all three seeds. Other metrics are reported, not used for the verdict.

## 7. Rules for every model

1. Same split, ranked population, labels, alert rule and metrics as above; graph models build the graph of day t from transfers on or before t.
2. Inputs only from the columns allowed in `docs/data_spec.md` section 6 (checked in code by `src/earlymule/data/leakage.py`; the 13 leakage checks of the research plan are in `reports/data_leakage_audit.md`).
3. Every choice (features, hyperparameters, training population, early stopping) is made on validation. Early stopping tracks validation PR-AUC only, with a patience of 1,000 rounds: on the weekly task with seed 1, validation PR-AUC was 0.742 at round 3, fell to 0.53 by round 50 and only reached its best, 0.797, at round 1,416.
4. Every run is logged in `experiments/experiments.csv` with commit, seed, config and metrics; a run from uncommitted code is marked dirty and not reported.
5. Scores are ranked as log-odds or raw scores, not saturated probabilities, which avoids most ties; remaining ties follow the ranking order above.

**Current reference (LightGBM, daily, test, 3 seeds, K = 10, 30-day cooldown, commit `c62fa61`):** detected before forwarding 0.761 ± 0.003 (seed 0: 0.763, interval [0.697, 0.831]); before completion 0.651 (multi-day mules 0.797); by completion 0.959; PR-AUC 0.650 ± 0.004; alert precision 0.118 (ceiling about 0.12); 30% of alerts on hard negatives; about 1,660 distinct accounts among 1,800 alerts; median TTD 0 days, lead time 2 days; 1 mule alerted in the 7 days before its onset. Before forwarding by typology (seed 0): M3 and M4 0.86, M1 0.78, M2 0.76, cycle 0.42; median lead time M1 and M3 4 days, M4 2, M2 and cycle 0.

Alongside: K = 5, 20 and 50 catch 0.678, 0.821 and 0.842 before forwarding. Over all accounts, K = 10 catches 0.631 ± 0.062 (PR-AUC 0.570); without hard negatives it catches 0.802 ± 0.019 (PR-AUC 0.741). The threshold rule catches 0.497 ± 0.045 with 7.2 alerts per day and alert precision 0.348; at θ, F1 0.626, MCC 0.639, balanced accuracy 0.761. Weekly sanity check: PR-AUC 0.780 ± 0.024 on regular accounts, Recall@100 0.968.

**TGN baseline (daily, test, 3 seeds, K = 10, 30-day cooldown, commit `12addab`, trained on Kaggle):** detected before forwarding 0.713 ± 0.033; by completion 0.822; PR-AUC 0.437 ± 0.020; ROC-AUC 0.969; Recall@100 0.781; alert precision 0.102; 32% of alerts on hard negatives. K = 5, 20 and 50 catch 0.638, 0.763 and 0.782 before forwarding; the threshold rule catches 0.472 ± 0.009 with 7.8 alerts per day. Before forwarding by typology (mean of 3 seeds): M3 0.88, M4 0.75, M1 0.74, M2 0.65, cycle 0.40 (0.25 to 0.58 across seeds). Paired against LightGBM at K = 10: seed 0 −1.2 points [−7.7, +5.3], seed 1 −6.9 [−12.3, −1.2], seed 2 −6.4 [−12.1, −0.6]; TGN is not better, and worse in two of three seeds. Early stopping on validation PR-AUC with a patience of 6 epochs, at most 30: validation PR-AUC swings between epochs (seed 0: 0.398, 0.325, 0.261, 0.372 over four epochs), and with a patience of 3 seed 0 stopped at epoch 8 while seed 1 was still improving at epoch 14. Best epochs 11, 14 and 8.

**TGN with the LightGBM features (commit `d334e9e`, `configs/tgn_features.yaml`, trained on Kaggle):** the account head also reads the 42 daily LightGBM features of each row (log1p, scaled with train-row statistics, a flag for each column that can be missing). Same split, rows, training loop and early stopping as the TGN baseline. Detected before forwarding 0.825 ± 0.009; before completion 0.694; by completion 0.973; PR-AUC 0.561 ± 0.004; ROC-AUC 0.988; Recall@100 0.886; alert precision 0.120; 32% of alerts on hard negatives. K = 5, 20 and 50 catch 0.738, 0.846 and 0.854; the threshold rule catches 0.557 ± 0.012 with 9.1 alerts per day. Before forwarding by typology (mean of 3 seeds): M3 0.94, M4 0.90, M1 0.80, M2 0.78, cycle 0.60. Paired against LightGBM at K = 10: +6.9 points [+1.7, +12.5], +5.2 [+1.2, +9.6], +6.9 [+2.7, +12.3] for seeds 0 to 2; the interval is above 0 for all three, so by section 6 it is better than LightGBM. Against the TGN baseline: +8.1, +12.1 and +13.3 points, also above 0 for all three. PR-AUC stays below LightGBM: the gain is in the top of each day's ranking, which is what the alert budget reviews. Best epochs 12, 12 and 16; 138 to 163 minutes per seed.

**Control without the graph (commit `a4f49a6`, `configs/mlp_features.yaml`):** the same account head, features, training loop, negative sampling and early stopping, without memory, attention or link loss. Detected before forwarding 0.701 ± 0.039; PR-AUC 0.399 ± 0.055. Paired against LightGBM: −7.5 [−12.6, −2.8], −9.3 [−14.7, −4.6], −1.2 [−4.4, +1.8], not better. The TGN with features beats it by +14.5 [+8.1, +20.9], +14.5 [+8.9, +20.7] and +8.1 [+3.5, +13.1] points. The neural head alone does not reach LightGBM, so the gain of the TGN with features comes from the graph part. Learning rate of the control, chosen on mean validation PR-AUC over the 3 seeds (commit `a4f1e03`, logged as `mlp_features_lr<X>`): 0.0001 0.339, 0.0003 0.339, 0.001 0.391, 0.003 0.355, 0.01 0.292, 0.03 0.312, so the control keeps 0.001; only the control was tuned, which favours it. Test detection before forwarding does not follow validation here (0.744 at 0.01, the highest of the sweep); even that rate, picked on test only as a sensitivity check, is not better than LightGBM (−2.3, −1.7, −1.2 points, intervals across 0) and is beaten by the TGN with features (+9.3 [+5.0, +13.6], +6.9 [+1.8, +11.9], +8.1 [+3.9, +13.5]).

**Daily GraphSAGE (commit `f5f8af9`, `configs/graphsage.yaml`, trained on Kaggle):** the plan's static-graph row. The graph of day t holds the transfers of days (t − 30, t], one directed edge per sender–receiver pair; node inputs are the deposit and the window's in/out counts and amounts; two SAGEConv layers over in-edges and out-edges; the account head reads the embedding, the deposit and the 42 row features, as the TGN with features does. Same rows, negative sampling and early stopping; days are independent, so a seed takes 5 to 7 minutes on a T4 (best epochs 14, 16 and 9). Detected before forwarding 0.761 ± 0.026 (seeds 0.786, 0.734, 0.763); before completion 0.658; by completion 0.974; PR-AUC 0.280 ± 0.111 (0.351, 0.152, 0.337); ROC-AUC 0.993; Recall@100 0.835; alert precision 0.120; 32% of alerts on hard negatives. K = 5, 20 and 50 catch 0.547, 0.832 and 0.830. Before forwarding by typology (mean of 3 seeds): M3 0.86, M1 0.76, M4 0.76, M2 0.73, cycle 0.64. Paired against LightGBM at K = 10: +2.3 points [−2.9, +7.3], −2.9 [−8.9, +3.0], +0.6 [−4.5, +6.1], neither is better. Against the TGN with features: −4.6 [−10.8, +1.7], −8.1 [−14.5, −1.7], −6.4 [−12.3, −1.1]; the TGN with features is ahead in two of three seeds, not in all three, so section 6 does not call it better. Against the control without the graph: +9.8 [+5.5, +14.4], +6.4 [+2.1, +11.3], +1.7 [−4.2, +7.7]. Reading: with the same features and head, a 30-day graph brings the neural model level with LightGBM at the alert budget, where the control without the graph stays below it; the TGN's memory over the whole history adds a further 5 to 8 points in two seeds. PR-AUC is low and unstable across seeds (0.15 to 0.35) while the top of the daily ranking is stable, and the test logits reach 30 to 39, so the validation threshold θ gives 10 to 47 alerts per day on test (threshold rule 0.466 ± 0.043): the scores rank well at the top but are not calibrated. Not tuned (learning rate, window, layers at their TGN values), as the other baselines.

**Feature version 2: 72 features, all families of the research plan (2026-11-13).** The paragraphs above use the 42 features of version 1. Version 2 adds burstiness, rapid-forward ratios, dwell time, two-hop flow, counterparty entropy and signed flow imbalance per window (`docs/data_spec.md` section 7; code `e24d98e`, export `9d2d4d0`). With 72 features the daily LightGBM still improved on validation at its old cap of 3,000 rounds (seed 0, best_iteration = 3,000), so the cap became 10,000 (`7bfcfb4`); early stopping now picks rounds 4,924, 1,017 and 2,867. Results below are on test, K = 10, 30-day cooldown, mean ± sd over 3 seeds; STA-GT seed 0 ran out of GPU memory sharing a T4 with seed 2 and was rerun alone (kernel `earlymule-stagt-seed0`, same commit).

| Model | Commit | Before forwarding | K = 5 / 20 | Before completion | PR-AUC | Threshold rule |
|---|---|---|---|---|---|---|
| LightGBM | `7bfcfb4` | 0.832 ± 0.015 | 0.759 / 0.861 | 0.702 | 0.694 ± 0.005 | 0.609 |
| Control without the graph | `9d2d4d0` | 0.775 ± 0.021 | 0.617 / 0.819 | 0.665 | 0.509 ± 0.013 | 0.403 |
| STA-GT-inspired | `9d2d4d0` | 0.761 ± 0.030 | 0.599 / 0.823 | 0.656 | 0.480 ± 0.070 | 0.426 |
| TGN with features (A3) | `9d2d4d0` | 0.825 ± 0.019 | 0.744 / 0.850 | 0.688 | 0.592 ± 0.052 | 0.565 |
| + role state (A4, `tgn_role`) | `7bfcfb4` | 0.825 ± 0.023 | 0.763 / 0.844 | 0.691 | 0.529 ± 0.004 | 0.511 |
| + role transition (A5, `tgn_transition`) | `7bfcfb4` | 0.848 ± 0.009 | 0.746 / 0.865 | 0.708 | 0.564 ± 0.011 | 0.551 |
| Full EarlyMule-TG, gated fusion (A6, `earlymule_tg`) | `7bfcfb4` | 0.809 ± 0.035 | 0.713 / 0.832 | 0.683 | 0.544 ± 0.014 | 0.530 |

Paired comparisons (section 6, difference in points, 95% interval, seeds 0 / 1 / 2):
- The new features lift LightGBM by +5.2 [+2.4, +8.3], +7.5 [+3.8, +11.6], +8.7 [+4.9, +12.7] over version 1: better in all three seeds. They lift the control without the graph by +9.3, +12.1, +0.6.
- With version 2, no neural model is better than LightGBM in all three seeds. TGN with features: +2.3, −3.5, −1.2, all intervals across 0. The version 1 finding, TGN with features better than LightGBM, does not hold once LightGBM has the full feature set.
- The graph still adds over the same head without it: TGN with features vs the control +5.8 [+1.2, +10.3], +1.2 [−3.3, +5.6], +8.1 [+3.4, +13.3] (two of three seeds). STA-GT vs LightGBM: −8.7 [−14.1, −3.7], −6.9 [−12.0, −2.1], −5.8 [−10.5, −1.3], worse in all three seeds; against the control without the graph −5.2, −2.3, +3.5.
- Ablation steps: role state vs TGN with features 0.0, −0.6, +0.6 (no change); role transition vs role state +1.2 [−2.1, +4.6], +4.0 [0.0, +7.9], +1.7 [−1.8, +5.8]; role transition vs TGN with features +1.2, +3.5 [0.0, +7.4], +2.3; role transition vs LightGBM +3.5 [−1.3, +8.6], 0.0, +1.2. The transition step has the highest mean and the smallest spread of all models, and is ahead of every step before it in all seeds, but no interval clears 0, so by section 6 it is not shown better.
- Gated fusion vs role transition: −8.1 [−12.4, −4.1], −1.2, −2.3; worse in seed 0, and its spread is the largest (seed 0 stopped at its best epoch 1 of 7). The research plan's gated fusion lowers the result instead of adding to it.
- By typology (before forwarding, mean of seeds): the transition step reaches M3 0.94, M2 0.84, M1 0.83, M4 0.90, cycle 0.62, against LightGBM 0.86, 0.87, 0.82, 0.93, 0.58: ahead on M3 and cycle, behind on M2 and M4.
- Collapse checks (`run_meta.json`, validation rows): the role vectors keep a spread (mean sd 0.40 to 0.49 for role state, 0.15 to 0.47 for transition), but the window attention often settles on one window (7 days: α ≈ 0.92 in seed 2 of both steps; 1 day: α ≈ 0.73 in seed 1 of the transition step). The gate of the full model averages 0.28 to 0.39, so most of the fused vector comes from the role branch.
- Training length: the transition step reached its best epoch at 27 of at most 30 in seed 0, so the epoch cap may bind there; the role state and full model stopped early in every seed.

**Effect of the v0.2 rules** (seed 1, K = 10, detected before forwarding / alert precision): v0.1 rules 0.67 / 0.33; finished mules kept in the ranking, no cooldown 0.55 / 0.28; finished mules removed, 30-day cooldown 0.77 / 0.12; v0.2 (both) 0.76 / 0.12. Removing finished mules from the ranking inflated the v0.1 numbers; the cooldown removes repeated alerts, which also keeps finished mules out of the top K, since they were usually alerted during their case.

## 8. Decisions

| # | Decision | Reason | Status |
|---|---|---|---|
| 1 | Main task: daily early detection; primary metric: share of mules alerted before their first forwarding at K = 10. Detection before and by completion (the plan's definition) and the weekly ranking are still reported | Before its first outflow the money is still in the account and can be held | Changes the plan's definition; sent to the advisor 2026-11-09, no reply by 2026-11-12, applied as proposed; switching back needs only re-evaluation of the saved scores |
| 2 | Hard negatives (sources, exits, one-sided hubs) stay negatives | A bank does not know which accounts belong to a case | Applied; results without them reported alongside |
| 3 | Scored population: regular accounts, at least 20 transfers up to day t; all accounts reported alongside | Over all accounts activity alone separates mules (section 4) | Applied |
| 4 | Alert budget: K = 10 new accounts per day with a 30-day cooldown; 5, 20, 50 and the plan's threshold rule also reported | A bank reviews a fixed number of cases per day; 30 days is the longest feature window | Applied |
| 5 | Folds: one fixed time split with three seeds | Allowed by the plan (section 21.4); keeps the full test period | Applied |
| 6 | The TGN with the LightGBM features is the plan's ablation row "TGN + multi-scale behaviour", a step towards EarlyMule-TG, not an outside baseline. The role and role-transition steps must each beat the step before them, so the bar for EarlyMule-TG is this model, not LightGBM or the plain TGN | The plan's ablation table adds multi-scale behaviour to the TGN before the role modules; each added module has to show its own gain | Applied 2026-11-12 |
| 7 | Early stopping stays on validation PR-AUC for every model, although test detection before forwarding does not always follow it (control learning-rate sweep, section 7) | Changing the stopping metric means rerunning every model; to be decided once the outside baselines are in | Open |

## 9. Commands on saved run outputs

Each daily run saves its test scores and per-mule table under `experiments/runs/<run>/<commit>/seed<N>/`, so these views need no retraining:

- `python -m earlymule.evaluation.views [--run lightgbm_daily]`: the daily results without hard negatives, logged as `<run>_no_hard_negatives`.
- `python -m earlymule.evaluation.import_scores --from <folder>`: evaluates a model trained elsewhere (the temporal graph models run on Kaggle) from its `scores_{val,test}.parquet` and `run_meta.json`. Labels are attached here from the local ground truth, the validation labels the model saw are checked against them, and the run is logged with the commit it was trained at. LightGBM and imported runs share one evaluation function (`earlymule.evaluation.daily`); re-evaluating the saved LightGBM scores with it reproduces all 590 logged validation and test values.
- `python -m earlymule.evaluation.compare A B [--k 10]`: the paired comparison of section 6 for every seed both runs have; writes `experiments/runs/comparisons/<A>_vs_<B>.csv` and prints whether A is better than B.
