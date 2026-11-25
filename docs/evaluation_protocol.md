# Evaluation protocol v0.4

Version 0.4, 2026-11-18, updated 2026-11-23 with the run-to-run variation of the neural models (sections 6, 7, decisions 9 and 10), label scarcity (section 11), and on 2026-11-24 with the rule for choosing variants (decision 11), the choice of the proposed model and the end of decision 7 (decision 12), and the pre-registered confirmation on new simulations (section 12, decision 13) (v0.3: 2026-11-10). Fixes how every model in the thesis is scored, so that LightGBM, the graph baselines and EarlyMule-TG are compared under the same rules. The dataset, labels and allowed columns are in `docs/data_spec.md`. Choices that stay within the research plan are decisions, each with its reason; the one change to a definition in the plan, the primary early-detection metric, was sent to the advisor on 2026-11-09, applied as proposed on 2026-11-12 and approved by the advisor on 2026-11-15 (section 8). Section 9 lists what is not implemented yet.

v0.2 answered two reviews of v0.1: finished mules stay in the ranking, the alert budget counts new accounts with a 30-day cooldown, the primary metric is detection before the first forwarding, model comparisons use a paired case bootstrap, and ties and the start of the alert history follow fixed rules. v0.3 adds the research plan's threshold rule, its secondary metrics and per-typology table, results over all accounts for the daily task, and early stopping on validation PR-AUC only (section 7). All logged runs were redone under these rules (commit `c62fa61`). v0.4 adds a second task, scoring each outgoing transfer just before it executes (section 10), after the advisor's suggestion of 2026-11-15; it is reported next to the primary metric and does not replace it.

## 1. Task

At the end of each day t the model gives every scored account a suspicion score, using only transfers dated on or before t. Accounts are ranked by score and an analyst reviews K new accounts of the day.

- **Main task: daily early detection.** Score every day; the question is whether a mule is alerted before it forwards the illicit money.
- **Second task: scoring at payment initiation.** Every outgoing transfer is scored just before it executes and the riskiest are held; the question is whether a mule's transfer is held at or before its first illicit outflow (section 10).
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

Day resolution throughout: an alert on day d uses transfers up to the end of day d. Timestamps carry no time of day; the execution order within a day is known from the simulator's balance log (data spec, limitation 5), but end-of-day scoring does not use it.

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

**Why "before forwarding" is primary.** A mule caught before its first illicit outflow can still be stopped before the money leaves; one caught later, but before completion, may already have passed most of it on. The metric needs at least one day between onset and first outflow; mules that receive and forward on the same day, and the 40 cycle initiators (they send first and receive the loop's return later, so onset is their first outflow), cannot be caught before forwarding by end-of-day scoring (scoring a transfer before it is executed, planned as a second task, can reach the 182 that receive first). They still count in the other detection metrics; detection before forwarding and by completion are also broken down by typology. A mule can also miss the chance because it only joins the ranked population (20 transfers) after its first outflow; each run reports this count so that a population limit is not read as a model error. In the current test set all 173 are rankable before their first outflow.

**Alert budget.** K = 5, 10, 20 and 50 new accounts per day are all reported; **K = 10 is the headline**, as an assumed daily review capacity. About 1.2 mules start per test day (219 in 180 days), and a case lasts at most 19 days, so under the cooldown each mule can be hit by at most one alert: alert precision is capped near 222 / 1,800 = 0.12 at K = 10. Precision is therefore reported but not used to compare models.

**Research plan's threshold rule.** The plan detects a mule on the first day its score reaches a threshold θ chosen on validation (plan section 24). θ is the score with the best F1 over labelled validation rows: the plan names no criterion, and F1 weighs missed mules and false alerts equally without a cost ratio. Every account-day at or above θ is an alert, with no daily budget and no cooldown. The same detection metrics are reported for it, plus alerts per day and the plan's secondary metrics (section 23.2) over labelled test rows: precision, recall, F1, F1-macro, MCC and balanced accuracy.

**Why TTD does not rank models here.** At day resolution the median TTD is 0 days for every typology: most detected mules are alerted on their onset day. The plan's KPI of a 10% lower median TTD therefore cannot be met by any model; detection before forwarding and lead time carry the timing comparison, and TTD is still reported.

**Sanity check.** Mules alerted in the 7 days before their onset are counted; a high count would point to leakage.

## 6. Uncertainty and comparisons

- Three seeds (0, 1, 2) per model; report mean ± standard deviation.
- 95% intervals by case bootstrap on test (1,000 resamples). Scores and alerts are computed once and kept fixed; only the mules are resampled, whole laundering cases at a time, because mules of one case move together. Negatives, including the non-mule members of a case, are not resampled.
- **Comparing two models (A against B).** On the primary metric at K = 10, for each seed s compare A's run with seed s to B's run with seed s: draw 1,000 resamples of the cases, compute both models' shares on the same draw, and take the 2.5th and 97.5th percentiles of A − B (`paired_bootstrap_difference` in `src/earlymule/evaluation/uncertainty.py`, on the per-mule tables that each daily run saves). A is called better than B only if this interval lies above 0 for all three seeds. Other metrics are reported, not used for the verdict.
- **What the interval leaves out.** The case bootstrap keeps each trained model fixed, so it does not cover training randomness. For LightGBM the 1% sample of training negatives alone moves the primary metric by 3 to 4 points: the same features with another sample give 80.0–82.1% against 81.5–84.4% (`reports/feature_analysis.md`, 2026-11-16). Requiring the interval above 0 in all three seeds keeps a single lucky sample from deciding a comparison. The feature ablation avoids this source altogether by giving every configuration of a seed the same negative sample.
- **Neural models are not reproducible by seed.** Two runs of the role-transition step with the same code and seed (`7bfcfb4` and `fad3097`, which only adds a config with a higher epoch cap) already differ in validation PR-AUC after epoch 1 (seed 0: 0.448 against 0.455), since GPU training is not deterministic. Their test results differ by up to 5.2 points in the same seed (section 7). Repeated runs of one configuration are therefore all reported and pooled, and a difference between two neural runs smaller than this spread is described as within training noise (decision 9).

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
- Training length: the transition step reached its best epoch at 27 of at most 30 in seed 0, so the epoch cap might bind there; the role state and full model stopped early in every seed. The follow-up below shows the cap does not bind.

**Follow-up runs of the role-transition step (2026-11-23).** Two variants, first compared on mean validation PR-AUC (see decision 11 for the rule used since 2026-11-24):
- Epoch cap 50 (`tgn_transition_e50`, `fad3097`): 0.804 / 0.832 / 0.804 before forwarding, PR-AUC test 0.568, validation 0.510 (original 0.524), best epochs 5 / 23 / 5. Against the original: −4.6 [−8.8, −0.6], −0.6 [−4.4, +3.1], −5.2 [−9.9, −1.1]; against LightGBM −1.2, −0.6, −4.1, all intervals across 0. Only the cap differs, so this run is a repeat of the original with different GPU randomness (section 6), not a test of the cap. Pooled over the six runs the step detects 0.831 ± 0.022, level with LightGBM (0.832): the 0.848 of the first three runs is partly luck.
- Temporal transition r(t) − r(t − 7) (`tgn_transition_t7`, `beb2a5d`): 0.821 / 0.844 / 0.803 (0.823 ± 0.020), PR-AUC test 0.642 / 0.621 / 0.610, validation 0.548 / 0.511 / 0.516 (mean 0.525, original 0.524). Against the original −2.9 [−6.6, +0.6], +0.6 [−3.9, +4.6], −5.2 [−9.9, −1.1]; against LightGBM +0.6, +0.6, −4.1, all intervals across 0. Seed 2 first shared a GPU with seed 0 and produced no output; it was rerun alone (kernel `earlymule-tgn-transition-t7-seed2`, same commit). Validation PR-AUC is level with the original, well within the spread between repeats (decision 9), so the original step stays: it is simpler and needs less GPU memory (peak 3.9 against 7.0 GiB per seed), which lets three seeds share one Kaggle session.

**Class imbalance of the account head (2026-11-23).** The original step trains on every mule row and 1% of the other rows with plain BCE. Two losses on the same sample (`account_loss` in `tgn.py`, commit `f366a59`):
- Class-weighted BCE (`tgn_transition_wbce`), positive weight 25.5 = kept negatives per positive: 0.786 / 0.803 / 0.803, PR-AUC test 0.574, validation 0.455 (original 0.524), best epochs 4 / 9 / 7. Against the original −6.4 [−10.8, −1.8], −3.5 [−8.1, +0.6], −5.2 [−8.6, −2.1]: worse in two seeds and lower on validation, so it is not used.
- Focal loss (`tgn_transition_focal`, γ = 2, α = 0.25): 0.815 / 0.775 / 0.815, PR-AUC test 0.610 / 0.569 / 0.585, validation 0.562 / 0.527 / 0.569 (mean 0.553, the highest of all variants), best epochs 22 / 7 / 23. Against the original −3.5 [−7.6, +0.6], −6.4 [−10.6, −2.2], −4.1 [−8.9, +0.6]; against LightGBM 0.0, −6.4 [−10.2, −2.8], −2.9.
- Validation PR-AUC does not follow the primary metric across these variants. Detection before forwarding on validation (K = 10, mean of 3 seeds): original 0.836, epoch cap 50 0.809, temporal transition 0.818, class-weighted 0.797, focal 0.791. Focal is first on validation PR-AUC and last on the primary metric, on validation and on test alike. Variants are therefore chosen on the primary metric on validation from now on (decision 11). Under that rule the original step stays and neither loss is used: both are lower than the original by more than the spread between its repeats (0.809 to 0.836), and class-weighted BCE is also lower on validation PR-AUC.

**Effect of the v0.2 rules** (seed 1, K = 10, detected before forwarding / alert precision): v0.1 rules 0.67 / 0.33; finished mules kept in the ranking, no cooldown 0.55 / 0.28; finished mules removed, 30-day cooldown 0.77 / 0.12; v0.2 (both) 0.76 / 0.12. Removing finished mules from the ranking inflated the v0.1 numbers; the cooldown removes repeated alerts, which also keeps finished mules out of the top K, since they were usually alerted during their case.

## 8. Decisions

| # | Decision | Reason | Status |
|---|---|---|---|
| 1 | Main task: daily early detection; primary metric: share of mules alerted before their first forwarding at K = 10. Detection before and by completion (the plan's definition) and the weekly ranking are still reported | Before its first outflow the money is still in the account and can be held | Changes the plan's definition; sent to the advisor 2026-11-09, applied 2026-11-12, approved by the advisor 2026-11-15, who also asked for a feasibility check from the bank's side and for scoring just before the first outgoing transfer is executed (open) |
| 2 | Hard negatives (sources, exits, one-sided hubs) stay negatives | A bank does not know which accounts belong to a case | Applied; results without them reported alongside |
| 3 | Scored population: regular accounts, at least 20 transfers up to day t; all accounts reported alongside | Over all accounts activity alone separates mules (section 4) | Applied |
| 4 | Alert budget: K = 10 new accounts per day with a 30-day cooldown; 5, 20, 50 and the plan's threshold rule also reported | A bank reviews a fixed number of cases per day; 30 days is the longest feature window | Applied |
| 5 | Folds: one fixed time split with three seeds | Allowed by the plan (section 21.4); keeps the full test period | Applied |
| 6 | The TGN with the LightGBM features is the plan's ablation row "TGN + multi-scale behaviour", a step towards EarlyMule-TG, not an outside baseline. The role and role-transition steps must each beat the step before them, so the bar for EarlyMule-TG is this model, not LightGBM or the plain TGN | The plan's ablation table adds multi-scale behaviour to the TGN before the role modules; each added module has to show its own gain | Applied 2026-11-12 |
| 7 | Early stopping stays on validation PR-AUC for every model, although test detection before forwarding does not always follow it (control learning-rate sweep, section 7) | Changing the stopping metric means rerunning every model. Closed before the confirmation runs (section 12): PR-AUC uses every labelled validation row and moves smoothly between epochs, whereas detection before forwarding counts about 170 validation mules and jumps by whole mules; the choice between variants already uses the primary metric (decision 11) | Closed 2026-11-24: PR-AUC kept for every model and every dataset |
| 8 | Second task at payment initiation, reported next to the primary metric, not instead of it. Label: illicit outflow of a mule; budget: H new accounts held per day with a 30-day cooldown and a fixed threshold chosen on validation | Advisor's suggestion (2026-11-15) to score when the bank is about to execute the first transfer. A fixed threshold, because a hold must be decided when the order arrives; accounts as the budget unit, so H = K compares the same review load | Applied (section 10) |
| 9 | Repeated runs of one neural configuration are all reported and pooled; the verdict rule of section 6 is unchanged, and a difference smaller than the spread between repeats of the same seed is described as within training noise | GPU training is not reproducible by seed: two runs of the same role-transition configuration differ by up to 5.2 points in one seed (section 7, follow-up runs) | Applied 2026-11-23 |
| 10 | Label scarcity keeps the labels of whole training cases (a seeded shuffle of cases, nested across fractions); the other training mules count as negatives. Leaving them out of the loss is reported as a sensitivity run | A bank learns of mules case by case, through investigations. It cannot remove mules it does not know of from its negatives, so leaving them out would use the ground truth | Applied 2026-11-23 (section 11) |
| 11 | Variants of a model (configuration, loss, learning rate) are chosen on the primary metric on validation, mean of 3 seeds. A difference within the spread between repeats of one configuration counts as a tie, broken by mean validation PR-AUC and then by the simpler variant. Early stopping inside a run stays on validation PR-AUC (decision 7) | Validation PR-AUC ranked the role-transition variants against the primary metric: focal loss was first on it and last on detection before forwarding, on validation and on test (section 7). Test is still never used for a choice. Changed on 2026-11-24, after the focal result; the choices made before it (original role-transition step, no class weighting, control learning rate 0.001) were checked under the new rule and all stand (control: 0.001 and 0.003 tie at 0.709 on validation, 0.001 is higher on PR-AUC) | Applied 2026-11-24 |
| 12 | The proposed model is the role-transition step (A5, `configs/tgn_transition.yaml`), without the gated fusion of A6. STA-GT and the plain TGN are kept as baselines; the gated fusion and the role-state step are reported on the main dataset only | Validation, detection before forwarding at K = 10, mean of 3 seeds: role transition 0.836 (0.823 pooled with its repeat), TGN with features 0.824, role state 0.821, gated fusion 0.773, GraphSAGE 0.776 (72 features), LightGBM 0.779, STA-GT 0.730. The fusion is lower than the transition step by more than the spread between its repeats (0.809 to 0.836), so it is dropped. The transition step and the TGN with features tie on the primary metric; the tie goes to the transition step on validation PR-AUC (0.517 pooled against 0.510), and whether the transition adds anything is the confirmatory comparison 2 of section 12 | Applied 2026-11-24 |
| 13 | Confirmation on new simulations, pre-registered before they are generated (section 12). The main dataset becomes development data: its test period has served about 30 comparisons, a learning-rate sweep read on test and the change of selection rule in decision 11 | A conclusion that rests on one simulated draw, whose test set already informed choices, cannot be called confirmed | Pre-registered 2026-11-24 |

## 9. Commands on saved run outputs

Each daily run saves its test scores and per-mule table under `experiments/runs/<run>/<commit>/seed<N>/`, so these views need no retraining:

- `python -m earlymule.evaluation.views [--run lightgbm_daily]`: the daily results without hard negatives, logged as `<run>_no_hard_negatives`.
- `python -m earlymule.evaluation.import_scores --from <folder>`: evaluates a model trained elsewhere (the temporal graph models run on Kaggle) from its `scores_{val,test}.parquet` and `run_meta.json`. Labels are attached here from the local ground truth, the validation labels the model saw are checked against them, and the run is logged with the commit it was trained at. LightGBM and imported runs share one evaluation function (`earlymule.evaluation.daily`); re-evaluating the saved LightGBM scores with it reproduces all 590 logged validation and test values.
- `python -m earlymule.evaluation.compare A B [--k 10]`: the paired comparison of section 6 for every seed both runs have; writes `experiments/runs/comparisons/<A>_vs_<B>.csv` and prints whether A is better than B.

## 10. Second task: scoring at payment initiation

Added 2026-11-18 after the advisor's suggestion of 2026-11-15 to score when the bank is about to execute the first transfer, from the payee, the amount and what the bank knows about the payee. Why a bank can act at that moment, and what it knows about a payee at another bank: `reports/bank_feasibility.md`. This task is reported next to the primary metric of section 5 and does not replace it.

**Unit and timing.** Every outgoing transfer x of a sender with at least 20 transfers by the end of the previous day (the scored population of section 4), on days 31–719. x is scored from transfers strictly before it in execution order (day, then `seq`, the `tx_log.csv` row, `docs/data_spec.md` section 6) plus its own payee and amount; the sender's balance just before x is its `tx_log.csv` old balance. A conservative variant cuts at the start of x's day instead, so it needs no within-day order. 1,551,375 transfers; about 2,480 a day on validation and test.

**Label.** x is positive when it is an illicit transfer sent by a mule (675 train, 220 validation, 331 test positives). Illicit transfers of sources and exits are negatives, as those accounts are in section 3. A variant labels every illicit transfer positive. Rejected: every transfer of a mule from onset to its first outflow (legitimate transfers would carry the label, so the transfer's own values become noise), and the first illicit outflow alone (later illicit outflows, identical in kind, would be negatives).

**Hold rule.** x is held when its score reaches a fixed threshold θ and its sender was not held in the previous 30 days; a held account has been reviewed and is not held again, as with the alert cooldown of section 5. θ is chosen on validation so that on average H new accounts are held per day (H = 10, 20, 50; 10 is the headline, the same review load as K = 10) and is applied unchanged to test, where the realised number of holds per day is reported. The hold history starts empty on the first day of each split. Rejected: ranking a whole day's transfers (a hold must be decided when the order arrives), counting held transfers instead of accounts (not comparable with K), a hard daily cap (late-day mules always pass, so the result depends on the order of the day rather than the model), and a threshold set by the false-hold rate (about 2.5 holds a day; kept as a reported metric).

**Metrics.** Primary: share of mules with a transfer held at or before their first illicit outflow, on or after onset, over the mules whose first outflow is a scored test transfer (220), and over the 174 of them with a day between onset and first outflow (the population of the primary metric; the end-of-day alerts reach 173 of them). Alongside: the 46 mules that forward on their onset day (end-of-day alerts cannot precede their first outflow), legitimate transfers held per 1,000, the share of held transfers that are illicit outflows of mules, the share of first illicit outflows held themselves and of their amount, PR-AUC over transfers. A hold before onset counts as a miss and starts a cooldown, as an alert before onset does.

**Features** (`src/earlymule/features/transfer_time.py`, `docs/data_spec.md` section 7): the sender's 72 point-in-time features at the end of the previous day, the sender's activity earlier the same day, the order against the sender's history (amount, ratios to recent inflow, balance, mean and largest earlier outflow, days and transfers since the last inflow), the pair (earlier payments to and from the payee), the payee's 72 point-in-time features at the end of the previous day and its activity earlier the same day, the payee's daily LightGBM score of the previous day, and the share of the sender's 7-day inflow from accounts in the top 1% of the previous day's scores. On train days the daily score is out of fold: five LightGBM models (73 features, seed 0, 1% of negatives, early stopping on validation PR-AUC), each trained without one fifth of the accounts, all members of a case in the same fifth, score that fifth (`src/earlymule/models/oof_scores.py`, Kaggle CPU, commit `8ec6f34`; their PR-AUC on the train labels is 0.49 against 0.60 on validation, so the score does not carry the case's own label). Validation and test days use the full-train model `lgbm_abl_full` seed 0.

**Model and variants** (`src/earlymule/models/transfer_lgbm.py`, `configs/transfer_scoring.yaml`): LightGBM with the daily settings, every negative kept (no sampling), early stopping on validation PR-AUC, 3 seeds, trained on Kaggle CPU from the export of commit `cee23a2` at commits `d6128f0` and `5e00cd3` (13 to 31 minutes a run). Variants add the feature groups in turn (sender; + order; + pair; + payee; + propagation = full) and change one thing of the full set: payee at another bank (the payee's outflow side, balance-driven timing and risk score removed; its inflow side stays, so still a best case), no shortcut columns, no new counterparties, no shortcuts at all, conservative cut, wide label.

**Results** (test, H = 10, mean of 3 seeds; `reports/figures/transfer/summary.csv`):

| Variant | Features | Caught, mules with a day before forwarding (174) | All mules (220) | Same-day mules (46) | PR-AUC over transfers | Legitimate held per 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| Sender only | 78 | 0.942 | 0.942 | 0.942 | 0.872 | 2.77 |
| + order | 86 | 0.954 | 0.958 | 0.971 | 0.940 | 3.29 |
| + pair | 90 | 0.979 | 0.979 | 0.978 | 0.981 | 2.58 |
| + payee | 168 | 0.988 | 0.986 | 0.978 | 0.983 | 2.47 |
| **Full** (+ propagation) | 169 | **0.996** | 0.992 | 0.978 | 0.985 | 2.48 |
| Payee at another bank | 119 | 0.994 | 0.991 | 0.978 | 0.987 | 2.63 |
| No new counterparties | 140 | 0.988 | 0.988 | 0.986 | 0.969 | 3.01 |
| No shortcuts at all | 114 | 0.985 | 0.983 | 0.978 | 0.954 | 3.19 |
| Conservative cut | 169 | 0.983 | 0.970 | 0.920 | 0.954 | 2.33 |
| Wide label | 169 | 0.981 | 0.980 | 0.978 | 0.442 | 2.38 |

Full set, H = 10: 8.25 new accounts held a day on test (threshold from validation), 12.7% of held transfers are illicit outflows of mules and 26% are illicit at all; 85% of first illicit outflows are held themselves, 99.2% of their amount is stopped. H = 20 and 50 catch 0.985 and 0.933: a lower threshold holds more mule accounts before their onset, and the cooldown then skips them, the same effect as for end-of-day alerts. The variant "no shortcut columns" (4 columns dropped, 0.998) tests nothing, since `pair_n_before = 0` repeats `first_to_payee`; it is kept in the log and superseded by the two stricter variants.

**Paired comparisons** (`reports/figures/transfer/comparisons.csv`, case bootstrap, seeds 0 / 1 / 2, points):
- Holds against end-of-day alerts of the daily LightGBM (72 features, `7bfcfb4`) on the 173 mules both cover: H = 10 vs K = 10 +17.9 [+11.7, +23.8], +16.2 [+9.9, +21.5], +15.0 [+9.9, +20.5]; H = 20 vs K = 20 +11.6, +13.3, +12.1 (all intervals above 0).
- Holds and end-of-day alerts together (H = K = 10) vs end-of-day K = 20, the same review load: +12.7 [+8.1, +17.6], +15.0 [+9.2, +20.7], +12.7 [+8.0, +17.6].
- Against the full set: sender only −5.2, −5.8, −5.2 (below 0 in all seeds); + order −5.8, −4.0, −2.9; + pair −1.2, −2.3, −1.7; the other variants within 2.3 points.

**Reading.** On this simulator, scoring at payment initiation catches almost every mule before its money moves on, about 15 to 18 points more than end-of-day alerts with the same review load, and it reaches the same-day mules that end-of-day alerts cannot. The level is an upper bound of the simulator, not an estimate for real data: with every shortcut column removed the full set still catches 98.5%, and the strongest remaining features are timing (days since the last inflow and outflow, 7-day against 30-day activity of sender and payee). Normal accounts transact on fixed schedules with a few fixed partners (`docs/data_spec.md` section 9, item 3), so any departure gives a mule away, and a transfer that happens off schedule is itself a signal. The second task therefore needs the noisier normal behaviour planned in `docs/data_spec.md` section 10, item 5, before its level can be read as a result. Results with the payee seen only from outside barely change here, because every sender in the simulator is a customer of the one bank.

**Commands.** `python -m earlymule.features.transfer_time [--conservative]` (about 5 minutes, 2.6 GB); `python -m earlymule.models.oof_scores folds` then `train` on Kaggle (`notebooks/kaggle/oof_scores/`); `python -m earlymule.models.transfer_lgbm export`, then `train` per (variant, seed) on Kaggle (`notebooks/kaggle/transfer_scoring/make_batches.py`), then `import --from <downloaded folder>`; `python -m earlymule.analysis.transfer_summary --runs <folders>`.

## 11. Label scarcity

Research plan section 19 (decision 10). Only the training mules of a share of the 221 training cases keep their label (449 training mules on `80K-earlymule-mule`); cases are shuffled once per seed and the first max(1, round(fraction × 221)) are kept, so 1% is 2 cases, 5% is 11, 10% is 22 and 25% is 55, and each set contains the smaller ones. The other training mules count as negatives; validation and test labels, the graph and the time split are unchanged. Code: `earlymule.data.label_fraction`, `--label-fraction` of `lightgbm_export` and `tgn` (commit `4b362f3`); runs on Kaggle CPU (`docs/setup.md` section 4.7).

**Daily LightGBM, full feature set** (test, K = 10, mean ± sd over seeds 0 / 1 / 2). The 100% row is `lgbm_abl_full` (`956b8a1`), which draws the same negative sample per seed, so each comparison isolates the labels.

| Labelled cases | Labelled mules (seeds 0 / 1 / 2) | Before forwarding | PR-AUC | Against 100%, points [95% interval] |
|---|---|---|---|---|
| 1% | 3 / 8 / 5 | 0.541 ± 0.194 | 0.292 | −15.6 [−22.3, −8.9], −48.0 [−56.4, −39.5], −16.8 [−22.8, −10.5] |
| 5% | 13 / 17 / 23 | 0.742 ± 0.062 | 0.540 | −4.0 [−8.1, 0.0], −12.7 [−18.4, −7.0], −3.5 [−8.1, +0.7] |
| 10% | 44 / 32 / 60 | 0.744 ± 0.066 | 0.558 | −5.8 [−10.8, −0.6], −12.7 [−18.1, −6.9], −1.2 [−5.0, +2.7] |
| 25% | 117 / 79 / 125 | 0.794 ± 0.019 | 0.633 | −4.1 [−8.7, +0.6], −1.2 [−5.8, +3.5], +0.6 [−2.3, +3.5] |
| 100% | 449 | 0.809 ± 0.012 | 0.687 | |

- With a quarter of the cases labelled, LightGBM is within the run-to-run spread of the full-label model; the loss grows below 10% and is large at 1%. Seed 1 is the weakest from 1% to 10%; at 1% it has the most labelled mules (8), but 7 of them come from one cycle case (seeds 0 and 2: relay chain and fan-in/fan-out cases), so which cases are labelled matters more than how many mules they hold.
- Leaving the unlabelled mules out of training instead (sensitivity run, `_drop`) changes little: 0.593, 0.738, 0.755, 0.792 at 1 / 5 / 10 / 25%. Against counting them as negatives, the only clear gap is 1% seed 1 (+15.0 [+7.9, +22.6]); at 5% and above no interval excludes 0.

**Graph models** (test, K = 10, mean and range over seeds 0 / 1 / 2; same labelled cases per seed as LightGBM; Kaggle T4, commit `cf346a3`, batches `verzol/earlymule-tgn-lf-1` to `-5`). The 100% column is the full-label run of each model; LightGBM at 25% is the only 25% row so far.

| Model | 1% | 5% | 10% | 100% |
|---|---|---|---|---|
| LightGBM, full feature set | 0.541 (0.318–0.665) | 0.742 (0.671–0.780) | 0.744 (0.671–0.798) | 0.809 |
| TGN with features | 0.514 (0.376–0.636) | 0.715 (0.676–0.734) | 0.721 (0.699–0.740) | 0.825 |
| Role-transition step | 0.349 (0.156–0.613) | 0.692 (0.642–0.740) | 0.715 (0.647–0.751) | 0.848 (0.831 pooled over 6 runs, decision 9) |
| STA-GT-inspired | 0.480 (0.243–0.659) | 0.724 (0.659–0.757) | 0.713 (0.688–0.740) | 0.761 |

- No graph model is shown better than LightGBM at the same fraction in any seed (paired case bootstrap, `compare`); most intervals contain 0, and the largest gaps go against the graph models (role-transition step at 1%: −19.2 points on average).
- The models that lead with full labels lose the most: from 100% to 5% the role-transition step drops 15.6 points, TGN with features 11.0, LightGBM 6.7 and STA-GT 3.7. At 5% and 10% all four lie within about 3 points of each other, so the gain of the temporal models on the main dataset needs most of the training cases.
- At 1% the spread between seeds (up to 46 points) is larger than any gap between models. Seed 1 (the cycle-case draw above) is the weakest for LightGBM, TGN with features and STA-GT; the role-transition step is weakest on seed 0 (0.156, against 0.278 on seed 1).
- STA-GT seed 0 ran out of GPU memory when two STA-GT runs shared one T4 and was rerun alone (batch 5); the notebook now gives STA-GT one run per GPU.
- Still to run on the main dataset: 25% for the graph models and the `_drop` sensitivity at 5% and 25%.

## 12. Confirmation on new simulations (pre-registered 2026-11-24)

Written before any of these datasets is generated, at the commit tagged `prereg-2026-11-24`. Nothing below changes after the first model is trained on them; a deviation is reported as such, with its reason.

**Data.**
- **Confirmation draws G1, G2, G3:** `80K-earlymule-mule` regenerated with the same AMLSim configuration and three new random seeds; ground truth, export, leakage checks and dataset statistics as for the main dataset. LightGBM also runs on two more draws, G4 and G5, to estimate the variation between draws.
- **Harder data, two levels (moderate, hard):** normal accounts with irregular timing and new partners, legitimate accounts that pass most of what they receive on, mules that forward a variable share and amount, and mules paid by partners they already knew (`docs/data_spec.md` decision 5). The generator parameters of both levels are written into the data spec before either level is generated, and every level is reported.
- **IBM AML HI-Small:** only if the feasibility check of 2026-12-06 says so. Its windows, cooldown and split are then fixed in a note before any model is trained on it.

**Models.** Frozen at the tag, with no tuning on the new data:
- LightGBM with 72 features (clarified 2026-11-25, see below).
- GraphSAGE.
- TGN with features.
- STA-GT-inspired.
- The role-transition step (decision 12).
- DyGFormer with the 72 features, once integrated (main dataset and G1–G3 only).
- Two reference rows: random ranking, and a rule after the Vietnamese indicator of many incoming transfers followed by immediate forwarding (Circular 17/2024/TT-NHNN, Appendix 02). The rule's exact form is fixed in `reports/bank_feasibility.md` before it is scored.
- Three seeds per model and dataset; label scarcity runs at 5% of cases for the role-transition step and LightGBM.

**Confirmatory family.** Each research question gets one comparison, all on detection before forwarding at K = 10:
1. TGN with features against LightGBM: does a temporal graph add to the tabular model?
2. Role-transition step against TGN with features: does the transition representation add?
3. Role-transition step against LightGBM at 5% of labelled cases.
4. Role-transition step against LightGBM on mules paid by a partner they already knew before their onset (`known_upstream` in `earlymule.analysis.error_analysis`). Found on the main test set with 23 mules, so it is a hypothesis here, not a result.
5. As comparison 1, on IBM AML HI-Small, if that dataset is run.

**Statistic.**
1. For each draw, each mule's outcome is averaged over the three seeds of each model; the difference of the two models is taken per mule.
2. Hierarchical case bootstrap: 10,000 resamples. Within each draw, laundering cases are resampled; the draws are pooled with weights equal to their numbers of counted mules.
3. Two-sided p-value: twice the smaller tail share of the bootstrap distribution around 0. Holm's correction across the family.
4. **Better:** the Holm-adjusted p-value is below 0.05 with a positive pooled difference, and the difference is positive in at least two of the three confirmation draws.
5. **Not shown:** otherwise, with the pooled difference and its 95% interval still reported.
6. The per-seed rule of section 6 is reported for each draw alongside.
7. Comparisons outside the family (harder data, unseen typologies, other alert budgets, other models) are descriptive: intervals without correction, no verdict.

**LightGBM in the family (clarification, 2026-11-25).** Written while the first LightGBM runs on G1 were still running and before any result on a new draw was read. The main dataset has two LightGBM runs with the same 72 features and parameters that differ only in how the 1% of training negatives is drawn: `lightgbm_daily` (`earlymule.models.daily_scoring`, local, draws per day while building the splits; 0.832) and `lgbm_abl_full` (`earlymule.models.lightgbm_export`, Kaggle CPU, one draw over all training rows of the export; 0.809). Paired on the main test set, the local run leads by −0.6 [−3.3, +1.7], +4.1 [+1.6, +7.0] and +3.5 [+1.1, +6.8] points on seeds 0 / 1 / 2.
- The family uses the export run (`lightgbm_export --ablation full`, run names `lgbm_abl_full_<draw>`): the same code runs on every draw and needs no GPU quota.
- Because the local run may be the stronger baseline, it also runs on G1–G3 (`daily_scoring --dataset <draw>`, run names `lightgbm_daily_<draw>`, three seeds), and comparisons 1 and 4 (full labels) are repeated against it as a sensitivity row; comparison 3 (5% of cases) has only the export run. If the verdict differs between the two LightGBM runs, both are reported and the thesis states that the conclusion depends on the baseline's negative sample.

**Unseen typologies** (descriptive). For each of the five typologies, its training mules are removed from the training rows, not relabelled as negatives, and the model is scored on that typology's test mules. LightGBM runs all five, the role-transition step two (collector to a single exit, cycle), three seeds each, on the main dataset. Expected: little effect, because the share of inflow from new partners leads every typology (AUC 0.94 to 0.98) and each typology has few test mules.

