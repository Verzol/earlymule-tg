# Evaluation protocol v0.3

Version 0.3, 2026-11-10. Fixes how every model in the thesis is scored, so that LightGBM, the graph baselines and EarlyMule-TG are compared under the same rules. The dataset, labels and allowed columns are in `docs/data_spec.md`. Choices that stay within the research plan are decisions, each with its reason; the one change to a definition in the plan, the primary early-detection metric, was sent to the advisor on 2026-11-09 (section 8). Section 9 lists what is not implemented yet.

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
2. Inputs only from the columns allowed in `docs/data_spec.md` section 6 (checked in code by `src/earlymule/data/leakage.py`).
3. Every choice (features, hyperparameters, training population, early stopping) is made on validation. Early stopping tracks validation PR-AUC only, with a patience of 1,000 rounds: on the weekly task with seed 1, validation PR-AUC was 0.742 at round 3, fell to 0.53 by round 50 and only reached its best, 0.797, at round 1,416.
4. Every run is logged in `experiments/experiments.csv` with commit, seed, config and metrics; a run from uncommitted code is marked dirty and not reported.
5. Scores are ranked as log-odds or raw scores, not saturated probabilities, which avoids most ties; remaining ties follow the ranking order above.

**Current reference (LightGBM, daily, test, 3 seeds, K = 10, 30-day cooldown, commit `c62fa61`):** detected before forwarding 0.761 ± 0.003 (seed 0: 0.763, interval [0.697, 0.831]); before completion 0.651 (multi-day mules 0.797); by completion 0.959; PR-AUC 0.650 ± 0.004; alert precision 0.118 (ceiling about 0.12); 30% of alerts on hard negatives; about 1,660 distinct accounts among 1,800 alerts; median TTD 0 days, lead time 2 days; 1 mule alerted in the 7 days before its onset. Before forwarding by typology (seed 0): M3 and M4 0.86, M1 0.78, M2 0.76, cycle 0.42; median lead time M1 and M3 4 days, M4 2, M2 and cycle 0.

Alongside: K = 5, 20 and 50 catch 0.678, 0.821 and 0.842 before forwarding. Over all accounts, K = 10 catches 0.631 ± 0.062 (PR-AUC 0.570). The threshold rule catches 0.497 ± 0.045 with 7.2 alerts per day and alert precision 0.348; at θ, F1 0.626, MCC 0.639, balanced accuracy 0.761. Weekly sanity check: PR-AUC 0.780 ± 0.024 on regular accounts, Recall@100 0.968.

**Effect of the v0.2 rules** (seed 1, K = 10, detected before forwarding / alert precision): v0.1 rules 0.67 / 0.33; finished mules kept in the ranking, no cooldown 0.55 / 0.28; finished mules removed, 30-day cooldown 0.77 / 0.12; v0.2 (both) 0.76 / 0.12. Removing finished mules from the ranking inflated the v0.1 numbers; the cooldown removes repeated alerts, which also keeps finished mules out of the top K, since they were usually alerted during their case.

## 8. Decisions

| # | Decision | Reason | Status |
|---|---|---|---|
| 1 | Main task: daily early detection; primary metric: share of mules alerted before their first forwarding at K = 10. Detection before and by completion (the plan's definition) and the weekly ranking are still reported | Before its first outflow the money is still in the account and can be held | Changes the plan's definition; sent to the advisor 2026-11-09 |
| 2 | Hard negatives (sources, exits, one-sided hubs) stay negatives | A bank does not know which accounts belong to a case | Applied; results without them still to be added (section 9) |
| 3 | Scored population: regular accounts, at least 20 transfers up to day t; all accounts reported alongside | Over all accounts activity alone separates mules (section 4) | Applied |
| 4 | Alert budget: K = 10 new accounts per day with a 30-day cooldown; 5, 20, 50 and the plan's threshold rule also reported | A bank reviews a fixed number of cases per day; 30 days is the longest feature window | Applied |
| 5 | Folds: one fixed time split with three seeds | Allowed by the plan (section 21.4); keeps the full test period | Applied |

## 9. Not yet implemented

- The daily view without hard negatives. It can be computed from the saved test scores of each run (`experiments/runs/<run>/<commit>/seed<N>/scores_test.parquet`) without retraining.
- A command that runs the paired comparison over the saved per-mule tables of two models (`detection_test.parquet` in the same folders); the function exists and is tested.
