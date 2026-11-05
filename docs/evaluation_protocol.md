# Evaluation protocol v0.1

Version 0.1, 2026-11-05. Fixes how every model in the thesis is scored, so that LightGBM, the graph baselines and EarlyMule-TG are compared under the same rules. The dataset, labels and allowed columns are in `docs/data_spec.md`. Items marked **proposal** wait for the advisor (section 8). Most rules are already implemented and tested; section 9 lists what is not yet.

## 1. Task

At the end of each day t the model gives every scored account a suspicion score, using only transfers dated on or before t. Accounts are ranked by score and an analyst reviews the top K accounts of the day.

- **Main task: daily early detection (proposal).** Score every day; the question is whether a mule is alerted while its case is still running, ideally before it finishes forwarding.
- **Secondary task: weekly ranking.** Score every 7 days; positive if the mule's onset fell in the last 7 days. Kept as a sanity check only: LightGBM already reaches Recall@100 of 0.98 there, so it cannot separate stronger models (`reports/difficulty_check.md`).

## 2. Data split

| Split | Days | Use |
|---|---|---|
| Train | 0–431 (60%) | Fitting; daily snapshots from day 30, when the 30-day window is full |
| Validation | 432–539 (15%) | Early stopping, hyperparameters, feature and design choices |
| Test | 540–719 (25%) | Final numbers only; never used to choose anything |

One fixed split, three seeds (proposal, instead of three rolling time folds). Features of a day never read later transfers (unit-tested). Weekly label windows stay inside their split.

## 3. Labels at day t

**Daily.** A mule is positive from its onset day to its completion day (first and last illicit transfer), both included. After completion it is left out: the case is over, so it is neither a positive nor a fair negative. A mule before its onset is a negative, like any other account.

**Weekly.** Positive if onset falls in (t − 7, t]; left out if onset is earlier (it was due at an earlier snapshot); negative if onset is later.

## 4. Scored population

**Regular accounts:** at least 20 transfers (any kind) up to day t. Results on all accounts are reported next to them, never alone.

Reason: case members were drawn from active accounts, so over all accounts activity alone separates mules (ROC-AUC 0.84 for a model on transfer counts); on regular accounts it does not (ROC-AUC 0.57, PR-AUC 0.0008 against a mule rate of 0.0006). The model is trained on the same population: on validation this raised daily PR-AUC from 0.26 to 0.55.

**Hard negatives (proposal: keep).** Sources, exits and one-sided fan hubs of laundering cases stay as negatives in the main results, because a bank does not know which accounts are in a case. Results without them are reported as a second view. They take about 40% of daily alerts at K = 10, and removing them raises weekly PR-AUC from 0.79 to 0.89.

## 5. Metrics

Day resolution throughout; within a day the order of transfers is unknown.

| Metric | Definition | Unit |
|---|---|---|
| PR-AUC | Average precision over all scored (account, day) rows of the split, pooled | 0–1; random ranking = mule rate |
| Detected by completion | Share of mules alerted at least once on a day d with onset ≤ d ≤ completion | share of mules |
| Detected early | Same, with d < completion | share of mules |
| Time to detect (TTD) | d − onset for the first alert d; median over mules detected by completion | days |
| Lead time | completion − d; median over the same mules | days |
| Alert precision | Alerts on a mule during its active days ÷ all alerts | share of alerts |
| Alerts on hard negatives | Alerts on non-mule case members ÷ all alerts | share of alerts |
| Recall@K (weekly) | Share of positives in the top K of their own snapshot, K = 100 | share of positives |

**Alert budget.** K = 5, 10, 20 and 50 alerts per day are all reported; **K = 10 is the headline (proposal).** On an average test day about 4.7 mules are active, so K = 10 is about twice that load and caps alert precision near 0.47.

**Which mules count.** Mules whose onset falls in the test period and who are in the scored population on at least one active day (219 in the current test set).

**Same-day mules** (onset = completion; 40 of the 219) can be detected by completion but never early at day resolution. "Detected early" is reported over all mules and over multi-day mules only (179); the multi-day figure is the headline for early detection.

**Cycle initiators** (40 accounts that send first and receive the loop's return) satisfy the two-sided rule and stay positive. Detection by completion is also broken down by typology, so cycles can be read apart from M1–M4.

**Sanity check.** Mules alerted in the 7 days before their onset are counted; a high count would point to leakage.

## 6. Uncertainty and comparisons

- Three seeds (0, 1, 2) per model; report mean ± standard deviation.
- 95% intervals by case bootstrap on test (1,000 resamples): whole laundering cases are resampled, because mules of one case move together; negatives stay fixed.
- A model is called better only if it wins on all three seeds and by more than the seed spread. Current spread: daily PR-AUC ± 0.004, weekly PR-AUC ± 0.06.

## 7. Rules for every model

1. Same split, population, labels and metrics as above; graph models build the graph of day t from transfers on or before t.
2. Inputs only from the columns allowed in `docs/data_spec.md` section 6 (checked in code by `src/earlymule/data/leakage.py`).
3. Every choice (features, hyperparameters, training population, early stopping) is made on validation.
4. Every run is logged in `experiments/experiments.csv` with commit, seed, config and metrics; a run from uncommitted code is marked dirty and not reported.
5. Scores are ranked as log-odds or raw scores, not saturated probabilities, so ties do not decide the top K.

**Current reference (LightGBM, daily, test, 3 seeds, K = 10):** PR-AUC 0.650 ± 0.004; detected by completion 0.953; detected early 0.598 (multi-day mules 0.732); alert precision 0.330; 40% of alerts on hard negatives; median TTD 0 days, lead time 1–2 days; no mule alerted in the 7 days before its onset.

## 8. Decisions for the advisor

| # | Question | Proposal |
|---|---|---|
| 1 | Main task | Daily early detection; weekly ranking as a sanity check only |
| 2 | Hard negatives (sources, exits, one-sided hubs) | Keep as negatives; also report results without them |
| 3 | Scored population | Regular accounts, at least 20 transfers up to day t; all accounts reported alongside |
| 4 | Headline alert budget | K = 10 per day, with 5, 20 and 50 also reported |
| 5 | Folds | One fixed time split with three seeds, instead of three rolling time folds |

## 9. Not yet implemented

- Daily scoring reports only regular accounts; the views over all accounts and without hard negatives exist for the weekly task (`python -m earlymule.models.difficulty`) and still have to be added to the daily run.
- Detection early and TTD by typology (only detection by completion is broken down now).
