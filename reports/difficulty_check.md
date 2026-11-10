# Difficulty and leakage check

Dataset `80K-earlymule-mule`, checked 2026-11-05, rerun 2026-11-10 after the early-stopping fix (commit `c62fa61`, seed 0; early stopping on validation PR-AUC only, patience 1,000): accounts without a label (mules that started in an earlier week) stay in the ranking and can take top-100 slots.
Reproduce: `python -m earlymule.models.difficulty` (about 7 minutes; logs one row per configuration as `difficulty_<name>` in `experiments/experiments.csv`, writes `data/processed/difficulty_table.csv`) and `python -m earlymule.data.shortcut_audit` (writes `data/processed/shortcut_audit.csv`, `shortcut_members.csv`).

## Setup

Same weekly task as the LightGBM walking skeleton: a snapshot every 7 days, positive if the mule's onset falls in the last 7 days, features from transfers up to the snapshot day, split by day 60/15/25, train negatives subsampled to 5%. Test: 25 snapshots, days 546–714.

Configurations:

| Name | What ranks the accounts |
|---|---|
| random | Uniform random score |
| rule_fan_in | Distinct payers in the last 7 days |
| rule_fan_out | Distinct payees in the last 7 days |
| rule_pass_through | min(money received, money sent) in the last 7 days |
| rule_new_peers | Counterparties never seen before the 7-day window (both directions) |
| lgbm_static | LightGBM on `initial_deposit` only |
| lgbm_activity | LightGBM on transfers up to the snapshot day only (the activity shortcut) |
| lgbm_full | LightGBM on the 43 weekly v2 features (same model as `lightgbm_weekly_v2`, stopped at round 1,777) |

Test populations: **all** accounts; **regular** accounts (at least 20 transfers up to the snapshot day); **regular, no other members** (regular without the alert members that are not mules: sources, exits and the one-sided fan_in/fan_out hubs).

## Results (test)

PR-AUC with 95% case-bootstrap interval; Recall@100 = share of mules in the top 100 of their own snapshot.

| Configuration | All: PR-AUC | All: ROC-AUC | Regular: PR-AUC | Regular: ROC-AUC | Regular: Recall@100 | Regular, no other members: PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0001 | 0.50 | 0.0006 | 0.50 | 0.005 | 0.0007 |
| rule_fan_in | 0.0020 | 0.97 | 0.0022 | 0.86 | 0.00 | 0.0026 |
| rule_fan_out | 0.071 | 0.95 | 0.072 [0.043, 0.107] | 0.80 | 0.17 | 0.139 |
| rule_pass_through | 0.0052 | 0.95 | 0.0053 | 0.84 | 0.07 | 0.0067 |
| rule_new_peers | 0.019 | 0.99 | 0.019 [0.014, 0.030] | 0.97 | 0.23 | 0.022 |
| lgbm_static | 0.0003 | 0.62 | 0.0006 | 0.48 | 0.01 | 0.0007 |
| lgbm_activity | 0.0007 | 0.91 | 0.0008 | 0.55 | 0.01 | 0.0009 |
| **lgbm_full** | **0.797** [0.744, 0.840] | 1.00 | **0.800** [0.746, 0.846] | 1.00 | **0.959** | **0.890** [0.853, 0.922] |

Rows: all 2,388,097 (224 mules, rate 0.0001); regular 361,640 (219, 0.0006); regular without other members 298,615 (219, 0.0007).

lgbm_full Recall@100 by typology (regular): M4 collect-disperse 1.00, M2 relay chain 0.98, M3 collect-exit 0.95, M1 fan-in-out 0.94, cycle 0.89. rule_new_peers puts no cycle or relay-chain mule in its top 100.

Share of test rows with at least one new counterparty in the last 7 days: mules 99.1%; regular non-mules 9.9%; all non-mules 1.6%.

## Activity shortcut: hubs or member selection?

Median normal transfers over the whole period (`shortcut_members.csv`):

| Group | Accounts | Median normal transfers | Regular (≥ 20) |
|---|---:|---:|---:|
| Non-members with any normal transfer | 28,029 | 8 | 43% |
| Non-members, regular | 11,971 | 110 | 100% |
| Mules (not main account) | 502 | 111.5 | 98% |
| Mules (main account) | 440 | 113.5 | 99% |
| Sources / destinations | 1,467 / 1,050 | 111 / 110 | 97% / 97% |
| fan_out hub / fan_in hub (main) | 30 / 30 | 141 / 222 | 100% |

The shortcut comes from drawing every alert member from accounts with regular normal activity (AMLSim change 2), not from picking hubs: inside the regular population mules are as active as other accounts. Only the 60 one-sided fan hubs, which are negatives, are more active.

## Conclusions

1. **No label leakage found.** Static input (`initial_deposit`) and activity level rank mules at random on regular accounts (PR-AUC 0.0006 and 0.0008 against a rate of 0.0006). Over all accounts the activity level still lifts ROC-AUC to 0.91, so results over all accounts must always come with the regular population.
2. **No single rule solves the task.** The best rule (distinct payees in 7 days, fan-out) reaches PR-AUC 0.07; LightGBM reaches 0.80. The model combines window features, mainly new counterparties and the 7-day share of 30-day counterparties.
3. **The weekly task is easy, for a simulator reason.** 99% of mules meet a new counterparty in the week they start; only 10% of regular accounts do, because AMLSim normal models pay fixed counterparties. Recall@100 is 0.97, so weekly scoring cannot separate stronger models. The daily early-detection task (76% of mules caught before their first forwarding at 10 new alerts per day, `docs/evaluation_protocol.md`) remains the comparison task.
4. **Sources and exits are the hard negatives.** Removing them raises PR-AUC from 0.80 to 0.89 and doubles the fan-out rule (0.07 to 0.14). They stay as negatives in the main results, because a bank does not know which accounts belong to a case.
5. **What to fix, and at what cost.** (a) Add occasional new counterparties to normal accounts in AMLSim: closes the main realism gap, but needs simulator changes and a new dataset with all checks rerun. (b) Keep the data and evaluate on daily scoring only, stating the limitation: no new work, but absolute numbers stay optimistic. (c) Add legitimate pass-through accounts as negatives: harder and more realistic, but needs a new normal model. Proposal: (b) now, (a) as the first change for the next dataset version, decided with the advisor.

Seeds: one seed. The weekly v2 model varied by ±0.06 PR-AUC across three seeds (`lightgbm_weekly_v2`), so differences under about 0.1 between LightGBM variants are not meaningful; the rule and random rows have no training randomness.
