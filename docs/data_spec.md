# Data Spec v0.2: AMLSim-Mule dataset `80K-earlymule-mule`

Version 0.2, 2026-11-09 (v0.1: 2026-11-04). Describes the dataset used for every experiment from 2026-11-03 on, how mule labels are derived, which columns a model may use, and the known limitations. It is the reference for the weekly reports. Every number can be reproduced with the commands in section 11.

## 1. Summary

| Item | Value |
|---|---|
| Simulator | IBM AMLSim, EarlyMule-TG fork, commit `926fa7f` (`AMLSim/EARLYMULE-TG.md`) |
| Configuration | `AMLSim/conf_80K-earlymule-mule.json`, random seed 0 |
| Period | 720 days, day 0 = 2017-01-01, last day 719 = 2018-12-21 |
| Time resolution | One day; every transfer is stamped 00:00 |
| Accounts | 96,344 (one bank); 31,560 take part in at least one transfer (sent or received), 15,416 in at least 20 |
| Transfers | 1,789,235, of which 3,059 illicit |
| Laundering cases (alerts) | 500 |
| Mule candidates (positive labels) | 942 accounts, 0.98% of all accounts |

## 2. How the data is generated

**Accounts and normal behaviour.** Six blocks of about 16,000 accounts with an initial balance drawn from 50,000–100,000 (before change 5 below). 24,000 normal models (4,000 each of single, fan-in, fan-out, forward, mutual, periodical) make the normal transfers; 99.98% of normal amounts lie between 100 and 1,000 (smaller ones occur when a balance runs low).

**Changes to AMLSim.** The original simulator could not produce the data the thesis needs. Each change is documented with before and after numbers in `AMLSim/EARLYMULE-TG.md`.

| # | Problem in the original | Change |
|---|---|---|
| 1 | Fan-in/fan-out cases stretched over most of the simulation (median 363 days) | Keep the case window planned by the generator |
| 2 | Half of the case members never made a normal transfer, so "no activity" revealed the label and mules had no normal phase | Draw case members from accounts with regular normal activity |
| 3 | Cycle members sent money before receiving it | Ordered schedule for cycles (parameter only) |
| 4 | No typology in which an intermediary both receives and forwards illicit money (fan-in and fan-out hubs are one-sided) | New layered mule typologies M1–M4 (section 3) |
| 5 | Accounts that send on two or more normal models ran out of money mid-simulation (62–99% of them), so a mule placed there could not forward | Multiply the initial balance of every such account by (number of models + 1) |

For change 5 a first approach, excluding those accounts from cases, was tried and rejected: mules then never came from busy accounts, and outgoing activity alone separated them (ROC-AUC 0.67 among regular accounts, against 0.53 before).

## 3. Laundering cases

Layered typologies move money from sources through one or more mule layers to exits. A mule forwards only after all its planned inflows have arrived, passes on the illicit amount it received minus a 10% margin, waits 0–2 days per transfer, and never sends more than its balance.

| Typology (code name) | Research plan | Cases | Accounts per case | Structure |
|---|---|---:|---|---|
| `mule_fan_in_out` | M1 fan-in, rapid fan-out | 150 | 5–10 | sources → 1 mule → exits |
| `mule_relay_chain` | M2 relay chain | 50 | 4–6 | 1 source → 2–4 mules in a chain → 1 exit |
| `mule_collect_exit` | M3 collector, single exit | 150 | 4–8 | sources → 1 collector mule → 1 exit |
| `mule_collect_disperse` | M4 collector, relays, dispersal | 50 | 7–12 | sources → collector → 2–3 relay mules → one exit each |
| `cycle` | M5 (optional) ring | 40 | 5–10 | closed loop, 5–20 days |
| `fan_in` | not a mule pattern | 30 | 5–10 | sources → one hub, 5–20 days |
| `fan_out` | not a mule pattern | 30 | 5–10 | one hub → destinations, 5–20 days |

Source transfers are 100–200 each and happen within 3–10 days (1–3 days for M2). M1 and M3 have one mule per case and M2 and M4 about three, so case counts were chosen to give each layered typology about 150 mules. `fan_in` and `fan_out` stay in the data as hard negatives: accounts involved in a laundering case that are not mules.

## 4. Mule label

**Definition (advisor, 2026-11-02).** A mule is an account that both receives and forwards money within a laundering case. Accounts that only send at the start of a case or only receive at its end are not mules.

**Rule.** An account is a mule candidate when it has at least one illicit inflow and at least one illicit outflow in the same case. Roles planned by the generator are used only to check this rule, never to set the label.

| Typology | Mules | Not mules |
|---|---|---|
| M1 | the hub: 150 | 468 sources, 549 exits, 1 member without illicit transfers |
| M2 | every chain member between source and exit: 151 | 50 sources, 50 exits |
| M3 | the collector: 150 | 601 sources, 150 exits, 3 without illicit transfers |
| M4 | collector and relays: 170 | 164 sources, 120 exits |
| cycle | every member: 321 | none |
| fan_in | none | 30 hubs (receive only), 184 sources, 8 without illicit transfers |
| fan_out | none | 30 hubs (send only), 181 destinations |
| **Total** | **942** | **2,589** of 3,531 case members |

**Checks** (`python -m earlymule.data.ground_truth_audit`):
- all 3,059 illicit transfers match the full transaction export;
- 621 of 621 planned layered mules have both directions; 2,373 of 2,377 planned transfers occurred (the rest belong to sources without funds); no illicit transfer outside the plan;
- no layered mule sends before it receives; no account belongs to two cases; every candidate has all four milestones.

## 5. Ground truth table

`data/processed/ground_truth.parquet`, one row per case member, built by `python -m earlymule.data.ground_truth`. Evaluation only: no column may become a model input.

| Column | Meaning |
|---|---|
| `alert_id`, `acct_id` | Case and account |
| `typology` | Case type (section 3) |
| `is_main` | Account that drives the case in the simulator |
| `role` | Structural role from illicit in/out counts: relay, collector, disperser, source, destination, no_illicit |
| `is_mule_candidate` | The label (section 4) |
| `n_illicit_in`, `n_illicit_out` | Illicit transfers received and sent in the case |
| `onset_day` | First illicit transfer of the account, either direction |
| `first_illicit_inflow_day`, `first_illicit_outflow_day` | First illicit transfer per direction; empty if that direction never occurs |
| `completion_day` | Last illicit transfer of the account, either direction |
| `upstream_accounts`, `downstream_accounts` | Case members it received from and sent to |
| `planned_layer` | Generator layer of layered typologies; for audits only |

Compared with the research plan's ground truth list, every field is available except the cash-out time: AMLSim records no cash withdrawals, so completion is the last illicit transfer.

**Timing of the 942 mules.** Onset to completion: median 2 days, 90th percentile 7, maximum 19; 174 mules start and finish on the same day. First inflow before first outflow for 720, same day for 182 (order within a day unknown), outflow first for 40 (the cycle initiators, which send and later receive the loop's return).

## 6. Columns a model may use

Full table with reasons: `reports/column_classification.md`. Enforced in code by `src/earlymule/data/leakage.py` and its tests.

| Class | Columns |
|---|---|
| Input | `transactions.csv`: `orig_acct`, `bene_acct`, `base_amt`, `tran_timestamp` (as a day index); `tx_log.csv` balances; `accounts.csv`: `initial_deposit` |
| Key only | `acct_id`, `tran_id`. Low account IDs are more active, so the ID itself is not a feature |
| Evaluation only | `is_sar`, `alert_id`, `alert_type`, `prior_sar_count` (equals case membership), the whole ground truth table, the alert files, the generator's member and normal-model files |
| Dropped | Constant or empty columns, and random personal and location data (names, addresses, birth dates, coordinates) |

Features at day t use only transfers dated on or before t. Illicit transfers are part of the input, as a bank would see them; only their labels are hidden.

## 7. How experiments use the data

| Item | Current choice |
|---|---|
| Split | By day: train 0–431, validation 432–539, test 540–719 (60/15/25%) |
| Feature windows | 1, 7 and 30 days, end of day t (advisor, 2026-11-02) |
| Scored population | Regular accounts: at least 20 transfers up to day t. Weekly scoring also reports all accounts; daily scoring does not yet |
| Weekly scoring | Every 7 days; positive if the mule's onset falls in the last 7 days; earlier mules are left out |
| Daily scoring (early detection) | Every day; positive while the mule is active (onset to completion); after completion a mule has no label but stays in the ranking; top K new alerts per day with a 30-day cooldown |
| Uncertainty | 3 seeds; 95% intervals by resampling whole cases |

**Difficulty so far** (LightGBM, test, mean of 3 seeds, `experiments/experiments.csv`). Preliminary: early stopping can stop on the default log-loss instead of PR-AUC (weekly seed 1 stopped at round 3), so these runs will be repeated after the fix.
- weekly: PR-AUC 0.75 (random ranking: 0.0001), Recall@100 0.96;
- daily, 10 new accounts alerted per day with a 30-day cooldown (`docs/evaluation_protocol.md`): 76% of mules alerted before their first forwarding (of those that leave a day for it), 96% by their completion day, 65% strictly before it; 30% of alerts land on sources, exits and one-sided hubs of cases.

Weekly scoring is close to its ceiling; daily scoring is where early-detection methods can still be compared.

**Difficulty check** (`reports/difficulty_check.md`, weekly test, regular accounts): static input and activity level alone rank at random (PR-AUC 0.0006 and 0.0008, mule rate 0.0006); the best unlearned rule (distinct payees in the last 7 days) reaches 0.07; LightGBM 0.79. No label leakage was found.

## 8. Changes from the research plan

| Research plan | This dataset | Reason |
|---|---|---|
| Feature windows 1 h, 24 h, 7 d | 1, 7, 30 days | AMLSim records days, not times (advisor, 2026-11-02) |
| M1–M4 mule typologies | Implemented as layered typologies (section 3) | The original fan-in/fan-out hubs are one-sided and are not mules |
| M5 cycle optional | Included | Already available; 321 mules |
| Cash-out time | Not available | AMLSim records no cash withdrawals |

## 9. Known limitations

1. **Activity shortcut.** Case members are drawn from accounts with regular activity, while most accounts barely transact. Counting normal transfers alone separates mules with ROC-AUC 0.92 over all accounts, but only 0.51 over regular accounts; this is why the regular population is the main one. The cause is member selection, not hubs: inside the regular population mules have the same median activity as other accounts (111–114 against 110 normal transfers).
2. **Initial balance follows activity** because of change 5 (ROC-AUC 0.64 over all accounts, 0.50 over regular accounts).
3. **Normal behaviour is very regular.** Normal models pay fixed counterparties on fixed schedules, so "new counterparty" features are strong: 99% of mules meet a new counterparty in the week they start, against 10% of regular accounts. Real customers pay new people more often, so the task is likely easier than in practice.
4. **Mule behaviour is clean.** Mules forward exactly 90% (2 of 621 forward less, because a normal transfer took most of their balance), and illicit amounts are smaller than normal ones (median 156 against 550).
5. **Same-day activity.** 174 mules start and finish on one day; at day resolution they cannot be caught before completion.
6. **One case per mule.** No account is reused across cases, so recruitment and repeated use over weeks are not represented.
7. **Small test set.** About 220 mules in the test period; 95% case-bootstrap intervals are about ±0.05 on weekly PR-AUC and ±0.07 on detection before forwarding.

## 10. Decisions

1. **Main early-detection target**: detection before the first illicit outflow, with detection before and by completion still reported. This changes the research plan's definition, so it was sent to the advisor for confirmation on 2026-11-09.
2. **Sources, exits and one-sided hubs of cases**: negatives in the main results, because a bank does not know which accounts belong to a case. They take about 30% of daily alerts, so results without them are reported alongside.
3. **Regular-population threshold**: at least 20 transfers up to the scoring day, because over all accounts activity alone separates mules (section 9, item 1). All-account results are reported alongside.
4. **Cycle initiators** (40 accounts that send before they receive): kept as mules, because they receive and forward within the case as the agreed definition requires; reported separately, since they cannot be caught before forwarding.
5. **More realistic data** is left for the evaluation phase: a harder version with noisy normal behaviour (occasional new counterparties), legitimate pass-through accounts, mules with variable forwarding and amounts, and mules reused across cases over several weeks.

## 11. Reproduce

In WSL, AMLSim environment, in `AMLSim/` (after `mvn -q compile`):

```bash
python scripts/transaction_graph_generator.py conf_80K-earlymule-mule.json
bash scripts/run_AMLSim.sh conf_80K-earlymule-mule.json
python scripts/convert_logs.py conf_80K-earlymule-mule.json
```

In the repo root, `earlymule` environment (`configs/paths.yaml` selects the dataset):

```bash
python -m earlymule.data.ground_truth          # ground truth table
python -m earlymule.data.ground_truth_audit    # label checks
python -m earlymule.data.shortcut_audit        # single-feature shortcut screening
python -m earlymule.models.tabular --seed 0    # weekly LightGBM
python -m earlymule.models.daily_scoring --seed 0   # daily scoring and early detection
python -m earlymule.evaluation.summarize_runs  # mean and sd over seeds
```

Related records: `reports/ground_truth_check.md`, `reports/column_classification.md`, `AMLSim/EARLYMULE-TG.md`.
