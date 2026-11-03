# AMLSim output columns: model input or evaluation only

Run `80K-earlymule-a200`, checked 2026-11-01. The split is enforced in `src/earlymule/data/leakage.py` and tested in `tests/test_leakage.py`.

**2026-11-03:** the configured dataset is now `80K-earlymule-mule`. Every output file has the same columns as `80K-earlymule-a200`, so the classes below still apply. Generator-only additions (`layer`, `minDwell`, `maxDwell` in `alert_members.csv`, `planned_layer` in the ground truth) are evaluation-only. In the new run `initial_deposit` follows activity because of the balance boost (see `ground_truth_check.md`).

Each column is one of:
- **Input**: a bank would know it at the time; feature code may use it, but only from transactions dated on or before the prediction day.
- **Key**: used to join tables, never as a feature value.
- **Evaluation only**: a simulator label, or something derived from one. Used for ground truth and scoring.
- **Drop**: constant, empty or random filler.

"ROC-AUC" below means one column used alone to rank the 96,344 accounts, with the 715 mule candidates of the ground-truth table as positives (0.50 = no signal).

**Label correction, 2026-11-02:** those scores are historical and used the previous 715-positive rule. The corrected same-alert, two-sided rule gives 598 candidates; see `ground_truth_check.md`. Column provenance decisions still apply, but all label-dependent scores and percentages below must be recomputed in EMTG-42 before being reported with the new labels.

## accounts.csv

| Column | Class | Reason |
|---|---|---|
| `acct_id` | Key | Low IDs are more active and more often alert members (ROC-AUC 0.64, see below) |
| `initial_deposit` | Input | Known at account opening; ROC-AUC 0.50 |
| `prior_sar_count` | Evaluation only | Is true for exactly the alert members (100% match) |
| `dsply_nm` | Drop | `C_<acct_id>`, a copy of the key |
| `type`, `acct_stat`, `acct_rptng_crncy`, `branch_id`, `bank_id`, `country`, `open_dt`, `close_dt` | Drop | One value for every account |
| `tx_behavior_id` | Drop | Empty |
| `first_name`, `last_name`, `street_addr`, `ssn` | Drop | Random personal data, unique or near-unique per account |
| `city`, `state`, `zip`, `lat`, `lon` | Drop | Random location data; ROC-AUC 0.50–0.51 |
| `gender`, `birth_date` | Drop | Random; ROC-AUC 0.50. Birth dates go up to 2019, after the simulation starts (2017) |

## transactions.csv

| Column | Class | Reason |
|---|---|---|
| `tran_id` | Key | Sequence number |
| `orig_acct`, `bene_acct` | Input (graph edge) | Sender and receiver; also keys |
| `base_amt` | Input | Amount |
| `tran_timestamp` | Input | Converted to a day index; one step is one day at 00:00 |
| `tx_type` | Drop | Always `TRANSFER` |
| `is_sar`, `alert_id` | Evaluation only | Simulator labels |

## tx_log.csv

Same 1,586,209 transactions as `transactions.csv`, in simulator form, with balances.

| Column | Class | Reason |
|---|---|---|
| `step` | Input | Day index |
| `amount`, `nameOrig`, `nameDest` | Input | Same as `base_amt`, `orig_acct`, `bene_acct` |
| `oldbalanceOrig`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Input | A bank knows the balance at transaction time |
| `type` | Drop | Always `TRANSFER` |
| `isSAR`, `alertID` | Evaluation only | Simulator labels |

## Whole files

| File | Class | Reason |
|---|---|---|
| `alert_accounts.csv`, `alert_transactions.csv`, `sar_accounts.csv` | Evaluation only | Alert membership and illicit transactions |
| `tx_count.csv` | Evaluation only | Count of illicit transactions per day |
| `AMLSim/tmp/<run>/alert_members.csv` | Evaluation only | Generator view of alerts, including the main account |
| `AMLSim/tmp/<run>/normal_models.csv` | Evaluation only | Which normal pattern each account was placed in, and its hub; a bank does not see this |
| `AMLSim/tmp/<run>/accounts.csv` | Drop | Copy of `accounts.csv` plus `IS_SAR` |
| `accountMapping.csv` | Drop | One customer per account, one role, one date |
| `individuals-bulkload.csv`, `organizations-bulkload.csv` | Drop | Party records for graph-database import, mostly empty |
| `cash_tx.csv`, `resolvedentities.csv` | Drop | Empty: no cash transactions, no entity resolution |

## Findings

**Static columns carry no label signal except through the account ID.** Every allowed static input scores ROC-AUC 0.50–0.51. `prior_sar_count` is a direct copy of the label.

**Account ID order leaks activity.** Accounts are created in six blocks of 16,000 (`paramFiles/80K-earlymule-a200/accounts.csv`). In the lowest tenth of IDs, 1.80% of accounts are mule candidates; in the highest tenth, 0.01%. Low IDs are also more active: 62% of accounts with ID below 9,634 have normal transactions, against 29% above. The ID is therefore a key only.

**Activity alone separates mules (carried over to the difficulty check).** 67.3% of all accounts have no normal transaction in 720 days, while 98.3% of mule candidates have some (median 120 normal transactions, against 0 for the other accounts). Counting normal transactions alone gives ROC-AUC 0.92 over all accounts and 0.80 over the 31,509 active accounts. This is a side effect of choosing alert members among active accounts. It is behavioural, not a column, so it is handled in the difficulty and leakage check: candidate fixes are scoring only active accounts or raising normal activity across the whole population.
