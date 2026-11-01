# Ground-truth extraction check

Run `80K-earlymule-a200`, checked 2026-10-31.
Reproduce: `python -m earlymule.data.ground_truth` (writes `data/processed/ground_truth.parquet`), then `pytest tests/test_ground_truth.py`.

## Result

1,494 alert accounts in 200 alerts. 715 are mule candidates (relay, collector or disperser), and all 715 have an onset day and a completion day.

| Typology | Alerts | Collector | Disperser | Relay | Source | Destination | No illicit tx |
|---|---:|---:|---:|---:|---:|---:|---:|
| cycle | 80 | 0 | 0 | 598 | 7 | 7 | 2 |
| fan_in | 60 | 60 | 0 | 0 | 364 | 0 | 33 |
| fan_out | 60 | 0 | 57 | 0 | 0 | 341 | 25 |
| **Total** | 200 | 60 | 57 | 598 | 371 | 348 | 60 |

Days since mule onset to completion, over mule candidates: median 2, 75th percentile 6, max 19.

On the original `10K` run, the script reproduces the counts in the EDA notebook (742 accounts, 345 mule-like, 285 relays); this is a test in `tests/test_ground_truth.py`.

## Hand check of five alerts

Edge lists from `alert_transactions.csv` were compared with the ground-truth rows. The listing script is not committed; it prints each alert's transactions sorted by day next to its rows.

| Alert | Typology | What was checked | Matches |
|---:|---|---|---|
| 0 | fan_in | 8 sources send once on days 213–227; collector 72290 has 8 inflows, onset 213, completion 227, empty outflow day, 8 upstream accounts | Yes |
| 10 | fan_in | 4 of 5 sources send on days 639–643; source 3134 sends nothing and is `no_illicit` | Yes |
| 60 | fan_out | Disperser 61318 sends 6 times on days 200–206; each destination has one inflow | Yes |
| 120 | cycle | 9 relays; 36247→61 and 61→8096 both on day 350, so account 61 has inflow day = outflow day = 350 | Yes |
| 138 | cycle | 6 relays, one hop each, days 287–300; every non-main relay receives before it sends | Yes |

In a cycle the main account starts the chain, so its first outflow comes before its first inflow (alert 120: out 339, in 351). That is expected and is not an ordering error.

## Schema coverage

Fields from the ground-truth schema in the research plan (D1):

| Field | Column | Available |
|---|---|---|
| mule_id | `acct_id` (with `alert_id`) | Yes |
| typology | `typology` | Yes |
| role | `role` | Yes, from a structural rule on illicit in/out counts; not the final mule label |
| mule_start_time | `onset_day` | Yes |
| first_illicit_inflow_time | `first_illicit_inflow_day` | Relays and collectors only; empty for dispersers |
| first_illicit_outflow_time | `first_illicit_outflow_day` | Relays and dispersers only; empty for collectors |
| completion_time | `completion_day` | Proxy: last illicit event in either direction |
| cashout_time | none | No. AMLSim writes no cash transactions (`cash_tx.csv` is empty) |
| upstream_accounts / downstream_accounts | same names | Yes, within the same alert |

All times are day indices. AMLSim has no time of day, so events on the same day have no known order.

## Findings

1. **Cycle ordering is fixed.** Of 523 non-main cycle relays, none forwards money before first receiving it (45% on the original AMLSim `10K` run). 145 receive and forward on the same day.
2. **60 alert members have no illicit transaction.** AMLSim still marks them `is_sar`, so `is_sar` must not be used as the label.
   - Three fan_out alerts (78, 88, 111) have no transactions at all: the main account had almost no balance when the alert started, so all 22 members are silent. This is why there are 57 dispersers, not 60.
   - 20 of 33 silent fan_in sources had a balance under 1 at alert start, drained by normal transactions. The other 13 had money; the cause is not known yet.
3. Silent members are kept in the table with `role = no_illicit` and empty days, so they can be excluded or studied later.
