# Ground-truth extraction check

Run `80K-earlymule-a200`, extraction checked 2026-10-31; label rule corrected and audited 2026-11-02.
Reproduce: `python -m earlymule.data.ground_truth` (writes `data/processed/ground_truth.parquet`), then `pytest tests/test_ground_truth.py`.

## Result

1,494 alert accounts in 200 alerts. **598 accounts satisfy the operational mule-candidate rule:** at least one observed illicit inflow and one observed illicit outflow in the **same alert**. All 598 have onset, completion, first-inflow and first-outflow days. The previous label incorrectly included 60 one-sided collectors and 57 one-sided dispersers, giving 715 positives.

Structural roles are preserved for analysis. A collector or disperser name alone does not establish an intermediary. Normal transfers, account balance, generator membership and transfers belonging to another alert cannot supply a missing illicit side. Non-candidates are not asserted to be innocent in the real world; their inclusion/exclusion from evaluation remains a protocol decision.

| Typology | Alerts | Collector | Disperser | Relay | Source | Destination | No illicit tx |
|---|---:|---:|---:|---:|---:|---:|---:|
| cycle | 80 | 0 | 0 | 598 | 7 | 7 | 2 |
| fan_in | 60 | 60 | 0 | 0 | 364 | 0 | 33 |
| fan_out | 60 | 0 | 57 | 0 | 0 | 341 | 25 |
| **Total** | 200 | 60 | 57 | 598 | 371 | 348 | 60 |

Days since mule onset to completion, over the corrected 598 candidates: median 2, max 19. All are in cycle alerts; the present standalone fan-in/fan-out patterns do not generate a two-sided intermediary. This is a dataset coverage limitation and a possible topology shortcut for a classifier.

On the original `10K` run, the script reproduces the notebook's structural counts (742 accounts, 345 relay/collector/disperser rows, 285 relays), but only the 285 relays are now positive. The notebook's earlier label counts are historical.

## Label audit (2026-11-02)

Run `python -m earlymule.data.ground_truth_audit` to regenerate `data/processed/ground_truth_label_audit.csv` (every alert member) and `.json` (summary). These are evaluation-only artifacts. All 1,310 illicit edges match the complete `transactions.csv` export by transaction ID, alert, endpoints, amount and day; all endpoints have alert membership. Every member's in/out counts were independently checked against that full export. No account belongs to multiple alerts in this run.

| Check | Result |
|---|---:|
| Previous positives | 715 |
| Corrected two-sided candidates | 598 |
| Removed collectors (no illicit outflow) | 60 |
| Removed dispersers (no illicit inflow) | 57 |
| Positive prevalence over all 96,344 accounts | 0.6207% |
| Candidates with missing milestones | 0 |
| Silent alert members, retained as non-candidates | 60 |

57/60 collectors have normal outgoing payments during their alert's observed first-to-last illicit day; 27/57 dispersers have normal incoming payments in that window. Those normal payments are not marked as belonging to the laundering case, and AMLSim does not track the provenance of money. They therefore cannot justify retaining these 117 labels. The Java fan-in model generates only sources-to-hub edges; fan-out generates only hub-to-destination edges. This is not an omitted direction in the extraction script.

### Two-sided membership versus forwarding order

The implemented predicate checks membership in both directions; it does **not** claim to prove that a particular inflow funded a later outflow.

| First illicit inflow versus first illicit outflow | Candidates |
|---|---:|
| Inflow on an earlier day | 378 |
| Same day; within-day order unavailable | 145 |
| Outflow on an earlier day (cycle main accounts) | 75 |

The 75 cycle main accounts initiate a loop and later receive its return payment. They have both directions, so they remain candidates under the explicit two-sided predicate. **If "exclude the source at the start" is intended to exclude cycle initiators even when they later receive money, exclude these 75 separately: that alternative gives 523 candidates.** Requiring a provably earlier inflow day additionally drops 145 same-day accounts and gives 378. Neither stronger rule is silently substituted for the approved two-sided wording; record the cycle-initiator treatment in the evaluation protocol before interpreting early-detection results.

Onset remains the first illicit event in either direction, and completion remains the last. For cycle initiators, onset is their outgoing event. Sources and destinations without both directions remain non-candidates. The 60 silent members have empty milestones.

The earlier ROC-AUC figures in `column_classification.md` used 715 positives; they must be remeasured with this rule in EMTG-42. No corrected PR-AUC or LightGBM result has been produced by this audit.

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
| (generator plan) | `planned_layer` | Layered mule typologies only (0 = sources); `<NA>` otherwise. For audits, never for labels or features |

All times are day indices. AMLSim has no time of day, so events on the same day have no known order.

## Findings

1. **Cycle ordering is fixed.** Of 523 non-main cycle relays, none forwards money before first receiving it (45% on the original AMLSim `10K` run). 145 receive and forward on the same day.
2. **60 alert members have no illicit transaction.** AMLSim still marks them `is_sar`, so `is_sar` must not be used as the label.
   - Three fan_out alerts (78, 88, 111) have no transactions at all: the main account had almost no balance when the alert started, so all 22 members are silent. This is why there are 57 dispersers, not 60.
   - 20 of 33 silent fan_in sources had a balance under 1 at alert start, drained by normal transactions. The other 13 had money; the cause is not known yet.
3. Silent members are kept in the table with `role = no_illicit` and empty days, so they can be excluded or studied later.

## Layered mule typologies and shortcut screening (2026-11-03)

The AMLSim fork now has layered mule typologies for M1–M4 (`AMLSim/EARLYMULE-TG.md`, sections 4–5). The label is unchanged: a candidate still needs an observed illicit inflow and outflow in the same alert. Two audits were added:

- `python -m earlymule.data.ground_truth_audit` also compares layered alerts with the generator plan (`layered` in the JSON): planned versus observed edges, unplanned illicit edges, planned mules that are two-sided, out-before-in, same-day forwarding, and the forward ratio.
- `python -m earlymule.data.shortcut_audit` ranks accounts by one whole-period activity feature at a time, in four populations: all accounts, active (any normal transfer), regular (at least 20 normal transfers), and regular without the other alert members. Output: `data/processed/shortcut_audit.csv`.

| Check | `80K-earlymule-a200` | `10K-earlymule-mule` (smoke test) | `80K-earlymule-mule` |
|---|---:|---:|---:|
| Mule candidates | 598 (all cycle) | 311 (148 cycle, 163 layered) | 750 (307 cycle, 175 M4, 168 M2, 50 M1, 50 M3) |
| Planned layered mules two-sided | n/a | 163 / 163 | 443 / 443 |
| Planned edges observed | n/a | 476 / 477 (one source never had funds) | 1,251 / 1,256 (sources without funds) |
| Unplanned illicit edges; layered mules out before in | n/a | 0; 0 | 0; 0 |
| Layered mules forwarding on the day they receive | n/a | 38 | 106 |
| Normal transfer count, ROC-AUC: all / active / regular | 0.92 / 0.78 / 0.53 | 0.92 / 0.80 / 0.51 | 0.91 / 0.75 / 0.49 |
| `initial_deposit`, ROC-AUC: all / regular | 0.50 / 0.50 | 0.64 / 0.49 | 0.63 / 0.49 |

Activity alone separates candidates over all or active accounts because alert members are drawn from accounts with regular normal activity. Among regular accounts every screened feature stays between 0.45 and 0.57 (0.48–0.51 in `80K-earlymule-mule`), so the evaluation population should be restricted to regular accounts; the exact rule (threshold, and counting only transfers before the cut-off day) belongs in the evaluation protocol. The higher `initial_deposit` score in the smoke test comes from the balance boost, which follows activity.

In `80K-earlymule-mule`, M1 and M3 give one mule per alert, so they have 50 candidates each against 168–307 for the other typologies. Per-typology results need enough mules in each test split; more M1/M3 alerts would balance this.
