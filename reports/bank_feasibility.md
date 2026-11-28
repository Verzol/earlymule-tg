# Can a bank stop a mule before its first outgoing transfer?

2026-11-16, EMTG-71. Feasibility, from the bank's side, of the early-detection milestone the advisor approved on 2026-11-15: a mule counts as detected early only if it is alerted before its first illicit outgoing transfer. Written so that it can go into the discussion chapter of the thesis. Sources come from a literature and regulation search on 2026-11-15 (working notes: `.local/research/bank_feasibility_2026-11-15/report.md`). Each claim carries a mark: **(primary)** the source itself was read; **(secondary)** a full secondary text was read (law-text mirror, legal note, news); **(unverified)** seen only in a summary, to be checked before it is quoted. Simulator numbers come from `reports/post_run_eda.md`.

## 1. The moment in question

A mule's first outgoing transfer passes through these steps in a mobile or internet banking app:

1. The account holder enters the payee account, amount and reference.
2. The payer authenticates; in Vietnam, biometrically above 10 million VND per transfer or above 20 million VND cumulative per day under Circular 50/2024/TT-NHNN (Art. 10 and Appendices 01–02, in force 1 January 2025, which codified and replaced Decision 2345/QĐ-NHNN repealed by Decision 2872/QĐ-NHNN) (secondary).
3. **The sending bank screens and scores the order.** This is the advisor's proposed moment: the order exists, so the bank knows the payee, the amount and what it knows about the payee.
4. The bank submits the order to the payment scheme (NAPAS 24/7 in Vietnam).
5. The receiving bank credits the payee.

Scheme time limits start only at step 4. The EU Instant Payments Regulation counts its 10 seconds from the "time of receipt", which comes after the payee-name check and the payer's authentication (primary, European Commission Q&A); UK Faster Payments allows payments to be "held pending investigation … prior to submission" (primary, Faster Payments System Principles v11, 2026); FedNow's 20-second clock starts at the timestamp of the message the sending bank submits (primary). For Vietnam no NAPAS rulebook value was found; press reports give 1 to 3 seconds per transfer (unverified). Scoring at step 3 therefore does not consume the scheme's budget; the binding constraint is the customer's patience, and vendors report scores within milliseconds (unverified for Mastercard; Visa, primary).

## 2. What the bank knows at that moment

| Information | Payee at the same bank | Payee at another bank |
|---|---|---|
| Sender's full history, balance, device, login, KYC | Yes | Yes |
| The order: payee account, amount, reference | Yes | Yes |
| Payee's history, balance, own counterparties | Yes | No |
| Payee's own risk score or earlier alerts | Yes | Only alerts the bank raised itself |
| Earlier payments by the bank's customers to that payee | Yes | Yes |
| Payee-name check | Not needed | Yes (EU Verification of Payee, primary; UK Confirmation of Payee, primary; Vietnam: account name shown through NAPAS, unverified) |
| Shared lists of suspicious accounts | Yes | Yes, if the bank queries them (Vietnam: SIMO, secondary) |

The simulator has one bank, so every payee is internal: payee features in the transfer-level scoring (EMTG-72) are a best case. The `external_payee` variant (119 features, cutting payee-side features to the right-hand column) was run in EMTG-72 and detected 99.4% of multi-day mules versus 99.6% for the full set—nearly unchanged because every sender in the single-bank simulator still has a full local history and very regular normal behaviour.

## 3. How much time there is

| Source | Time from receipt to onward transfer |
|---|---|
| Lloyds Bank data on known mule accounts, two months (RUSI, July 2025) (primary) | About 28% of the money received left within 15 minutes, a further 25% within an hour; under 15% was still there after 24 hours |
| FATF, Interpol and Egmont, 2023 (primary) | Recovery works best within 24 to 72 hours of the fraudulent payment |
| Simulator, 942 mules (`reports/post_run_eda.md`) | Median 2 days from onset to the first illicit outflow; 23.6% forward on their onset day, 22.5% one day later; cycles and relay chains have a median of 1 day |

Real mules are much faster than the simulated ones. A daily review can stop a first transfer only if the mule waits at least a day; most real mules do not. Scoring each outgoing transfer before it is executed is the only point at which the money is still certainly in the account. In the simulator it also reaches the 182 mules that receive and forward on the same day, which end-of-day scoring cannot catch in time.

## 4. What the bank may do

| Jurisdiction | Action before execution | Limits |
|---|---|---|
| Vietnam | Delay a transaction linked to a crime or a blacklisted party (Law on Anti-Money Laundering 2022, Art. 44) (secondary) | At most 3 working days; report at once to the competent authority and the State Bank; no liability for a lawful delay |
| Vietnam | Refuse a payment order (Circular 17/2024/TT-NHNN, Art. 21(1)(b)) (secondary) | Listed grounds include invalid or incomplete orders, insufficient balance, closed/frozen accounts, authority requests, prohibited acts (Decree 52/2024/NĐ-CP Art. 8), doubts about truthfulness or transaction purpose under AML law, and other cases agreed in the account contract |
| Vietnam | Block immediately or require identification for transfers to accounts flagged in SIMO (secondary, press quoting the State Bank) | In practice mostly a warning to the payer, who decides; no rule found that delays transfers to flagged accounts |
| United Kingdom | Delay an outbound payment up to 4 business days (SI 2024/1013, in force 30 October 2024) (secondary) | Only when fraud by a person other than the payer is suspected, i.e. to protect a victim; a complicit mule's own transfer goes through the anti-money-laundering consent route (7 working days, then possibly 31 more) (unverified) |
| European Union | No right to reject an instant transfer for suspected fraud under the Instant Payments Regulation; the payer decides (primary, Commission Q&A) | A blocking power is being negotiated in the Payment Services Regulation |

Vietnam also writes the mule pattern into regulation. Appendix 02 of Circular 17/2024/TT-NHNN (updated by Circular 25/2025/TT-NHNN, issued 31 August 2025, which also added Appendix 03 for corporate accounts) sets the monthly reporting template for accounts suspected of fraud or illegal activity. Among the nine suspicious-reason codes of the template are an account that receives money from many different payment accounts and forwards or withdraws it immediately within a very short time leaving little or no balance (code 3), more than 3 incoming transfers from accounts on the suspicious list (code 4), and one device used for more than one account (code 8) (wording checked against the full text of Circular 25/2025, 2026-11-28; section 9 quotes codes 3 and 4). The first two are what the strongest features in the simulator measure (new senders, inflow surge, rapid forwarding) and what the propagation features of EMTG-72 measure.

A warning works on a deceived payer; it does not stop a complicit mule. For mules, the useful action at the moment of the first outgoing transfer is a delay or refusal followed by a report, which Vietnamese law allows for up to 3 working days.

## 5. Evidence that pre-execution scoring is used

Industry services score account-to-account payments in real time, mostly the victim's payment into the mule: Mastercard Consumer Fraud Risk (UK, 2023; uses payment values, payer and payee history and the payee's links to scam accounts) (unverified, primary page blocked); Visa Protect for A2A, whose Pay.UK pilot on 12 months of UK payments found 54% of fraud that had passed the banks' own checks (primary). Pay.UK's Enhanced Fraud Data exchange carries data between sending and receiving banks before a payment (primary). BioCatch reports flagging 70% of later-confirmed mule accounts before their first incoming transfer, from session behaviour (secondary, vendor figure). In Vietnam, SIMO had issued more than 5.6 million warnings and stopped or cancelled more than 1.9 million transactions, over 6,300 billion VND, by early October 2026 (secondary, press). None of these is a controlled evaluation.

The advisor's moment, the mule's own outgoing transfer, is one hop later than most of these services. At that hop the strongest signal is the sender's own recent inflow, which the bank holds in full.

## 6. Verdict

- **Technically feasible.** The bank has the sender's full history and the pending order before it submits the payment; scoring there is common practice and does not use the scheme's time budget.
- **Necessary.** Real mules forward most of the money within an hour, so end-of-day review is too late for most cases; the first outgoing transfer is the last point at which the money is certainly still there.
- **Legally possible in Vietnam, with conditions.** A delay of up to 3 working days with an immediate report (AML Law Art. 44), or a refusal on the grounds of Circular 17/2024.
- **Best case in the simulator.** Every payee is internal; there is no device, login, identity, time-of-day, cash or card data; normal accounts are very regular. Results on the simulator are an upper bound on what the transaction data alone can do.

## 7. Consequences for the thesis

1. The primary metric stays detection before the first illicit outflow under end-of-day scoring (advisor, 2026-11-15).
2. A second task scores each outgoing transfer just before it is executed, with features of the order and the payee (EMTG-72). The within-day execution order is recoverable from the simulator's balance log (`earlymule.data.intraday_order`), so this needs no assumption about which same-day transfer came first.
3. The second task reports a variant with payee features limited to what a bank sees for an external payee, and a variant without simulator shortcut features.
4. The alert budget of the second task is the number of transfers held per day, with false holds per 1,000 transfers, since a held legitimate transfer is a cost to a customer.
5. The discussion chapter states the legal limits above and that a warning, unlike a hold, does not stop a complicit mule.

## 8. Verification status against primary texts (audited 2026-11-19)

- **Verified via official/legal mirrors:** Law on Anti-Money Laundering 2022 Art. 44 (max 3 working days, immediate report, immunity); Circular 17/2024/TT-NHNN Art. 21(1)(b); Circular 50/2024/TT-NHNN (superseding repealed Decision 2345/QĐ-NHNN) thresholds "above 10 million VND" per transfer and "above 20 million VND" per day; Circular 25/2025/TT-NHNN updating Appendix 02 of Circular 17/2024; SIMO has no publicly numbered standalone establishing act (deployed under SBV supervision directives such as Official Letter 3276/NHNN-TT).
- **Checked 2026-11-28:** the wording of the nine suspicious-reason codes of Appendix 02 as replaced by Circular 25/2025/TT-NHNN (Art. 13(1)), from the full text of the circular on thuvienphapluat.vn; the ">3 incoming transfers" threshold is code 4. The circular defines no number for "many" accounts or for "a very short time" in code 3.
- **Still to check if quoted verbatim in thesis:** the same codes in the State Bank's published PDF (Công báo), and official NAPAS 24/7 rulebook timeout values.

## 9. Rule baseline after Appendix 02 (fixed 2026-11-28, before any score)

Evaluation protocol section 12 lists a rule after the Vietnamese indicators as a reference row. This section fixes its form before it is run on any dataset; nothing below is tuned on validation or test.

**Source.** Appendix 02 of Circular 17/2024/TT-NHNN as replaced by Circular 25/2025/TT-NHNN, column 13 ("Lý do nghi ngờ"):

- Code 3: "Tài khoản thanh toán nhận tiền từ nhiều tài khoản thanh toán khác nhau và được chuyển đi hoặc rút ra ngay trong thời gian rất ngắn (không để lại số dư hoặc để lại rất ít)." (the account receives money from many different accounts and it is forwarded or withdrawn within a very short time, leaving no or very little balance)
- Code 4: "Tài khoản thanh toán có hơn 03 giao dịch nhận tiền từ các tài khoản thanh toán nằm trong danh sách có dấu hiệu nghi ngờ liên quan đến lừa đảo, gian lận, giả mạo,..." (more than 3 incoming transfers from accounts on the list of suspected fraud)

The other codes need data the simulator does not have (identity records, online advertising of accounts, warning lists, customer contact, devices) or are open-ended (code 6, unusual transactions, which is what the learned models do).

**Main rule (code 3).** At the end of day t, over the window (t − 7, t] (point-in-time features, `docs/data_spec.md` section 7), an account fires when both hold:

| Condition | Feature | Threshold | Reading of the text |
|---|---|---|---|
| Many different senders | `peers_in_7d` | ≥ 3 | "nhiều tài khoản khác nhau": more than two |
| Forwarded at once, little left | `fwd_1d_7d` | ≥ 0.8 | at least 80% of the week's inflow sent on the same or the next day; a day is the finest time unit of AMLSim, and the simulator records no cash withdrawals |

Fired accounts are ranked by `amt_in_7d` (largest inflow first, then `acct_id`), the order in which a reviewer would take them. Only fired accounts are alerted, under the same budget and cooldown as every model (10 new accounts a day, 30-day cooldown); on a day with fewer than 10 fired accounts the rule raises fewer alerts, and the realised alerts per day are reported. Population, labels and metrics are those of the protocol (regular accounts, detection before the first illicit outflow as the primary metric, before and by completion alongside).

**Second row (code 3 or code 4).** The "list" is the bank's own: accounts the main rule fired on any day up to t − 1. An account also fires when, up to day t, it has received more than 3 transfers from listed accounts. Code-3 accounts come first, in the order above; code-4-only accounts follow, by the number of such transfers.

**Random ranking.** A uniform random score for every account-day of the regular population, the same budget and cooldown; 10 repetitions (seeds 0 to 9), mean and range reported.

**Sensitivity, reported as a grid and never used to choose:** senders ≥ 2, 3, 5 × forwarded share ≥ 0.5, 0.8, on the same window. The main row stays (3, 0.8) whatever the grid shows.

**What is expected, stated in advance.** Code 3 needs forwarding to be observed, and a mule's first illicit outflow is its first forwarding of case money. The rule can therefore alert a mule before that outflow only when the mule's earlier normal activity already matches, so it should score low on the primary metric; its comparison with the models rests on detection by completion. The 0.8 share was set knowing that simulated mules forward exactly 90% (`docs/data_spec.md` section 9, item 4), so on this data the rule's numbers are an upper bound for a rule of this kind.

**Where it runs.** Main dataset (test period, development data) and draws g1 to g5 (test period), from the exported point-in-time features; no training, so one run per dataset.
