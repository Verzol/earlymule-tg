# Can a bank stop a mule before its first outgoing transfer?

2026-11-16, EMTG-71. Feasibility, from the bank's side, of the early-detection milestone the advisor approved on 2026-11-15: a mule counts as detected early only if it is alerted before its first illicit outgoing transfer. Written so that it can go into the discussion chapter of the thesis. Sources come from a literature and regulation search on 2026-11-15 (working notes: `.local/research/bank_feasibility_2026-11-15/report.md`). Each claim carries a mark: **(primary)** the source itself was read; **(secondary)** a full secondary text was read (law-text mirror, legal note, news); **(unverified)** seen only in a summary, to be checked before it is quoted. Simulator numbers come from `reports/post_run_eda.md`.

## 1. The moment in question

A mule's first outgoing transfer passes through these steps in a mobile or internet banking app:

1. The account holder enters the payee account, amount and reference.
2. The payer authenticates; in Vietnam, biometrically above 10 million VND per transfer or 20 million VND per day (Decision 2345/QĐ-NHNN, in force 1 July 2024) (secondary).
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

The simulator has one bank, so every payee is internal: payee features in the transfer-level scoring (EMTG-72) are a best case. The plan includes a run with payee features cut to the right-hand column.

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
| Vietnam | Refuse a payment order (Circular 17/2024/TT-NHNN, Art. 21) (secondary) | Listed grounds include frozen accounts, authority requests, prohibited acts and agreed cases; whether suspected fraud is itself a ground was not confirmed word for word (unverified) |
| Vietnam | Block immediately or require identification for transfers to accounts flagged in SIMO (secondary, press quoting the State Bank) | In practice mostly a warning to the payer, who decides; no rule found that delays transfers to flagged accounts |
| United Kingdom | Delay an outbound payment up to 4 business days (SI 2024/1013, in force 30 October 2024) (secondary) | Only when fraud by a person other than the payer is suspected, i.e. to protect a victim; a complicit mule's own transfer goes through the anti-money-laundering consent route (7 working days, then possibly 31 more) (unverified) |
| European Union | No right to reject an instant transfer for suspected fraud under the Instant Payments Regulation; the payer decides (primary, Commission Q&A) | A blocking power is being negotiated in the Payment Services Regulation |

Vietnam also writes the mule pattern into regulation. Appendix 02 of Circular 17/2024 lists, among the signs a bank must report: an account that "receives money from many different accounts and sends it out or withdraws it within a very short time", "more than 3 incoming transfers from accounts on the list of suspicious accounts", and one device (MAC address) used for more than one account (secondary). The first two are what the strongest features in the simulator measure (new senders, inflow surge, rapid forwarding) and what the propagation features of EMTG-72 will measure.

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

## 8. To check against primary texts before quoting

- Circular 17/2024/TT-NHNN Art. 21(1)(b) and Appendix 02, exact wording (official text blocked automated access).
- Law on Anti-Money Laundering 2022 Art. 44, exact wording.
- The State Bank act that established SIMO, and any rule on holding transfers to flagged accounts.
- Decision 2345/QĐ-NHNN: "above" or "from" 10 million VND.
- NAPAS 24/7 timeout values.
