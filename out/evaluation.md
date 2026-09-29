# Evaluation: does the router work, and how often does it not?

Truth = the team that **closed** each request (`resolution_log.final_team`), with the two renamed teams merged (policy §5). The bot's `team_label` is shown as a baseline, not as truth: it matches the closing team on only 77% of the 10,822 historical requests.

## 1. Out-of-time test (the main number)

Trained on 8,687 requests from 1 Apr 2025 – 31 Mar 2026; scored on the **2,135 requests of Apr – Jun 2026** that it never saw (all on the new CRM, like the Jul – Sep test set).

| | matches closing team | matches bot label |
|---|---|---|
| **Router (this work)** | **85.7%** (95% CI 84.3% – 87.2%) | 88.1% |
| Vendor bot | 76.7% (95% CI 74.9% – 78.5%) | 100% |
| Model trained to copy the bot's labels (the brief as written) | 77.0% | **91.8%** |

The copy-the-bot model passes the "90% match with the labels" bar and routes no better than the bot. Matching the labels is the wrong test; matching where requests are actually closed is the right one.

### Where it is right and wrong

| request type | share | router right | bot right |
|---|---|---|---|
| States a need, wording seen before | 83.6% (1,785) | 98.0% | 87.8% |
| States no need ("please call me about my purifier") | 15.7% (336) | 20.2% | 19.3% |
| States a need, new wording | 0.7% (14) | 85.7% | 28.6% |

### Per team

| team | precision | recall | requests |
|---|---|---|---|
| Billing | 99.5% | 85.5% | 255 |
| Filters & Consumables | 97.2% | 84.1% | 207 |
| Installs & Demo | 98.5% | 81.3% | 321 |
| Product Advice | 95.9% | 79.2% | 269 |
| Repairs | 64.3% | 99.0% | 498 |
| Returns & Replacement | 97.4% | 81.0% | 326 |
| Warranty Claims | 97.6% | 79.9% | 259 |

Confusion matrix (rows = closing team, columns = router):

```
     BIL  CON  INS  ADV  REP  RET  WAR
BIL  218    2    1    3   31    0    0
CON    0  174    1    3   28    1    0
INS    0    1  261    2   55    2    0
ADV    0    1    0  213   53    1    1
REP    0    0    2    0  493    0    3
RET    0    0    0    1   60  264    1
WAR    1    1    0    0   47    3  207
```

Most errors sit in the columns for Repairs: that is where vague requests go (best available guess).

### Is the confidence honest?

| confidence level | share of requests | router right |
|---|---|---|
| high | 83.6% | 98.0% |
| medium | 0.6% | 100.0% |
| low | 15.8% | 20.1% |

"High" can be auto-routed; "low" is almost entirely the vague requests that need one question.

## 2. Stability: 5-fold cross-validation on all 10,822 requests

Router: **86.1% ± 0.3%** (folds 85.8% – 86.4%). Requests that state a need: 98.3%. Vague requests: 21.7%.

## 3. New wording: what happens when customers phrase things differently

Every phrasing in a validation fold is held out of training, so the router must fall back on its keyword rules and text model. This is harsher than the test set (99.5% of test wording has been seen before).

| keyword rules used for new wording | overall | requests that state a need | range across folds |
|---|---|---|---|
| Shipped rules (written after reading the history – optimistic) | 86.1% | 98.3% | 98.0% – 98.7% |
| **Fair check: rules from the teams.csv wording only** | **77.1%** | **87.3%** | 69.8% – 98.0% |

The shipped rules cover every phrasing in the history, so the first row is not a real test of new wording. The second row is: expect genuinely new customer wording to be routed correctly about 87.3% of the time, which is why new wording is marked for review when unsure.

## 4. How often it fails, and why

Out-of-time it is wrong on **305 of 2,135 requests (14.3%)**:

* **268 vague requests** – the message states no need, and past requests like these closed in all seven teams. No text-only router can do better than ~22% here; the fix is one question at intake (section 5).
* **37 requests that did state a need** (2.1% of them). 29 of these are records that say 0 transfers yet closed in a different team than they opened in; most of the rest read as clear requests closed by an unrelated team (e.g. "how to clean air fryer" closed by Billing). These look like recording noise in the resolution log, not routing mistakes, and set a ceiling of ~98% on requests with a stated need.

Random sample of the stated-need errors:

| request | router | closed by | transfers |
|---|---|---|---|
| pls help - vacuum arrived damaged, need replacement [no.] | Returns & Replacement | Warranty Claims | 0 |
| sir filter change reminder for room heater [no.] | Filters & Consumables | Product Advice | 2 |
| which mixer is right for 4 people thanks | Product Advice | Returns & Replacement | 0 |
| hi, how to clean air fryer very disappointed | Product Advice | Billing | 0 |
| sir box was open, ceiling fan scratched kindly resolve | Returns & Replacement | Warranty Claims | 0 |
| namaste, which vacuum is right for 4 people kindly resolve | Product Advice | Billing | 0 |
| hello team, need warranty certificate mixer grinder kindly resolve | Warranty Claims | Repairs | 1 |
| hello team, order candle and membrane for induction cooktop pls call back | Filters & Consumables | Billing | 1 |
| water purifier missing parts in box asap | Returns & Replacement | Warranty Claims | 0 |
| urgent: warranty card for room heater not registered asap | Warranty Claims | Product Advice | 0 |
| namaste, how to clean mixer grinder pls call back | Product Advice | Filters & Consumables | 0 |
| induction cooktop missing parts in box asap | Returns & Replacement | Installs & Demo | 0 |

## 5. What if vague requests were clarified at intake? (a projection, not a measurement)

If the 15.7% vague requests answered one multiple-choice question and were then routed as accurately as requests that state a need (97.9%), overall accuracy would be about **97.9%**. That assumes customers answer and answer correctly, which only a pilot can confirm.

## 6. Expected score on the Jul – Sep 2026 test set

Test mix: 83.7% seen wording, 15.7% vague, 0.5% new wording (same mix as the Apr – Jun window).
Weighted by the cross-validated accuracy of each type: **86.2%**; the out-of-time test gave 85.7%. Expected range **84–87%** against the closing team. Against the bot's labels it would score about 88.1%.

## 7. What it is worth, and who is busiest

**Misrouting today** (last 12 months, 8,655 requests, ~721 a month): 23.1% opened in the wrong team, 3,129 transfers. At Rs 305 per transfer and Rs 260 per extra contact that is **Rs 14.7 lakh a year**, about Rs 697 per misrouted request. Misrouted requests also take twice as long to close (median 16.6 h vs 8.2 h).

| scenario | wrong first team | cost of misrouting / year | licence | total vs today |
|---|---|---|---|---|
| Today: vendor bot | 23.3% | Rs 14.1 lakh | Rs 3.2 lakh | – |
| Router, bot switched off | 14.3% | Rs 8.6 lakh | Rs 0 | **saves Rs 8.7 lakh** |
| Router + one intake question for vague requests (projection) | ~2.1% | Rs 1.2 lakh | Rs 0 | saves up to Rs 16.0 lakh |

Error rates are the out-of-time rates from section 1 applied to 12 months of volume; cost per misroute is the historical average (it includes that request's own transfers). Run cost of the router: **Rs 0 per request** – no API, no per-request fee; it answers in under 1 ms on a laptop CPU.

**Who is busiest** (last 6 months, requests per month). Planning headcount on the bot's labels would over-staff Repairs, Billing and Filters & Consumables and under-staff Installs, Returns and Warranty:

| team | bot label says | actually closed | share | bot over/under-states by |
|---|---|---|---|---|
| Repairs | 208 | 166 | 23.0% | +42 |
| Returns & Replacement | 82 | 107 | 14.8% | -25 |
| Installs & Demo | 78 | 105 | 14.5% | -27 |
| Product Advice | 77 | 91 | 12.6% | -14 |
| Billing | 114 | 90 | 12.5% | +24 |
| Warranty Claims | 68 | 90 | 12.5% | -22 |
| Filters & Consumables | 96 | 73 | 10.1% | +23 |

