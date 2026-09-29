# Findings and decisions

A log of what the data showed, what I decided, and why. Every number here is produced by a script in this
repo (named in brackets). Truth throughout = the team that **closed** the request
(`resolution_log.final_team`), with renamed teams merged.

## 1. The brief's target is the wrong target

* The bot's `team_label` matches the closing team on **77.2%** of 10,822 requests, steadily every month
  (74–79%) [scratch analysis; reproduced in `evaluate.py` §1 as 76.7% for Apr–Jun 2026].
* A model trained to copy `team_label` reaches **91.8%** agreement with it (Ritu's 90% bar passed) and
  **77.0%** against where requests were actually closed, which is no better than the bot [`evaluate.py` §1].
* So "match the labels at 90%" would rebuild the bot, including its mistakes. **Decision:** train and score against
  the closing team, and report agreement with the bot's labels alongside it so Ritu can see both.
* The brief says we are scored "against outcomes you do not have". The closing team in the resolution log is the
  outcome we do have for history, so it is the best stand-in.

## 2. Where the bot goes wrong (confusion of bot label vs closing team, all history)

* When the bot says **Product Advice, Installs, Returns or Warranty it is ~96% right**.
* When it says **Billing (65%), Filters & Consumables (60%) or Repairs (64%)** it is often wrong:
  * **Payment remarks.** "water purifier not turning on *paid on upi*". The bot sent 525 of 626 such requests to
    Billing, and Billing closed **1** of them. Policy §3 says that mentioning payment does not make a request Billing.
    Meenal's email describes the same thing.
  * **"… purifier not working"** endings go to Filters & Consumables and are closed by Repairs
    ("Consumables keep getting purifier breakdowns", Meenal).
  * **Vague requests** ("please call back regarding my purifier", "issue with fan", "X query") go to Repairs, and
    are closed by every team.

## 3. How requests are really resolved (what the router learns)

* Messages are templated: greeting + one or two requests + filler (order/reg numbers, "pls call back",
  "thanks") + sometimes a payment remark. There are only 132 distinct deciding requests in 10,822 messages
  [`kestrel/text.py`].
* **Two requests in one message → the last one decides:** 542 of 549 (98.7%) two-request messages were closed by
  the team of the last request [`Router.two_request_stats`]. A bag-of-words model cannot see order, so the router
  treats the last request separately.
* **Vague last request → the outcome is spread across all seven teams**: Repairs 21.8%, Installs 15.8%,
  Returns 14.6% … over 1,727 requests. Channel, product, warranty status and month carry no usable signal (checked
  by cross-tab and by adding them to models: +0.0 to +0.6 points, within noise). This holds even when an earlier
  part of the message is specific (120 cases: the earlier part's team closed it only 15% of the time).
  **About 16% of requests cannot be routed from the text by anyone.** That caps any text-only router at about
  86% against the closing team.
* **Noise floor.** Of requests with a clear request, ~2% are closed by an unrelated team (for example "how to clean
  air fryer" closed by Billing), often with 0 transfers recorded. 126 records say 0 transfers yet closed in a
  different team than they opened in. These look like recording noise, and they cap clear requests at ~98%.

## 4. Data fixes (all in `kestrel/data.py`, each tested in `tests/test_pack.py`)

| issue | evidence | fix |
|---|---|---|
| Two teams renamed on 15 Jan 2026 (policy §5) | old names only before, new names only after | merged; predictions use the current names |
| Legacy Zoho text garbled (Tanmay's email) | 11% of legacy rows: "urgÃ©nt", "Ã¢â‚¬Â¦" | reversed the cp1252/UTF-8 double encoding; 0 left |
| Legacy Zoho `resolved_at` in UTC (policy §9) | 26% of legacy rows "resolved" before they were created; legacy median 4.0 h vs CRM 9.5 h | +5:30 for legacy rows → 0 negative, medians 9.5 h both |
| Inconsistent records | 126 rows: 0 transfers but first team ≠ closing team | kept (excluding them from training: +0.2 point out-of-time, −0.1 in CV, i.e. no effect) |

## 5. What I tried (`experiments/compare_models.py`, `experiments/hybrid.py`, `experiments/llm_benchmark.py`)

Three validation schemes: **A** out-of-time (train to Mar 2026, score Apr–Jun 2026: 2,135 requests),
**B** 5-fold CV, **C** held-out wording (every phrasing in a fold is unseen).

| approach | A: vs closing team | notes |
|---|---|---|
| Vendor bot | 76.7% | baseline |
| TF-IDF + LR trained on the bot's labels (the brief as written) | 77.0% | 91.8% agreement with the bot |
| Majority class | 23.3% | |
| TF-IDF + LR / SVM / Naive Bayes on outcomes | 82.3–83.3% | cannot tell which of two requests came last |
| + word+char n-grams | 83.8% | |
| + separate features for the last request | 84.8% | order problem fixed |
| + channel/product/warranty | 85.4% | +0.6 here, +0.1 in CV: noise |
| Frozen bge-small sentence embeddings (local) + LR | 85.7% | ~100× slower, needs PyTorch |
| Lookup of past outcomes for the last request | 85.8% | best, and explainable |
| Hand-written rules only | 84.7% | rules were written after reading the data |
| **Chosen router (after two parsing fixes)** | **85.7%** (CV 86.1% ± 0.3) | 98.0% on requests that state a need |
| Local LLM Qwen2.5-3B, zero-shot (300-request sample) | 71.3% | same 300: router 84.3%, bot 74.0%; 2.9 s/request |
| Local LLM Qwen2.5-3B, with data-derived hints | 68.0% | hints made it worse; 3.4 s/request |

**Held-out wording (C)** is where approaches really differ: TF-IDF collapses to 45–47% (it has never seen the
words); embeddings 79%; rules built only from the teams.csv wording 77% overall / 87% on stated needs. The test
set reuses 99.5% of known wording, so C matters for live traffic, not for the test score.

## 6. What I changed along the way

1. **Target**: bot labels → closing team (section 1).
2. **Word order**: added the last request as its own input: +1.8 points overall (83.0% → 84.8% out-of-time).
3. **Bug found in error analysis**: "please call back regarding X" had "please call back" stripped as a
   sign-off, leaving "regarding X", which was routed to Installs. Fixed and covered by a regression test.
4. **Vague = last request is vague**, not "whole message is vague". The error analysis found 120 messages that start
   specific and end vague; their outcomes are as random as fully vague ones.
5. **Vague requests go to the pooled best guess (Repairs, ~22%)** and not to per-phrase majorities, which were
   fitting noise (e.g. "someone contact me about X" → Billing at 18%). They are also flagged, with one clarifying
   question.
6. **Honest robustness number**: my first held-out-wording score (98%) used rules I wrote after reading every
   phrasing. I added a fair rule set from teams.csv only (87%) and report both.
7. **Conflicting keywords on new wording** ("jar lid … cracked") now lower confidence and ask, instead of
   silently picking the first rule.

## 7. What I threw away, and why

* **Training on `team_label`**: passes the 90% bar and fixes nothing (section 1).
* **Sentence embeddings in the product**: same accuracy on known wording, ~100× slower, adds PyTorch (~2 GB) to a
  service that must start on a clean machine. They were slightly better than TF-IDF on unseen wording, but keyword
  rules plus clarification are simpler and cost nothing.
* **A local LLM in the path**: 71% zero-shot and about 3 s per request on the laptop. A paid API model would add
  exactly the per-request bill Farhan does not want, and it could not fix the vague 16% either: the information is
  not in the message.
* **Metadata features** (channel, product, warranty): no measurable gain; the endpoint accepts them but ignores them.
* **Dropping inconsistent records from training**: no measurable gain.
* **Per-phrase majorities for vague requests**: these fit noise.

## 8. Champion vs challengers (a second, pre-registered round)

After shipping, I tested 11 more alternatives against the router under a protocol committed before running them
(`docs/CHALLENGER_PROTOCOL.md`): rolling-origin backtest over three quarters (6,502 predictions), McNemar tests with
Holm correction, then CV and held-out wording. Challengers included vague guesses by product, product × warranty and
recent months; lookup thresholds; gradient boosting with metadata and time features; a fine-tuned transformer;
noise-cleaned training; kNN on embeddings; and a majority-vote ensemble.

**Result: no challenger beat the router significantly, so it stays** (`docs/CHALLENGER_RESULTS.md`). The closest
was C1, which uses product to pick the best guess for vague requests: +0.25 points in the backtest, +0.15 in CV,
Holm p = 0.244. It is worth about Rs 12,000 a year if real, so it goes on the list to re-test on fresh data after the
side-by-side run. The fine-tuned transformer made exactly the router's predictions on clear requests, and gradient
boosting and kNN were significantly worse. An oracle ceiling of 86.25% (vs the router's 86.11% on the same
quarters) shows there was almost no room left.

## 9. Round 3: can anything else raise accuracy? (`experiments/operating_modes.py`, `out/operating_modes.md`)

**Remaining signals, ruled out:**
* **Outcome noise on clear requests (1.6%) is unpredictable.** It spreads evenly over all teams, and no column
  predicts it except a weak time-of-day link (p = 0.003, surprise rate 1.0–2.4%). Even the worst slot leaves the
  usual team at 97.6%, so no prediction could ever change.
* **Recent context for vague requests:** the majority closing team of clear requests (same product or all, 7/30/90
  days back, only closed ones) predicts vague outcomes 16.0–21.9% of the time, against 21.8% for "always Repairs".

**Operating levers, measured on the same backtest (6,502 predictions):**
* **Selective routing:** auto-routing the 84.3% of requests with confidence ≥ 0.70 gives **98.4%** accuracy on those;
  15.7% (almost all vague) go to a person or the clarifying question.
* **Top-k:** the right team is in the top 2 for 88.8% and the top 3 for 91.1% of requests, but for vague requests
  only 36.8% / 50.4%. Offering options is no substitute for asking.
* **Vague rule vs metric:** the shipped rule ("always Repairs") is already best for macro-F1 (87.83%). The
  product-based rule is marginally best for accuracy and cost (86.36%, not significant, see round 2). Rebalancing
  for per-team recall lowers both metrics. My expectation that it would help macro-F1 was wrong.

**Conclusion:** with this data the model is at its ceiling. The only way past ~86% is new information at intake:
the clarifying question for the vague 16%. It cannot be measured on history; it needs a pilot. On one channel, ask
it for half of the vague requests and not the other half, and compare "closed by first team". The gap to measure
is large (22% → likely 80%+), so a few hundred vague requests, a couple of weeks of volume, are enough.
