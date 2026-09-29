# Submission form: Kestrel Home, service-request routing (Variant B)

### What did you build, and what business decision does it support? *State the number and the rupees.*

A request router: one endpoint and one screen. It reads a customer's opening message and sends it to the team that
will actually **close** it, with reasons an agent can read. It is trained on where requests were closed
(`resolution_log.final_team`), not on where the vendor bot sent them (`team_label`).

**Decision it supports: switch the Rs 3.2 lakh/year routing bot off.**

* **Number:** 85.7% of requests reach the closing team first, against the bot's 76.7%. That is an out-of-time test
  on 2,135 Apr–Jun 2026 requests the model never saw. When the customer states a need, it is right 98.0% of the time.
* **Rupees:** misrouting costs about **Rs 14.7 lakh a year** today (3,129 transfers × Rs 305, plus Rs 260 extra
  contact per misroute, policy §4). The router cuts wrong-first-team from 23% to 14%: about **Rs 5.5 lakh** saved.
  With the **Rs 3.2 lakh** licence, the total is **about Rs 8.7 lakh a year**, at Rs 0 per request. If one intake
  question for vague requests works, up to Rs 16 lakh; that is a projection that needs a pilot.
* **Second decision it supports (headcount, which Ritu raised):** real monthly volumes per team. The bot's labels
  over-state Repairs (+42/month), Billing (+24) and Consumables (+23), and under-state Installs (−27),
  Returns (−25) and Warranty (−22).

### What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? *Say how you estimated it. We compare this with the real score.*

**Accuracy (share of requests whose predicted team equals the team that closed it): about 86%, range 84–87%.**
If you score macro-F1 instead: about 87%. If the hidden outcome is the bot's own label rather than the closing
team: about 88%.

**Why accuracy.** Every misroute costs Kestrel the same regardless of which team it lands in (Rs 305 per transfer
+ Rs 260 extra contact), so the share routed right converts directly into rupees. It is also the "match rate" Ritu
asked for, measured against the right target. Macro-F1 would weight the small Filters & Consumables queue as much as
Repairs, which is not how the cost works.

**How I estimated it:**
1. **Out-of-time:** trained on Apr 2025 – Mar 2026, scored on Apr – Jun 2026 (2,135 requests, all CRM, the same
   source as the test set). Accuracy **85.7%** (bootstrap 95% CI 84.3–87.2%), macro-F1 87.5%.
2. **5-fold cross-validation** on all 10,822 requests: **86.1% ± 0.3%**.
3. **Mix-weighted:** the router tags each test request by type. 83.7% use wording seen before (98.3% right in CV),
   15.7% state no need (21.7%), 0.5% use new wording (87.3% on the fair held-out-wording test). Weighted: **86.2%**.
4. **Why not higher:** ~16% of requests say only "please call me about my X", and past ones were closed by all seven
   teams (best guess, Repairs, is right 22% of the time). About 2% of clear requests were closed by an unrelated team,
   which looks like noise in the log. Together these cap any text-only router at ~86–87%.

**Team names:** I used *Installs & Demo* and *Filters & Consumables*. Every test request is after the 15 Jan 2026
rename, and the history uses only the new names after that date. If the scorer uses the old names, those two teams
would score zero, so please map them.

### How do you know it works? *How you validated, on what split, error rate, and the kind of case it gets wrong.*

Truth = the team that closed the request. All figures come from `out/evaluation.md` (`python evaluate.py`):

| check | split | router | bot |
|---|---|---|---|
| Out-of-time | train to 31 Mar 2026 / score Apr–Jun 2026 (2,135) | **85.7%** | 76.7% |
| Model trained to copy the bot's labels | same | 77.0% (but **91.8%** agreement with the labels) | – |
| 5-fold CV | all 10,822, stratified | 86.1% ± 0.3% | 77.2% |
| New wording | GroupKFold by phrasing (every phrasing unseen); keyword rules from teams.csv only | 87.3% on stated needs (folds 70–98%) | – |
| Local LLM, same 300 requests | out-of-time sample | 84.3% vs LLM 71.3% | 74.0% |
| **11 challengers, pre-registered** | rolling-origin backtest, 3 quarters, 6,502 predictions | **86.11%; none significantly better** (best: +0.25 pts, Holm p = 0.24) | – |

* **Is it the best we can get?** I ran a second round against 11 alternatives under a protocol committed before
  running them (`docs/CHALLENGER_PROTOCOL.md` → `docs/CHALLENGER_RESULTS.md`). These included gradient boosting,
  a fine-tuned transformer, kNN, noise-cleaned training, metadata-based guesses for vague requests and an ensemble.
  None beat the router after correcting for multiple tests. An oracle that sees the answers reaches only 86.25% on
  the same quarters (router 86.11%).
* **Confidence is honest:** "high" = 83.6% of requests, 98.0% right. "Low" = 15.8%, 20.2% right, almost all vague.
* **32 automated tests:** parsing, the policy §3 payment rule, last-request rule, vague flag, API contract, bad
  input (422), no model / no data (service starts, 503 with instructions), predictions file matches the sample, and
  an accuracy regression check on the real pack. Also run on a clean copy with a fresh virtualenv.

**Error rate: 14.3% (305 of 2,135).** What it gets wrong:
* **268 vague requests** ("please call back regarding my purifier", "issue with fan"). Nobody can route these from
  the text; the router flags them and offers one clarifying question.
* **37 clear requests (2.1%)** closed by an unrelated team, e.g. "how to clean air fryer" closed by Billing, often
  with 0 transfers. These look like recording noise, not routing errors.
* **On live traffic:** wording customers have never used, ~87% on stated needs, with wide variation. It is flagged
  when keywords conflict or none match.

### Did you change, narrow, or push back on the client's ask? *What, when, and why.*

Yes, in the first hour, as soon as I joined the resolution log to the labels:
* **Pushed back on "match the labels at 90%".** `team_label` is the bot's decision, and it matches the closing team
  only 77% of the time. I built the literal ask as a test: 91.8% agreement with the labels, and 77.0% right, which
  is no better than the bot. So I trained and scored on the closing team, and I report both numbers.
* **Said plainly that 90% against real outcomes is unreachable from the message alone**, because ~16% of requests
  contain no routable need. I proposed the fix (one intake question with seven options, built into the screen)
  instead of pretending a model can do it.
* **Answered the headcount question** Ritu raised in passing, using closing teams rather than the bot's labels.
* **Recommended a two-week side-by-side run** before the switch-off, rather than an immediate cut-over.

### What is wrong with what you are handing us, or with the data we handed you? *Be specific: bugs, shortcuts, columns you did not trust, rows that looked wrong.*

**The data:**
* **`team_label` is not ground truth.** It is the bot's guess: 77% right overall, 96% for Advice/Installs/Returns/
  Warranty, but 65% for Billing, 60% for Consumables and 64% for Repairs. I did not train on it.
* **`resolved_at` for legacy Zoho rows is UTC**, as policy §9 says. 26% of legacy rows were "resolved before they
  were created", and the legacy median was 4.0 h vs 9.5 h on the CRM. Fixed with +5:30; now 0 negative and both
  medians 9.5 h.
* **Legacy Zoho text is garbled** in 11% of rows ("urgÃ©nt", "Ã¢â‚¬Â¦"). I reversed the double encoding and
  none is left.
* **126 rows record 0 transfers but closed in a different team than they opened in.** They are internally
  inconsistent; kept, since dropping them changed nothing.
* **`final_team` itself is noisy:** ~2% of clear requests closed by an unrelated team ("is air fryer safe for
  kids" → Billing). This caps what any model can score.
* **Two team names change mid-file** (15 Jan 2026). I merged them; predictions use the new names.
* **"Eighteen months"** is 15 months of labelled history (Apr 2025 – Jun 2026) plus the 3 unlabelled test months.
* **`channel`, `product_family` and `warranty_status` carry no routing signal** beyond the text (+0.0 to +0.6
  points, within noise).

**What I'm handing over:**
* **Tested on history, not live traffic.** The export is highly templated: 132 distinct deciding phrases. Real
  wording will be messier; the fair estimate for new wording is ~87% on stated needs, varying 70–98% by fold.
* **The shipped keyword rules for new wording were written after reading the data.** On the held-out-wording test
  they score 98%, which I do not count; I report the teams.csv-only rule set (87%) instead.
* **Vague requests are guessed as Repairs (22% right).** It is the best available, but it is still a guess. The
  98% "with clarification" figure is a projection that assumes customers answer the question correctly.
* **Rupee savings** use the historical average cost of a misroute (Rs 697) and the last 12 months' volume, and
  assume the router's remaining errors cost the same as the bot's.
* **The screen has no login and the API has no rate limiting.** It is a working demonstrator for a desk, not
  internet-hardened. Tested on macOS (M1, Python 3.12) only.

### What did you deliberately leave out, and why that rather than something else?

* **An LLM in the routing path.** Measured: a free local Qwen2.5-3B scored 71% zero-shot and 68% with hints, worse
  than the bot, at ~3 s per request. A paid API would add the per-request bill Farhan ruled out, and no model can
  route a message that contains no need.
* **Sentence embeddings / PyTorch.** Same accuracy on known wording (85.7% vs 85.8%), ~100× slower, and ~2 GB of
  install for a service that must start on a clean machine.
* **Metadata features, and the product-based guess for vague requests (C1).** Tested; C1 was +0.25 points but not
  significant (Holm p = 0.24), worth ~Rs 12,000/year if real, so it is left for re-testing on fresh data. A
  fine-tuned transformer and gradient boosting were also left out: no better, or worse, and far heavier.
* **Live IVR/CRM integration.** Needs Tanmay, and should follow the side-by-side run.
* **SLA and resolution-time modelling.** One finding was enough for the decision: misrouted requests take twice as
  long to close (16.6 h vs 8.2 h median).

I chose these because none of them changes the switch-off decision, and each would add cost or complexity to a
service whose value is being free and simple.

### Anything you built or found that nobody asked for?

* **The payment trap, quantified:** of 626 customers who mentioned paying ("paid on upi", "already paid in full"),
  the bot sent 525 to Billing, and Billing closed **1**. This confirms Meenal's complaint and policy §3.
* **Two requests in one message: the last one decides** (542 of 549 messages, 98.7%). The router tells the agent
  about the earlier request.
* **Misrouted requests take twice as long to close** (median 16.6 h vs 8.2 h), once the UTC fix is applied.
* **A seven-option clarifying question** in the screen for vague requests, each answer mapped to a team.
* **Real team volumes for headcount planning** (`out/team_volumes.csv`).

### What did you use AI for? *Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.*

**Tools:** Claude Code (desktop app, model Claude Opus 5.5) for data profiling, writing the Python, running the
experiments and drafting the docs. Benchmarked locally for free: Qwen2.5-3B-Instruct (4-bit, MLX) and
bge-small-en-v1.5 embeddings. **No paid API calls** in the product or the experiments; no data left the laptop
(policy §10). **Cost:** [FILL IN: your Claude plan / usage]. Local models: Rs 0.

**Where it helped:** joining the resolution log early and showing that the labels were the bot's guesses; finding
the template structure, the last-request rule and the unroutable 16%; running ~20 model variants on three
validation schemes quickly.

**Where it went wrong, and was caught:**
* The first parser treated "please call back" as a sign-off, so "please call back regarding my fan" became
  "regarding fan" and was routed to Installs. Caught by reading the errors; fixed; regression test added.
* The first "vague" definition looked at the whole message and missed 120 "specific, then vague" messages.
* The first held-out-wording score (98%) was inflated by rules written after reading every phrasing. I replaced it
  with a fair teams.csv-only check (87%).
* Several numbers in the first docs draft came from memory ("204 distinct phrases", actually 132; "+2 points",
  actually +1.8). Every figure was then re-checked against script output.
* The local LLM benchmark took ~15 minutes per prompt set on an 8 GB laptop, and it lost.

**Thrown away:** training on `team_label`; sentence embeddings; an LLM in the path; metadata features; dropping
inconsistent rows; per-phrase guesses for vague requests (they fit noise). Details in `docs/FINDINGS.md`.

**Screen recording:** [FILL IN: link]

### Your Public Google Drive Link

[FILL IN]

### Someone picks this up on Monday and you are unreachable. *The three things they need to know.*

1. **Put the pack in `data/`, then run `python train.py` and `uvicorn app.server:app`.** `python evaluate.py`
   regenerates every number in the memo and this form. Cost rates are in `kestrel/business.py` (policy §4).
2. **The truth is the closing team in the resolution log, never `team_label`.** Retrain monthly on newly closed
   requests. Watch the `new_wording` and `vague` shares in `out/test_predictions_detailed.csv`: if new wording
   climbs above a few percent, add phrasings or rules in `kestrel/router.py`.
3. **Two client actions are open:** (a) Tanmay runs the two-week side-by-side check and reports "closed by first
   team" weekly; (b) Meenal pilots the intake question on one channel. The Rs 16 lakh upside depends on (b).

### Honest hours spent. *One number.*

[FILL IN]

### Github Repo Link

https://github.com/vaibhav375/kestrel-request-router

The repo holds code, docs and aggregate results only. The data pack, the trained model and every per-request CSV
are excluded (`.gitignore`), because policy §10 forbids uploading customer data to public repositories.

### What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? *Show the arithmetic. If you used no paid calls, say so.*

**Rs 0 per prediction and Rs 0 a month in usage fees. No paid calls, no API key, no GPU.**

* One prediction takes 0.16 ms of CPU inside the model (1.5 ms median, 2.7 ms p95 for a full HTTP call, measured over
  200 calls on an M1 laptop). Training on all 10,822 requests takes ~2 s; the model file is 176 KB.
* A month at Kestrel's volume: 700 × 1.5 ms ≈ **1 second of CPU a month**. Retraining monthly adds ~2 s. Both fit
  on any server Kestrel already runs, so the marginal cost is Rs 0 (electricity only).
* There is no per-request charge, so the cost does not grow with volume (Farhan's condition). For comparison, the
  bot costs Rs 3.2 lakh a year ≈ Rs 26,700 a month ≈ **Rs 38 per request** at 700 a month.
