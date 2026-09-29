# Kestrel Home: service-request router

Routes each new service request to the team that will actually close it, and gives the reasons an agent can read.
It replaces the vendor routing bot (Rs 3.2 lakh a year).

**Headline (out-of-time test, Apr–Jun 2026, 2,135 requests the model never saw):** the router sends **85.7%** of
requests to the team that closed them; the bot managed **76.7%**. On requests that state a need it gets 98.0% right.
The other ~16% say only "please call me about my purifier". No text router can place those, so the router flags
them and suggests one clarifying question. Worth about **Rs 8.7 lakh a year**; costs **Rs 0 per request**.

## Run it

Needs Python 3.10+ (tested on 3.12, macOS M1, 8 GB). No API key, no paid calls, no GPU.

```bash
cd kestrel-routing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy the data pack into `data/`. The files can keep their downloaded `<uuid>-train.csv` names. The pack is
confidential (policy §10) and is **not** in this repo.

```
data/train.csv  data/test_unlabelled.csv  data/resolution_log.csv  data/teams.csv  data/sample_submission.csv
```

```bash
python train.py                         # trains in ~2 s -> models/router.joblib + predictions.csv
uvicorn app.server:app --port 8000      # then open http://localhost:8000
```

The very first start in a fresh virtualenv can take ~30 s while Python compiles pandas/scikit-learn; after that it
starts in a second or two. If you start the service before training, it trains itself from `data/` on startup. If `data/` is empty too, it
still starts: the screen loads, `/health` says `not_ready`, and `/route` answers 503 with what to do.

### The endpoint

```bash
curl -s -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"request_id":"SR510822","request_text":"water purifier not turning on paid on upi","channel":"chat"}'
```

Returns `team`, `confidence` / `confidence_level`, `needs_clarification` (+ the question and its seven
options), `alternatives`, `route_type` (`seen` / `vague` / `new_wording`) and `reasons`, for example:

```
"team": "Repairs", "confidence_level": "high",
"reasons": ["Customer asks: “water purifier not turning on”.",
            "221 past requests with this wording: >99% were closed by Repairs.",
            "Mentions “paid on upi”. Saying they paid does not make it a Billing request (ops policy §3). The old bot sent 525 of 626 such past requests to Billing; Billing closed 1 of them.",
            "Repairs handles: Product faults, breakdowns, error codes, noise, leaks - anything that needs a technician to fix."]
```

Only `request_text` is required. `channel`, `product_family` and `warranty_status` are accepted and validated but
do not change the decision: they were tested and added nothing. Interactive API docs are at `/docs`.

## Evidence

```bash
python evaluate.py                      # -> out/evaluation.md (~1 min)
pip install -r requirements-dev.txt && python -m pytest    # 32 tests, ~3 s
```

| file | what it shows |
|---|---|
| `out/evaluation.md` | out-of-time test with CIs, the copy-the-bot comparison, per-team precision/recall, confusion matrix, confidence calibration, 5-fold CV, new-wording test, every error type, expected test score, rupees, team volumes |
| `out/experiments.md`, `out/hybrid.md` | the 20 approaches compared on three validation schemes |
| `out/llm_benchmark.md` | a free local LLM (Qwen2.5-3B) tried on 300 requests |
| `out/errors_out_of_time.csv` | every request it got wrong in the out-of-time test |
| `docs/FINDINGS.md` | what the data showed, what I tried, changed and threw away |
| `docs/memo.md` | one-page memo to Ritu |
| `docs/submission-form.md` | the submission form |

Re-running the experiments needs `pip install -r requirements-experiments.txt` (sentence-transformers; `mlx-lm`
for the LLM benchmark on Apple Silicon). Neither is needed by the service.

## How it works

```
request_text ─► kestrel/data.py    fix legacy-Zoho mojibake; team renames merged (policy §5)
             ─► kestrel/text.py    strip greeting / order no. / sign-off / payment remark; split into requests
             ─► kestrel/router.py  the LAST request decides (98.7% of 549 two-request messages closed that way)
                 ├─ it states no need ("please call back regarding …")  -> best guess Repairs (22%), flag + 1 question
                 ├─ wording seen ≥3 times before                         -> team that closed most of those requests
                 └─ new wording                                          -> keyword rules (teams.csv + history), else TF-IDF + LR
```

Trained on `resolution_log.final_team` (who closed it), **not** on `team_label` (where the bot sent it). A model
trained on `team_label` matches it 91.8%, passing the brief's 90% bar, yet routes only 77.0% correctly.

Decisions I made where the brief was unclear:
* **Team names in `predictions.csv`** use the current names, *Installs & Demo* and *Filters & Consumables*. The test
  period is Jul–Sep 2026, after the 15 Jan 2026 rename, and the history uses only the new names after that date.
* **Truth** = the team that closed the request, since the client scores against "outcomes".
* **No LLM in the product.** A local 3B model scored 71% (worse than the bot) at ~3 s a request. A paid API would add
  the per-request bill Finance ruled out, and it cannot fix vague requests either, because the information is not
  in the message.

## Layout

```
kestrel/        data.py (loading + fixes) · text.py (parsing) · router.py (model + reasons) · business.py (rupees)
app/            server.py (FastAPI) · static/index.html (the screen)
train.py        trains, writes predictions.csv (checked against sample_submission.csv)
evaluate.py     writes out/evaluation.md
experiments/    compare_models.py · hybrid.py · llm_benchmark.py
tests/          32 tests: parsing, routing rules, API contract + failure modes, real-pack checks (skipped without data)
docs/           FINDINGS.md · memo.md · submission-form.md · RECORDING.md
```
