# Screen recording script (max 3 minutes, no slides)

Before recording: `source .venv/bin/activate && python train.py && uvicorn app.server:app --port 8000`,
then open `http://localhost:8000`, `out/evaluation.md`, `out/experiments.md` and `out/hybrid.md` in tabs.

| time | on screen | say |
|---|---|---|
| 0:00–0:20 | `data/email-thread.txt`, then the first table in `out/evaluation.md` | "Ritu asked for 90% match with the bot's labels. First thing I did was join the resolution log: the bot's label is the team that closed the request only 77% of the time. So the labels aren't ground truth, they're the bot's guesses." |
| 0:20–0:45 | same table: copy-the-bot row | **What I tried first.** "I built what was asked: a model trained on the labels. 92% agreement, target hit, and it's right 77% of the time, same as the bot. So I threw that away and trained on where requests were actually closed." |
| 0:45–1:15 | `out/experiments.md` section A | **What I tried.** "Then twenty variants, on three tests: out-of-time, cross-validation, and wording it has never seen. TF-IDF models, embeddings, rules, a lookup of past outcomes, a local LLM. Plain bag-of-words got stuck at 83%, because when a customer asks two things the *last* one decides, 99% of the time, and bag-of-words can't see order. Adding the last request as its own input fixed that." |
| 1:15–1:40 | `out/evaluation.md` "Where it is right and wrong" | **What I changed.** "Reading the errors found a parsing bug: 'please call back regarding my fan' was being treated as a sign-off and sent to Installs. And 16% of requests say nothing routable. Those ended up in all seven teams, so no model can place them. I stopped trying to guess those cleverly: best guess plus one clarifying question." |
| 1:40–2:20 | the screen: click *Mentions payment* → Route; then *Vague* → Route | "Here's the service. A customer who says 'paid on UPI' goes to Repairs, not Billing, with the reason: the bot sent 525 of 626 of these to Billing and Billing closed one. A vague request gets low confidence and the seven-option question." |
| 2:20–2:45 | `out/llm_benchmark.md`, then `out/hybrid.md` section C | **What I threw away.** "A local LLM: 71%, worse than the bot, three seconds a request. Embeddings: same accuracy, a hundred times slower, two gigabytes of install. And my own 98% robustness number: my rules were written after reading every phrasing, so I report the fair version, 87%." |
| 2:45–3:00 | `out/evaluation.md` section 7 | "Result: 86% vs the bot's 77%, about Rs 8.8 lakh a year including the licence, Rs 0 per request. Next week: two weeks side-by-side, then switch off." |
