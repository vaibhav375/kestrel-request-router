# Round 4 results: A1, B1–B5, C1–C2

Protocol: `docs/ROUND4_PROTOCOL.md`, committed (233048b) with the Hinglish dictionary **before** the Hinglish test set
was generated or any test was run. Raw tables: `out/round4.md`. Reproduce: `python experiments/generate_synthetic.py
paraphrases|hinglish`, then `python experiments/round4.py` (~3 min).

## Short comparison

| id | idea | target test | result vs round-3 router | decision |
|---|---|---|---|---|
| A1 | Vague requests: best guess by product | known wording (E1) | **+0.25 pts** (86.11% → 86.36%, p = 0.029 raw; not significant after correction in round 2) | **Adopted by client decision, shown to agents as a guess** |
| B1 | Fuzzy match to nearest known phrasing | new wording (E2) | +0.00 (never changed an answer on its own); did help typos (+1.2) | Not adopted: failed its target test; typos are covered better by B4 |
| B2 | Vote: rules + nearest phrasing + text model | new wording (E2) | **−10.1 pts**: the two word-overlap checks agree on the wrong team and outvote the rules | Rejected |
| B2e | Same vote with sentence embeddings | new wording (E2) | +2.36 pts, significant | Not adopted: below the 3-point bar for adding PyTorch (~2 GB) |
| B3 | Paraphrase lookup (666 LLM paraphrases) | new wording (E2) | +0.00 (generated sentences never match new wording exactly) | Not adopted |
| B4 | Hinglish dictionary + spelling repair | typos (E3), Hinglish (E4) | **Heavy typos 94.05% → 97.61% (+3.56, p < 0.001); light 96.66% → 97.72%; Hinglish 78.7% → 82.0% (+3.33, p = 0.006)**; no change on known wording | **Adopted** |
| B5 | Corrections loop (log unsure requests, agents confirm, retrain) | learning over time (E5) | New phrasing: **89.8% → 95.2% after 1 confirmed example → 98.4% after 3** (= known-wording level) | **Adopted**: `/feedback` endpoint, review queue, `train.py` picks up `data/corrections.csv` |
| C1 | `auto_route` flag on every answer | reporting | 84.3% of requests auto-routed at 98.4% accuracy | **Adopted** |
| C2 | Clarifying options ordered by product likelihood | first 3 options (E6) | 50.0% vs **52.5% for today's fixed order** | Rejected: the fixed order is better |

## What the final router scores (evaluate.py, out-of-time Apr–Jun 2026)

| | round-3 router | final router (A1 + B4, C1 flag, B5 loop) | bot |
|---|---|---|---|
| Accuracy vs closing team | 85.7% | **86.0%** (CI 84.5–87.5%) | 76.7% |
| 5-fold CV | 86.1% ± 0.3 | **86.3% ± 0.2** | 77.2% |
| Clear requests | 98.0% | 98.0% | 87.8% |
| Vague requests | 20.2% | **22.0%** | 19.3% |
| Macro-F1 | 87.5% | 87.2% (A1 trades 0.3 macro-F1 for accuracy) | 78.0% |
| Heavy typos (synthetic) | 94.1% | **97.6%** | – |
| Hinglish (synthetic, noisy) | 78.7% | **82.0%** | – |
| Saving per year | Rs 8.7 lakh | **Rs 8.8 lakh** | – |
| Test predictions changed | – | 49 of 2,178 (vague water-purifier requests: Repairs → Warranty Claims, A1) | – |
| Worst-case time per request | < 1 ms | 3.7 ms | – |

## Honest notes

* **A1 is a guess.** Its gain was not significant after correcting for multiple tests. Agents see: "This is a guess:
  it was not proven better than the overall most common team." Macro-F1 drops 0.3 points; accuracy and cost improve.
* **E3 and E4 are synthetic.** The typos are random edits. The Hinglish was rewritten by a 3B local model and is
  noisier than real Hinglish: about half contains Hindi words, many lines are heavy SMS-speak, and some lost their
  meaning. Treat these as stress tests, not live accuracy.
* **B1 and B3 do nothing on their own for new wording** under the strict test. The earlier "new wording" number was
  flattered by phrasings that differ only in a number ("after 7 months" vs "after 8 months"). Under the strict
  grouping the round-3 router scores 89.8% on clear requests with new wording.
* **The one clear improvement for new wording is embeddings (+2.4 points).** It was not adopted because the
  protocol required ≥ 3 points to justify adding PyTorch to the service. It is the first thing to revisit if live new
  wording turns out to be common, which the B5 review queue will show.
