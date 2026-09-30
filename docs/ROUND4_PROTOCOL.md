# Round 4 protocol: A1, B1–B5, C1–C2 (written and committed before any result)

Baseline = the router after round 3 ("R3"). Every component is a switch on the router, so each is measured
alone against R3, and all together.

## Components

| id | component | what it changes |
|---|---|---|
| A1 | Vague requests: best guess = most common closing team **for that product** (shrunk to the pooled guess, α = 50) | vague requests only. **Adopted by client decision regardless of significance**, and shown to agents as a guess |
| B1 | Fuzzy match: new wording → nearest known phrasing by character-n-gram similarity, if cosine ≥ 0.60 | new wording |
| B2 | Combined fallback: vote of keyword rules, nearest known phrasing (any similarity) and TF-IDF + LR | new wording that B1 does not settle |
| B3 | Paraphrase lookup: ~10 paraphrases per known phrasing generated **offline** by the free local LLM (Qwen2.5-3B), each mapped to its source phrasing's team | new wording |
| B4 | Repair: Hinglish → English phrase dictionary + spelling correction of unknown words against the training vocabulary; re-parse and re-check | new wording, typos, Hinglish |
| B5 | Corrections loop: requests the router was unsure about are logged; agent-confirmed teams are added at the next retrain | learning over time |
| C1 | `auto_route` flag (confidence ≥ 0.70 and no clarification needed) on every answer; reported as coverage + accuracy | reporting |
| C2 | Clarifying-question options ordered by likelihood for the product; top 3 marked | vague requests UX |

## Tests

| test | data | used for |
|---|---|---|
| E1 known wording | rolling-origin backtest, 3 quarters, 6,502 predictions (as round 2) | no-regression check for every component; A1's effect |
| E2 new wording (strict) | GroupKFold(5) by deciding phrase **with digits removed**, so "stopped working after 7 months" and "…8 months" are one group; keyword rules = teams.csv-only set; B3 paraphrases used only for phrasings in the training fold | B1, B2, B3 |
| E3 typos (synthetic) | Apr–Jun 2026 clear requests, router trained to Mar 2026; each word corrupted with probability 0.15 (light) and 0.30 (heavy) by a random deletion, swap, substitution or duplication, seed 0 | B4 |
| E4 Hinglish (synthetic) | 300 random Apr–Jun 2026 clear requests rewritten in Hinglish by the local LLM. **Generated after the B4 dictionary is committed**, so the dictionary cannot be tuned to the test | B4 |
| E5 corrections | E2 folds; for each new phrasing, add its first k = 0, 1, 3, 5, 10 agent-confirmed examples, retrain, score the later ones | B5 |
| E6 option order | vague requests in E1: share whose true team is in the first 3 options, pooled order vs product order | C2 |

E3 and E4 are synthetic and will be reported as such.

## Decision rules

* **B1–B4:** adopt a component if, on its target test, it improves accuracy on clear requests by ≥ 1.0 point with
  paired McNemar p < 0.05 vs R3; costs ≤ 0.1 point on E1; and keeps the service light (pip-installable from
  `requirements.txt` without PyTorch, < 50 ms per request, no paid API). An embedding-based variant of B2 is measured
  for comparison and adopted only if ≥ 3 points better than the best light variant.
* **B5:** adopt if a new phrasing reaches ≥ 95% of seen-wording accuracy after ≤ 5 corrected examples.
* **C1:** adopt (reporting only; cannot change predictions).
* **C2:** adopt if product ordering puts the true team in the first 3 at least as often as pooled ordering.
* **A1:** adopted by decision; E1 reports its measured effect.
