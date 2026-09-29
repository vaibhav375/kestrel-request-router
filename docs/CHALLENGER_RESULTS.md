# Champion vs challengers: results and decision

Protocol: `docs/CHALLENGER_PROTOCOL.md`, committed (17e9436) **before** any challenger was run.
Raw tables: `out/challengers.md` (backtest), `out/challenger_robustness.md` (rule 3).
Reproduce: `python experiments/challengers.py` (~31 min, the transformer is most of it), then
`python experiments/challenger_robustness.py C1 C3 C8`.

## How much room was there?

An oracle that peeks at each test quarter's own outcomes and picks the majority team per phrasing reaches **86.25%**
pooled over the three quarters (86.79% if it also uses product for vague requests). The champion scores **86.11%**,
so the most any challenger could possibly gain was ~0.1–0.7 points.

## Rolling-origin backtest (6,502 predictions, each quarter scored by models trained only on earlier data)

| model | pooled accuracy | vs champion | wins / losses | Holm p | verdict |
|---|---|---|---|---|---|
| C1 vague guess by product (shrunk) | 86.36% | +0.25 | 32 / 16 | 0.244 | higher, **not significant** |
| C8 noise-cleaned TF-IDF + LR | 86.14% | +0.03 | 2 / 0 | 1.000 | tie |
| C3 vague guess by product × warranty | 86.13% | +0.02 | 21 / 20 | 1.000 | tie |
| **C0 champion (shipped router)** | **86.11%** | – | – | – | |
| C2 vague guess from last 6 months | 86.11% | 0.00 | 0 / 0 | 1.000 | identical (still Repairs) |
| C4b lookup threshold 10 | 86.11% | 0.00 | 0 / 0 | 1.000 | identical |
| C7 fine-tuned transformer + vague override | 86.11% | 0.00 | 0 / 0 | 1.000 | identical, ~350× slower to train |
| C4a lookup threshold 1 | 86.05% | −0.06 | 0 / 4 | 0.875 | slightly worse |
| C10 majority vote (C0, C5, C9) | 86.05% | −0.06 | 0 / 4 | 0.875 | slightly worse |
| C6 fine-tuned transformer (bge-small, 3 epochs) | 85.68% | −0.43 | 61 / 89 | 0.244 | worse on vague requests |
| C5 gradient boosting (HGB on TF-IDF-SVD + metadata + time) | 85.22% | −0.89 | 93 / 151 | 0.002 | **significantly worse** |
| C9 kNN on embeddings | 82.91% | −3.20 | 1 / 209 | <0.001 | **significantly worse** |

Per-quarter accuracies are in `out/challengers.md`. The champion's ranking is stable across all three quarters.

## Rule 3: 5-fold CV and held-out wording (for the challengers that scored higher)

| model | 5-fold CV | vs champion | held-out wording (fair rules) | vs champion |
|---|---|---|---|---|
| C0 champion | 86.10% | – | 77.10% | – |
| C1 | 86.25% | +0.15 | 77.23% | +0.13 |
| C3 | 86.13% | +0.03 | 77.08% | −0.02 |
| C8 | 86.14% | +0.04 | 51.40% | **−25.70** (no keyword fallback for new wording) |

## Decision: the champion stays

* **No challenger passes rule 2.** C1 is the only one ahead in all three schemes, but by 0.13–0.25 points. On
  6,502 paired predictions that is 32 wins against 16 losses: raw p = 0.029, Holm-corrected p = 0.244 across 11
  challengers. It is not strong enough evidence to change a model.
* **What C1 would be worth if real:** +0.2 points × ~8,655 requests a year × Rs 697 per misroute ≈ Rs 12,000 a
  year. Too small to justify the change on weak evidence. **Revisit it after the two-week side-by-side run**, when
  fresh data can confirm or kill it without re-using this history.
* **More complex models found nothing the lookup does not already capture.** On clear requests the fine-tuned
  transformer made exactly the champion's predictions (98.39%), at ~1,800 s of training over the three folds against ~5 s for the champion. Gradient
  boosting and kNN were significantly worse.
* **Where the remaining 14% error lives is unchanged:** vague requests (best any model managed: 21.8%) and noise in
  the recorded outcomes. Only a better intake (one clarifying question), not a better model, moves it.

## Signals ruled out before building challengers

* Repeat order/registration numbers: 126 linked request pairs, same closing team only 18% of the time.
* Vague requests vs hour of day, weekday, channel, warranty, source, phrasing: χ² p = 0.30–0.63 (no link).
  Product and quarter: p = 0.005 each; tested as C1 and C2 above.
* Product for clear requests: in 0 of 603 phrasing × product cells does the majority team differ.
