# Round 4 results

Protocol: `docs/ROUND4_PROTOCOL.md` (committed before these tests). Baseline R3 = the router after round 3. p = paired McNemar vs R3 on the same requests.

## E1 Known wording: rolling backtest, 6,502 predictions

| variant | accuracy | vs R3 | wins/losses | p |
|---|---|---|---|---|
| R3 (round-3 router) | 86.11% | +0.00 | 0/0 | 1.000 |
| A1 vague guess by product | 86.36% | +0.25 | 32/16 | 0.029 |
| B1 fuzzy match | 86.11% | +0.00 | 0/0 | 1.000 |
| B2 combined fallback | 86.11% | +0.00 | 0/0 | 1.000 |
| B3 paraphrase lookup | 86.11% | +0.00 | 0/0 | 1.000 |
| B4 Hinglish + spelling repair | 86.11% | +0.00 | 0/0 | 1.000 |
| B1+B2+B3+B4 | 86.11% | +0.00 | 0/0 | 1.000 |
| ALL (A1 B1 B2 B3 B4 C2) | 86.36% | +0.25 | 32/16 | 0.029 |

**C1 (auto-route flag), ALL variant:** auto-routes 84.3% of requests at 98.43% accuracy; the other 15.7% get the clarifying question.

## E2 New wording (strict: phrasings differing only in a number are one group; teams.csv-only rules)

9,095 clear requests, every one scored by a router that never saw its phrasing.

| variant | clear requests right | vs R3 | wins/losses | p |
|---|---|---|---|---|
| R3 (round-3 router) | 89.79% | +0.00 | 0/0 | 1 |
| A1 vague guess by product | 89.79% | +0.00 | 0/0 | 1 |
| B1 fuzzy match | 89.79% | +0.00 | 0/0 | 1 |
| B2 combined fallback | 79.69% | -10.09 | 220/1138 | 2.08e-149 |
| B3 paraphrase lookup | 89.79% | +0.00 | 0/0 | 1 |
| B4 Hinglish + spelling repair | 89.79% | +0.00 | 0/0 | 1 |
| B1+B2+B3+B4 | 89.49% | -0.30 | 391/418 | 0.361 |
| ALL (A1 B1 B2 B3 B4 C2) | 89.49% | -0.30 | 391/418 | 0.361 |
| B2 with embeddings (comparison) | 92.15% | +2.36 | 391/176 | 9.17e-20 |

Which step decided, B1+B2+B3+B4 stack (E2):

| step | share | right |
|---|---|---|
| vote | 49.3% | 91.0% |
| vote_split | 27.4% | 84.5% |
| similar | 23.2% | 92.1% |

## E3 Typos (synthetic): light = 15% of words corrupted, heavy = 30%

| variant | clean | light | heavy | heavy vs R3 |
|---|---|---|---|---|
| R3 (round-3 router) | 97.94% | 96.66% | 94.05% | +0.00 (p 1) |
| A1 vague guess by product | 97.94% | 96.66% | 94.05% | +0.00 (p 1) |
| B1 fuzzy match | 97.94% | 96.94% | 95.28% | +1.22 (p 0.00319) |
| B2 combined fallback | 97.94% | 97.05% | 96.28% | +2.22 (p 8.58e-06) |
| B3 paraphrase lookup | 97.94% | 96.66% | 94.05% | +0.00 (p 1) |
| B4 Hinglish + spelling repair | 97.94% | 97.72% | 97.61% | +3.56 (p 1.82e-18) |
| B1+B2+B3+B4 | 97.94% | 97.50% | 97.17% | +3.11 (p 2.26e-10) |
| ALL (A1 B1 B2 B3 B4 C2) | 97.94% | 97.50% | 97.17% | +3.11 (p 2.26e-10) |

Example heavy-typo inputs: “need replacement filter for induction cooktop [no.]” · “namaste, please boo demo ad installation fmr purifier kindly resolve” · “sir aant to return thee air fryer kindly resolve”

## E4 Hinglish (synthetic, LLM-rewritten, 300 requests)

| variant | original English | Hinglish | Hinglish vs R3 |
|---|---|---|---|
| R3 (round-3 router) | 99.00% | 78.67% | +0.00 (p 1) |
| A1 vague guess by product | 99.00% | 78.67% | +0.00 (p 1) |
| B1 fuzzy match | 99.00% | 80.67% | +2.00 (p 0.146) |
| B2 combined fallback | 99.00% | 83.00% | +4.33 (p 0.0596) |
| B3 paraphrase lookup | 99.00% | 78.67% | +0.00 (p 1) |
| B4 Hinglish + spelling repair | 99.00% | 82.00% | +3.33 (p 0.00635) |
| B1+B2+B3+B4 | 99.00% | 80.67% | +2.00 (p 0.488) |
| ALL (A1 B1 B2 B3 B4 C2) | 99.00% | 80.67% | +2.00 (p 0.488) |

Example Hinglish inputs: “hi, purifier delivered bt n instaled ynt asap” · “kya mere apna vacuum spare jar buy karne ke liye kahan hai sir? aap mujhe to resolve do pls” · “extend warranty purifier, wo matra certificate naa chahiye naa warranty certificate naa extend kijiye purifier thanks”

## E5 Corrections loop (B5): accuracy on a new phrasing after k agent-confirmed examples are added

| confirmed examples added per new phrasing | accuracy on its later requests |
|---|---|
| 0 | 89.77% |
| 1 | 95.22% |
| 3 | 98.40% |
| 5 | 98.40% |
| 10 | 98.40% |

Reference: accuracy on known wording for requests that state a need = 98.39%; the protocol's bar is 95% of that = 93.47%.

## E6 Clarifying question: is the right team among the first 3 options? (vague requests, backtest)

| option order | true team in first 3 |
|---|---|
| fixed order (today) | 52.45% (n=1,022) |
| pooled likelihood order | 50.39% (n=1,022) |
| product likelihood order (C2) | 50.00% (n=1,022) |

