# Champion vs challengers: results

Protocol and decision rule: `docs/CHALLENGER_PROTOCOL.md` (committed before this ran). Rolling-origin backtest, 6,502 pooled predictions, each quarter scored by models trained only on earlier data. Truth = closing team.

Deviation: C5 uses scikit-learn HistGradientBoosting instead of LightGBM (LightGBM needs a system OpenMP library not present on this machine; same histogram-GBDT method).

| model | pooled accuracy | vs champion | wins / losses vs champion | McNemar p | Holm p | clear | vague | Oct-Dec 2025 | Jan-Mar 2026 | Apr-Jun 2026 | fit+predict s (3 folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 vague prior by product | 86.36% | +0.25 pts | 32 / 16 | 0.029 | 0.244 | 98.39% | 21.82% | 86.18% | 86.90% | 86.00% | 1 |
| C8 noise-cleaned TF-IDF+LR | 86.14% | +0.03 pts | 2 / 0 | 0.500 | 1.000 | 98.43% | 20.25% | 85.77% | 86.85% | 85.81% | 4 |
| C3 vague prior by product x warranty | 86.13% | +0.02 pts | 21 / 20 | 1.000 | 1.000 | 98.39% | 20.35% | 85.68% | 86.85% | 85.85% | 1 |
| **C0 champion** | 86.11% | +0.00 pts |  |  |  | 98.39% | 20.25% | 85.77% | 86.85% | 85.71% |  |
| C2 vague prior, last 6 months | 86.11% | +0.00 pts | 0 / 0 | 1.000 | 1.000 | 98.39% | 20.25% | 85.77% | 86.85% | 85.71% | 0 |
| C4b MIN_SUPPORT=10 | 86.11% | +0.00 pts | 0 / 0 | 1.000 | 1.000 | 98.39% | 20.25% | 85.77% | 86.85% | 85.71% | 2 |
| C7 transformer + vague override | 86.11% | +0.00 pts | 0 / 0 | 1.000 | 1.000 | 98.39% | 20.25% | 85.77% | 86.85% | 85.71% |  |
| C4a MIN_SUPPORT=1 | 86.05% | -0.06 pts | 0 / 4 | 0.125 | 0.875 | 98.32% | 20.25% | 85.68% | 86.76% | 85.71% | 2 |
| C10 majority vote (C0, C5, C9) | 86.05% | -0.06 pts | 0 / 4 | 0.125 | 0.875 | 98.32% | 20.25% | 85.77% | 86.67% | 85.71% |  |
| C6 fine-tuned transformer | 85.68% | -0.43 pts | 61 / 89 | 0.027 | 0.244 | 98.39% | 17.51% | 85.08% | 86.35% | 85.62% | 1799 |
| C5 gradient boosting (HGB) | 85.22% | -0.89 pts | 93 / 151 | 0.000 | 0.002 | 98.23% | 15.46% | 84.63% | 85.29% | 85.76% | 28 |
| C9 kNN on embeddings | 82.91% | -3.20 pts | 1 / 209 | 0.000 | 0.000 | 94.60% | 20.25% | 82.54% | 84.13% | 82.06% | 1 |

## Decision

No challenger beats the champion significantly (rule 2). 3 scored higher in the pooled backtest, none with Holm-corrected p < 0.05. **The champion stays.**
