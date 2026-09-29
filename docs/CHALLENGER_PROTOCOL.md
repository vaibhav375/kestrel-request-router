# Champion vs challengers: test protocol (written before any challenger was run)

**Question:** is the shipped router the best we can get from this data, or does a reasonable alternative beat it?

## Champion

The shipped router (`kestrel/router.py`): last-request parsing → vague requests get the pooled best guess →
seen wording gets the majority closing team → new wording goes to keyword rules, then TF-IDF + LR.

## Challengers and why each is reasonable

| id | challenger | hypothesis it tests |
|---|---|---|
| C1 | Champion + vague guess conditioned on **product** (shrunk to the pooled guess) | product was the one metadata field with a significant link to vague outcomes (χ² p = 0.005) |
| C2 | Champion + vague guess from the **last 6 months only** | vague outcomes drift by quarter (χ² p = 0.005) |
| C3 | Champion + vague guess conditioned on **product × warranty** (shrunk) | a finer split might catch more |
| C4 | Champion with **MIN_SUPPORT = 1** and **= 10** (lookup threshold) | hyper-parameter check |
| C5 | **LightGBM** on TF-IDF (full + last request) + channel/product/warranty + hour/weekday | a flexible learner may find interactions the lookup cannot |
| C6 | **Fine-tuned transformer** (bge-small, all layers, 3 epochs, local) on the raw message | a sequence model reads word order and context natively |
| C7 | C6 with vague requests overridden by the pooled guess | the transformer may be better on clear requests only |
| C8 | **Noise-cleaned** TF-IDF + LR: drop training rows whose label gets < 10% out-of-fold probability, vague → pooled | ~2% of outcomes look like recording noise |
| C9 | **kNN** (k = 25) on bge-small embeddings, vague → pooled | nearest past requests rather than exact wording |
| C10 | **Majority vote** of champion, C5 and C9 | ensembles often beat single models |

Signals checked and ruled out *before* building challengers (no challenger built for them):
repeat order/registration numbers (linked requests closed in the same team only 18% of the time: no signal);
hour of day, weekday, channel, warranty and phrasing for vague requests (all χ² p > 0.3); product for clear requests
(for 0 of 603 phrasing × product cells does the majority differ from the phrasing's majority).

## Validation

* **Primary: rolling-origin backtest.** Three quarterly folds, each trained only on everything before it:
  Oct–Dec 2025, Jan–Mar 2026 and Apr–Jun 2026 (≈ 6,500 predictions in total). This is exactly how the router will be
  used: trained on the past, scoring the next quarter.
* **Secondary:** 5-fold stratified CV on all 10,822 requests.
* **Robustness:** held-out wording (GroupKFold by deciding phrase) for any challenger that beats the champion.
* **Metric:** accuracy against the team that closed the request (as argued in the form: every misroute costs the same).

## Decision rule (fixed in advance)

A challenger replaces the champion only if **all** hold:
1. Higher pooled backtest accuracy than the champion.
2. The difference is significant: paired McNemar exact test on the pooled backtest predictions, p < 0.05 after
   Holm correction for the number of challengers.
3. Not more than 1 point worse than the champion in 5-fold CV and on held-out wording.
4. It meets the product constraints: starts on a clean machine from `requirements.txt`, no paid API, < 50 ms per request
   on a laptop CPU.

Otherwise the champion stays. On a tie the simpler model wins.
