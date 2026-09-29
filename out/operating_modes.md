# Round 3: operating levers (not model changes)

Rolling-origin backtest, 6,502 predictions over Oct 2025 – Jun 2026, router trained only on earlier data.
Router accuracy on this backtest: **86.11%**.

## A. Auto-route only when confident; send the rest to a person

| auto-route when | share auto-routed | accuracy when auto-routed | share needing a person |
|---|---|---|---|
| always (today's design) | 100.0% | 86.11% | 0.0% |
| confidence ≥ 0.70 (high + medium) | 84.3% | 98.43% | 15.7% |
| confidence ≥ 0.90 (high only) | 83.4% | 98.49% | 16.6% |
| not vague | 84.3% | 98.39% | 15.7% |

Reading: auto-routing only the confident 84% is ~98% right; the other ~16% (almost all vague) go to a person or get the clarifying question. Overall accuracy then depends on how well that person or question does, which history cannot tell us.

## B. Is the right team among the first k shown?

| k | all requests | clear requests | vague requests |
|---|---|---|---|
| 1 | 86.10% | 98.38% | 20.25% |
| 2 | 88.82% | 98.52% | 36.79% |
| 3 | 91.05% | 98.63% | 50.39% |

Reading: for clear requests the first answer is already right 98% of the time; showing more options adds little. For vague requests even three options cover only about half, so a list of options is no substitute for asking the customer.

## C. Decision rule for vague requests, by metric

Only vague requests change; everything else is the shipped router. Cost = misroutes × Rs 698 (historical average per misroute), per 1,000 requests.

| rule for vague requests | accuracy | macro-F1 | vague accuracy | misroute cost / 1,000 requests |
|---|---|---|---|---|
| most common team for the product (C1) | 86.36% | 87.46% | 21.82% | Rs 95,229 |
| product, rebalanced (gamma=0.5) | 86.31% | 87.30% | 21.53% | Rs 95,551 |
| always the most common team (shipped) | 86.11% | 87.83% | 20.25% | Rs 96,947 |
| product, rebalanced (gamma=1.0) | 85.50% | 85.00% | 16.34% | Rs 101,241 |
| random in proportion to past outcomes | 85.39% | 85.06% | 15.66% | Rs 101,993 |

Best for accuracy and cost: **most common team for the product (C1)**. Best for macro-F1: **always the most common team (shipped)**.

