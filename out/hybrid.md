# Hybrid router variants

Seen wording needs >= 3 training examples.

## A out-of-time

| model | vs outcome | vs bot | clear | vague | sd | secs |
|---|---|---|---|---|---|---|
| Hybrid + rules_data fallback | 0.858 | 0.882 | 0.980 | 0.202 |  | 0.0 |
| Hybrid + rules_teams fallback | 0.858 | 0.882 | 0.980 | 0.202 |  | 0.0 |
| Hybrid + TF-IDF(last+full) LR fallback | 0.857 | 0.881 | 0.979 | 0.202 |  | 0.3 |
| Hybrid + TF-IDF word+char LR fallback | 0.857 | 0.881 | 0.979 | 0.202 |  | 0.9 |
| Hybrid + bge-small embeddings fallback | 0.857 | 0.881 | 0.979 | 0.202 |  | 56.8 |
| TF-IDF(last+full)+meta LR alone, vague->pooled | 0.857 | 0.881 | 0.979 | 0.202 |  | 0.5 |
| rules_teams alone (vague->Repairs) | 0.792 | 0.818 | 0.902 | 0.202 |  | 0.0 |

## B 5-fold CV

| model | vs outcome | vs bot | clear | vague | sd | secs |
|---|---|---|---|---|---|---|
| TF-IDF(last+full)+meta LR alone, vague->pooled | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 1.7 |
| Hybrid + TF-IDF(last+full) LR fallback | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 1.9 |
| Hybrid + TF-IDF word+char LR fallback | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 4.6 |
| Hybrid + rules_data fallback | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 0.1 |
| Hybrid + rules_teams fallback | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 0.1 |
| Hybrid + bge-small embeddings fallback | 0.861 | 0.881 | 0.983 | 0.217 | 0.003 | 229.7 |
| rules_teams alone (vague->Repairs) | 0.790 | 0.813 | 0.898 | 0.217 | 0.004 | 0.1 |

## C unseen wording

| model | vs outcome | vs bot | clear | vague | sd | secs |
|---|---|---|---|---|---|---|
| Hybrid + rules_data fallback | 0.852 | 0.871 | 0.972 | 0.223 | 0.047 | 0.1 |
| Hybrid + bge-small embeddings fallback | 0.792 | 0.815 | 0.904 | 0.223 | 0.037 | 230.7 |
| Hybrid + rules_teams fallback | 0.789 | 0.813 | 0.897 | 0.223 | 0.073 | 0.1 |
| rules_teams alone (vague->Repairs) | 0.789 | 0.813 | 0.897 | 0.223 | 0.073 | 0.1 |
| Hybrid + TF-IDF word+char LR fallback | 0.489 | 0.531 | 0.541 | 0.223 | 0.084 | 4.4 |
| Hybrid + TF-IDF(last+full) LR fallback | 0.473 | 0.516 | 0.521 | 0.223 | 0.080 | 1.4 |
| TF-IDF(last+full)+meta LR alone, vague->pooled | 0.470 | 0.514 | 0.520 | 0.223 | 0.069 | 1.6 |
