# Model comparison

Truth = team that closed the request (resolution_log.final_team, renames merged).
`vs_outcome_clear` = requests with a stated need; `vs_outcome_vague` = 'please call me about my X' requests.

## A out-of-time

| model | vs outcome | vs bot label | clear requests | vague requests | sd | secs |
|---|---|---|---|---|---|---|
| Lookup on last clause (+TF-IDF fallback) | 0.858 | 0.852 | 0.979 | 0.205 |  | 0.6 |
| bge-small embeddings+LR on outcomes | 0.857 | 0.827 | 0.979 | 0.202 |  | 64.3 |
| TF-IDF+last-clause+meta+LR on outcomes | 0.854 | 0.810 | 0.979 | 0.182 |  | 0.8 |
| TF-IDF+last-clause+LR, clean rows only | 0.850 | 0.801 | 0.979 | 0.155 |  | 0.6 |
| TF-IDF+last-clause+SVM on outcomes | 0.849 | 0.801 | 0.979 | 0.149 |  | 0.7 |
| TF-IDF+last-clause+LR on outcomes | 0.848 | 0.802 | 0.979 | 0.146 |  | 0.6 |
| Rules on last clause (no training) | 0.847 | 0.870 | 0.968 | 0.202 |  | 0.0 |
| TF-IDF word+char+LR on outcomes | 0.838 | 0.793 | 0.968 | 0.143 |  | 2.0 |
| TF-IDF+SVM on outcomes | 0.833 | 0.787 | 0.959 | 0.161 |  | 0.3 |
| TF-IDF+LR on outcomes | 0.830 | 0.787 | 0.957 | 0.149 |  | 0.4 |
| TF-IDF+NB on outcomes | 0.823 | 0.751 | 0.948 | 0.155 |  | 0.1 |
| TF-IDF+LR trained on BOT labels (as briefed) | 0.770 | 0.918 | 0.878 | 0.190 |  | 0.4 |
| Vendor bot (team_label as the prediction) | 0.767 | 1.000 | 0.874 | 0.193 |  | 0.0 |
| Majority class (always Repairs) | 0.233 | 0.289 | 0.239 | 0.202 |  | 0.0 |

## B 5-fold CV

| model | vs outcome | vs bot label | clear requests | vague requests | sd | secs |
|---|---|---|---|---|---|---|
| Lookup on last clause (+TF-IDF fallback) | 0.859 | 0.861 | 0.983 | 0.204 | 0.004 | 2.4 |
| bge-small embeddings+LR on outcomes | 0.856 | 0.823 | 0.983 | 0.184 | 0.007 | 275.3 |
| TF-IDF+last-clause+meta+LR on outcomes | 0.853 | 0.805 | 0.983 | 0.166 | 0.007 | 3.1 |
| TF-IDF+last-clause+LR on outcomes | 0.852 | 0.800 | 0.983 | 0.158 | 0.007 | 2.4 |
| Rules on last clause (no training) | 0.852 | 0.871 | 0.972 | 0.217 | 0.003 | 0.1 |
| TF-IDF+last-clause+SVM on outcomes | 0.851 | 0.802 | 0.983 | 0.155 | 0.007 | 2.7 |
| TF-IDF+last-clause+LR, clean rows only | 0.851 | 0.801 | 0.983 | 0.155 | 0.006 | 2.2 |
| TF-IDF word+char+LR on outcomes | 0.843 | 0.791 | 0.973 | 0.159 | 0.007 | 6.9 |
| TF-IDF+SVM on outcomes | 0.835 | 0.786 | 0.963 | 0.162 | 0.007 | 1.3 |
| TF-IDF+LR on outcomes | 0.834 | 0.785 | 0.963 | 0.156 | 0.007 | 1.7 |
| TF-IDF+NB on outcomes | 0.828 | 0.756 | 0.956 | 0.151 | 0.006 | 0.6 |
| TF-IDF+LR trained on BOT labels (as briefed) | 0.773 | 0.922 | 0.880 | 0.210 | 0.006 | 1.5 |
| Vendor bot (team_label as the prediction) | 0.772 | 1.000 | 0.878 | 0.213 |  | 0.0 |
| Majority class (always Repairs) | 0.234 | 0.289 | 0.237 | 0.217 | 0.000 | 0.0 |

## C unseen wording

| model | vs outcome | vs bot label | clear requests | vague requests | sd | secs |
|---|---|---|---|---|---|---|
| Rules on last clause (no training) | 0.852 | 0.871 | 0.972 | 0.223 | 0.047 | 0.0 |
| bge-small embeddings+LR on outcomes | 0.789 | 0.737 | 0.910 | 0.161 | 0.037 | 241.9 |
| Vendor bot (team_label as the prediction) | 0.772 | 1.000 | 0.878 | 0.213 |  | 0.0 |
| TF-IDF+last-clause+SVM on outcomes | 0.461 | 0.464 | 0.514 | 0.199 | 0.059 | 2.8 |
| TF-IDF+last-clause+meta+LR on outcomes | 0.455 | 0.447 | 0.506 | 0.198 | 0.076 | 2.9 |
| TF-IDF+last-clause+LR on outcomes | 0.455 | 0.448 | 0.506 | 0.194 | 0.084 | 2.3 |
| Lookup on last clause (+TF-IDF fallback) | 0.455 | 0.448 | 0.506 | 0.194 | 0.084 | 2.4 |
| TF-IDF+last-clause+LR, clean rows only | 0.450 | 0.452 | 0.501 | 0.190 | 0.088 | 2.2 |
| TF-IDF word+char+LR on outcomes | 0.336 | 0.319 | 0.364 | 0.180 | 0.099 | 7.0 |
| TF-IDF+NB on outcomes | 0.335 | 0.319 | 0.366 | 0.191 | 0.135 | 0.5 |
| TF-IDF+LR on outcomes | 0.299 | 0.283 | 0.322 | 0.178 | 0.060 | 1.6 |
| TF-IDF+SVM on outcomes | 0.267 | 0.250 | 0.286 | 0.178 | 0.046 | 1.3 |
| Majority class (always Repairs) | 0.234 | 0.289 | 0.241 | 0.223 | 0.066 | 0.0 |
| TF-IDF+LR trained on BOT labels (as briefed) | 0.233 | 0.363 | 0.251 | 0.172 | 0.105 | 1.5 |
