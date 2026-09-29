# Local LLM benchmark

Model `mlx-community/Qwen2.5-3B-Instruct-4bit` on an 8 GB M1 laptop, 300 random out-of-time requests (Apr-Jun 2026, seed 0). The router (trained to 31 Mar 2026) and the bot are scored on the same requests.

| approach | vs closing team | clear requests | vague requests | vs bot label | median sec/request |
|---|---|---|---|---|---|
| Router (this work) | 84.3% | 96.9% | 15.2% | 85.7% | <0.001 |
| Vendor bot | 74.0% | 84.6% | 15.2% | 100.0% | - |
| LLM zero-shot (teams.csv + policy) | 71.3% | 82.3% | 10.9% | 74.0% | 2.94 |
| LLM with data-derived hints | 68.0% | 77.6% | 15.2% | 71.0% | 3.38 |

Adding the hints made the 3B model worse, not better: it followed the 'answer Repairs' hint for vague requests but lost accuracy on clear ones.
