"""Generate round-4 synthetic data with the free local LLM (Qwen2.5-3B-Instruct, MLX). No API, no cost.

    python experiments/generate_synthetic.py paraphrases   # -> kestrel/resources/paraphrases.csv (B3)
    python experiments/generate_synthetic.py hinglish      # -> out/synthetic_hinglish.csv (E4 test set)

Paraphrases are made from the known phrasings only (product masked to "water purifier"), so they carry no customer
data. The Hinglish test set was generated AFTER kestrel/repair.py was committed (docs/ROUND4_PROTOCOL.md).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.compare_models import build_frame  # noqa: E402

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def family(phrase: str) -> str:
    return re.sub(r"\d+", "#", phrase)


def chat(model, tok, system, user, max_tokens):
    from mlx_lm import generate
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt=p, max_tokens=max_tokens, verbose=False)


def paraphrases():
    from mlx_lm import load
    d = build_frame()
    clear = d[~d["vague"]]
    fams = clear.groupby(clear["last"].map(family))["last"].agg(lambda s: s.value_counts().index[0])
    model, tok = load(MODEL)
    system = ("You rewrite short customer-service requests about home appliances. Keep exactly the same problem or "
              "question. Do not add new problems. Plain everyday English, as a customer would type it.")
    rows = []
    for fam, example in fams.items():
        shown = example.replace("P", "water purifier")
        out = chat(model, tok, system, f"Request: \"{shown}\"\nWrite 10 different short ways a customer might say "
                                       "this. One per line, no numbering.", 400)
        for line in out.splitlines():
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"').strip()
            if 3 <= len(line) <= 120:
                rows.append({"source": example, "family": fam, "paraphrase": line})
        print(f"{fam[:50]:52s} {sum(r['family'] == fam for r in rows)} paraphrases", flush=True)
    out = ROOT / "kestrel" / "resources"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(["family", "paraphrase"]).to_csv(out / "paraphrases.csv", index=False)


def hinglish(n=300):
    from mlx_lm import load
    d = build_frame()
    va = d[(d["created_at"] >= "2026-04-01") & ~d["vague"]].sample(n, random_state=0)
    model, tok = load(MODEL)
    system = ("You rewrite customer messages the way a customer in Pune would type them on WhatsApp in Hinglish: "
              "Hindi written in English letters, mixed with English words. Keep the same meaning and the same product. "
              "Output only the rewritten message.")
    outs = []
    for i, text in enumerate(va["text"]):
        outs.append(chat(model, tok, system, f"Message: {text}", 80).strip().splitlines()[0].strip().strip('"'))
        if i % 25 == 0:
            print(i, text, "->", outs[-1], flush=True)
    va = va.assign(hinglish=outs)
    va[["request_id", "text", "hinglish", "product_family", "final_team_cur"]].to_csv(
        ROOT / "out" / "synthetic_hinglish.csv", index=False)


if __name__ == "__main__":
    {"paraphrases": paraphrases, "hinglish": hinglish}[sys.argv[1]]()
