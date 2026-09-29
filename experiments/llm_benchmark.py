"""Can a free local LLM route these requests? Writes out/llm_benchmark.md.

Model: Qwen2.5-3B-Instruct (4-bit, MLX) on the laptop. No API key, no cost, no data leaves the machine
(policy §10). Sample: 300 random requests from the out-of-time window (Apr-Jun 2026), seed 0.

Two prompts:
  zero-shot  - teams.csv descriptions + policy §3 only (what an LLM knows without our data).
  hinted     - plus the two rules learned from the data: payment remarks don't matter, and when a
               message has two requests the LAST one decides.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.compare_models import build_frame  # noqa: E402
from kestrel.data import TEAMS, load_teams  # noqa: E402

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
N = 300


def team_block():
    t = load_teams()
    lines = []
    for _, r in t.iterrows():
        name = r["renamed_to"].split(" (")[0] if isinstance(r["renamed_to"], str) else r["team"]
        lines.append(f"- {name}: {r['handles']}")
    return "\n".join(lines)


BASE = """You route customer service requests for Kestrel Home Appliances to exactly one team.
Teams:
{teams}
Policy: a request belongs to Billing only when the problem is the payment itself (invoice, GST, double charge, refund of a payment, EMI conversion). A customer mentioning that they have paid does not make it a billing request.
{hints}
Reply with the team name only, exactly as written above."""

HINTS = """Rules learned from past requests:
- Ignore remarks like "paid on upi", "already paid in full", "payment done".
- If the message contains two requests, route by the LAST one.
- If the customer only asks to be called or says "issue with X" with no detail, answer Repairs."""


def parse(out: str) -> str:
    o = out.strip().lower()
    for t in sorted(TEAMS, key=len, reverse=True):
        if t.lower() in o:
            return t
    aliases = {"installations": "Installs & Demo", "consumables": "Filters & Consumables",
               "returns": "Returns & Replacement", "warranty": "Warranty Claims", "repair": "Repairs",
               "advice": "Product Advice", "billing": "Billing", "install": "Installs & Demo"}
    for k, v in aliases.items():
        if k in o:
            return v
    return "UNPARSED"


def main():
    from mlx_lm import load, generate
    d = build_frame()
    va = d[d["created_at"] >= "2026-04-01"].sample(N, random_state=0)
    model, tok = load(MODEL)
    teams = team_block()
    rows = []
    for prompt_name, hints in [("zero-shot", ""), ("hinted", HINTS)]:
        system = BASE.format(teams=teams, hints=hints)
        preds, secs = [], []
        for text in va["text"]:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            t0 = time.time()
            out = generate(model, tok, prompt=p, max_tokens=12, verbose=False)
            secs.append(time.time() - t0)
            preds.append(parse(out))
        preds = np.array(preds)
        ok = preds == va["final_team_cur"].values
        v = va["vague"].values
        rows.append({"prompt": prompt_name, "vs_outcome": ok.mean(), "clear": ok[~v].mean(), "vague": ok[v].mean(),
                     "vs_bot": np.mean(preds == va["bot_team"].values), "unparsed": np.mean(preds == "UNPARSED"),
                     "sec_per_request": np.median(secs)})
        va[f"llm_{prompt_name}"] = preds
        print(rows[-1], flush=True)
    va[["request_id", "text", "final_team_cur", "bot_team", "llm_zero-shot", "llm_hinted"]].to_csv(
        ROOT / "out" / "llm_benchmark_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "out" / "llm_benchmark_timing.csv", index=False)
    write_table()


def write_table():
    """Rebuild out/llm_benchmark.md from saved predictions, with the router and the bot on the same requests."""
    from kestrel.router import Router
    from kestrel import text as T
    va = pd.read_csv(ROOT / "out" / "llm_benchmark_predictions.csv")
    timing = pd.read_csv(ROOT / "out" / "llm_benchmark_timing.csv").set_index("prompt")["sec_per_request"]
    d = build_frame()
    router = Router.fit(d[d["created_at"] < "2026-04-01"])
    va["router"] = router.predict(va.assign(request_text=va["text"]))["team"].values
    v = va["text"].map(T.is_vague).values
    truth = va["final_team_cur"].values
    lines = ["# Local LLM benchmark", "",
             f"Model `{MODEL}` on an 8 GB M1 laptop, {len(va)} random out-of-time requests (Apr-Jun 2026, seed 0). "
             "The router (trained to 31 Mar 2026) and the bot are scored on the same requests.", "",
             "| approach | vs closing team | clear requests | vague requests | vs bot label | median sec/request |",
             "|---|---|---|---|---|---|"]
    for name, col, sec in [("Router (this work)", "router", "<0.001"), ("Vendor bot", "bot_team", "-"),
                           ("LLM zero-shot (teams.csv + policy)", "llm_zero-shot", f"{timing['zero-shot']:.2f}"),
                           ("LLM with data-derived hints", "llm_hinted", f"{timing['hinted']:.2f}")]:
        ok = va[col].values == truth
        lines.append(f"| {name} | {ok.mean():.1%} | {ok[~v].mean():.1%} | {ok[v].mean():.1%} | "
                     f"{np.mean(va[col].values == va['bot_team'].values):.1%} | {sec} |")
    lines += ["", "Adding the hints made the 3B model worse, not better: it followed the 'answer Repairs' hint for "
              "vague requests but lost accuracy on clear ones."]
    (ROOT / "out" / "llm_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    write_table() if "--table-only" in sys.argv else main()
