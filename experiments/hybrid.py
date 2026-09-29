"""Hybrid router variants, same three validation schemes as compare_models.py. Writes out/hybrid.md.

Router = (1) vague last clause -> pooled best guess (+ flag for clarification)
         (2) wording seen in training (>= MIN_SUPPORT times) -> majority outcome for that wording
         (3) otherwise -> a fallback, which is what these variants compare.

Two rule sets are compared as fallbacks:
  * rules_data  - keywords I wrote AFTER reading the data (optimistic on the unseen-wording test: leakage).
  * rules_teams - keywords taken only from teams.csv 'handles' text + policy §3 (fair robustness check).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.compare_models import TfidfModel, RulesModel, EmbedModel, build_frame  # noqa: E402

MIN_SUPPORT = 3

TEAMS_RULES = [  # words from teams.csv 'handles' only
    ("Installs & Demo", r"install|demo|wall.?mount"),
    ("Returns & Replacement", r"damaged|wrong|incomplete|missing|return|exchange|replace(?!ment filter)"),
    ("Warranty Claims", r"warranty|shield|registration|register|coverage|covered|claim"),
    ("Billing", r"invoice|gst|double|charged|refund|payment|emi|coupon|bill"),
    ("Filters & Consumables", r"filter|candle|membrane|jar|brush|blade|amc|spare|consumable"),
    ("Repairs", r"fault|breakdown|error|noise|leak|not working|broken|repair|fix"),
    ("Product Advice", r"how|which|what|can |is P|usage|use|difference|best|safe"),
]


class TeamsRules(RulesModel):
    RULES = TEAMS_RULES


class Hybrid:
    def __init__(self, fallback):
        self.fallback_factory = fallback

    def fit(self, df, y):
        y = pd.Series(np.asarray(y), index=df.index)
        vague = df["vague"]
        self.vague_team = y[vague].value_counts().index[0] if vague.any() else "Repairs"
        clear = df[~vague]
        yc = y[~vague]
        g = pd.DataFrame({"k": clear["last"], "y": yc}).groupby("k")["y"]
        counts = g.size()
        self.table = g.agg(lambda s: s.value_counts().index[0])[counts >= MIN_SUPPORT].to_dict()
        self.fb = self.fallback_factory().fit(clear, yc.values)
        return self

    def predict(self, df):
        fb = self.fb.predict(df)
        out = []
        for v, k, f in zip(df["vague"], df["last"], fb):
            out.append(self.vague_team if v else self.table.get(k, f))
        return np.array(out)

    def route_kind(self, df):
        return np.where(df["vague"], "vague", np.where(df["last"].isin(self.table), "seen", "unseen"))


VARIANTS = {
    "Hybrid + TF-IDF(last+full) LR fallback": lambda: Hybrid(lambda: TfidfModel(last=True)),
    "Hybrid + TF-IDF word+char LR fallback": lambda: Hybrid(lambda: TfidfModel(last=True, char=True)),
    "Hybrid + rules_data fallback": lambda: Hybrid(RulesModel),
    "Hybrid + rules_teams fallback": lambda: Hybrid(TeamsRules),
    "Hybrid + bge-small embeddings fallback": lambda: Hybrid(EmbedModel),
    "TF-IDF(last+full)+meta LR alone, vague->pooled": lambda: Hybrid(lambda: TfidfModel(last=True, meta=True)),
}


def score(p, va):
    ok = p == va["final_team_cur"].values
    v = va["vague"].values
    return {"vs_outcome": ok.mean(), "vs_bot": np.mean(p == va["bot_team"].values),
            "clear": ok[~v].mean(), "vague": ok[v].mean()}


def main():
    d = build_frame()
    rows = []
    tr_idx = np.where(d["created_at"] < "2026-04-01")[0]
    va_idx = np.where(d["created_at"] >= "2026-04-01")[0]
    schemes = {
        "A out-of-time": [(tr_idx, va_idx)],
        "B 5-fold CV": list(StratifiedKFold(5, shuffle=True, random_state=0).split(d, d["final_team_cur"])),
        "C unseen wording": list(GroupKFold(5).split(d, d["final_team_cur"], groups=d["last"])),
    }
    # stand-alone rules for reference
    variants = dict(VARIANTS)
    variants["rules_teams alone (vague->Repairs)"] = lambda: Hybrid(TeamsRules)
    for scheme, splits in schemes.items():
        for name, f in variants.items():
            t0 = time.time()
            accs = []
            for a, b in splits:
                tr, va = d.iloc[a], d.iloc[b]
                m = f().fit(tr, tr["final_team_cur"].values)
                if name.startswith("rules_teams alone"):
                    m.table = {}
                accs.append(score(m.predict(va), va))
            r = pd.DataFrame(accs)
            rows.append({"scheme": scheme, "model": name, **r.mean().to_dict(), "sd": r["vs_outcome"].std(),
                         "secs": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    lines = ["# Hybrid router variants", "", f"Seen wording needs >= {MIN_SUPPORT} training examples.", ""]
    for scheme, g in res.groupby("scheme", sort=False):
        lines += [f"## {scheme}", "", "| model | vs outcome | vs bot | clear | vague | sd | secs |", "|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values("vs_outcome", ascending=False).iterrows():
            sd = "" if pd.isna(r["sd"]) else f"{r['sd']:.3f}"
            lines.append(f"| {r['model']} | {r['vs_outcome']:.3f} | {r['vs_bot']:.3f} | {r['clear']:.3f} | {r['vague']:.3f} | {sd} | {r['secs']} |")
        lines.append("")
    (ROOT / "out" / "hybrid.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
