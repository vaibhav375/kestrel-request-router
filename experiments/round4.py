"""Round 4 (docs/ROUND4_PROTOCOL.md): each switch alone vs the round-3 router, and all together.

    python experiments/round4.py            -> out/round4.md, out/round4_*.csv

Needs kestrel/resources/paraphrases.csv and out/synthetic_hinglish.csv (experiments/generate_synthetic.py).
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kestrel.router as KR  # noqa: E402
from experiments.challengers import FOLDS  # noqa: E402
from experiments.compare_models import build_frame  # noqa: E402
from kestrel import text as T  # noqa: E402

PARA = pd.read_csv(KR.PARAPHRASES_PATH)
VARIANTS = {
    "R3 (round-3 router)": set(),
    "A1 vague guess by product": {"A1"},
    "B1 fuzzy match": {"B1"},
    "B2 combined fallback": {"B2"},
    "B3 paraphrase lookup": {"B3"},
    "B4 Hinglish + spelling repair": {"B4"},
    "B1+B2+B3+B4": {"B1", "B2", "B3", "B4"},
    "ALL (A1 B1 B2 B3 B4 C2)": {"A1", "B1", "B2", "B3", "B4", "C2"},
}


class EmbedNearest(KR.Router):
    """B2 with bge-small embedding similarity instead of character n-grams (comparison only: needs PyTorch)."""
    _cache: dict = {}

    @classmethod
    def encode(cls, texts):
        from sentence_transformers import SentenceTransformer
        if "m" not in cls._cache:
            cls._cache["m"] = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        todo = [t for t in dict.fromkeys(texts) if t not in cls._cache]
        if todo:
            for t, e in zip(todo, cls._cache["m"].encode(todo, batch_size=128, normalize_embeddings=True,
                                                         show_progress_bar=False)):
                cls._cache[t] = e
        return np.vstack([cls._cache[t] for t in texts])

    def _nearest(self, last):
        if not self._fuzzy_keys or not last:
            return None, None, 0.0
        sims = self.encode(self._fuzzy_keys) @ self.encode([last])[0]
        i = int(sims.argmax())
        return self._fuzzy_keys[i], self._key_team(self._fuzzy_keys[i]), float(sims[i])


def fit(tr, feats, cls=KR.Router, rules=None):
    r = cls.fit(tr, features=feats, paraphrases=PARA)
    if rules is not None:
        r.rules = list(rules)
    return r


def mcnemar(ok_new, ok_base):
    b, c = int(np.sum(ok_new & ~ok_base)), int(np.sum(~ok_new & ok_base))
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def corrupt(text, p, rng):
    """Synthetic typos: each word (3+ letters) hit with probability p by one random edit."""
    def edit(w):
        if len(w) < 3 or rng.random() >= p:
            return w
        i = rng.randrange(len(w))
        op = rng.choice(["del", "swap", "sub", "dup"])
        if op == "del":
            return w[:i] + w[i + 1:]
        if op == "swap" and i < len(w) - 1:
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op == "sub":
            return w[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[i + 1:]
        return w[:i] + w[i] + w[i:]
    return re.sub(r"[A-Za-z]+", lambda m: edit(m.group(0)), text)


def table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    return [head, "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]


def main():
    t0 = time.time()
    d = build_frame()
    L = ["# Round 4 results", "", "Protocol: `docs/ROUND4_PROTOCOL.md` (committed before these tests). "
         "Baseline R3 = the router after round 3. p = paired McNemar vs R3 on the same requests.", ""]
    summary = {}

    # ------------------------------------------------ E1 known wording (rolling backtest)
    preds = {k: [] for k in VARIANTS}
    truth, kinds, auto = [], [], []
    for name, start, end in FOLDS:
        tr, va = d[d["created_at"] < start], d[(d["created_at"] >= start) & (d["created_at"] < end)]
        for k, feats in VARIANTS.items():
            p = fit(tr, feats).predict(va)
            preds[k].append(p["team"].values)
            if k.startswith("ALL"):
                kinds.append(p["route_type"].values)
                auto.append(p["auto_route"].values)
        truth.append(va["final_team_cur"].values)
    truth = np.concatenate(truth)
    kinds, auto = np.concatenate(kinds), np.concatenate(auto)
    ok_base = np.concatenate(preds["R3 (round-3 router)"]) == truth
    rows = []
    for k in VARIANTS:
        ok = np.concatenate(preds[k]) == truth
        b, c, p = mcnemar(ok, ok_base)
        rows.append({"variant": k, "accuracy": f"{ok.mean():.2%}", "vs R3": f"{100 * (ok.mean() - ok_base.mean()):+.2f}",
                     "wins/losses": f"{b}/{c}", "p": f"{p:.3f}"})
        summary.setdefault(k, {})["E1"] = ok.mean()
    L += ["## E1 Known wording: rolling backtest, 6,502 predictions", ""] + table(rows, list(rows[0])) + [""]
    ok_all = np.concatenate(preds["ALL (A1 B1 B2 B3 B4 C2)"]) == truth
    L += ["**C1 (auto-route flag), ALL variant:** "
          f"auto-routes {auto.mean():.1%} of requests at {ok_all[auto].mean():.2%} accuracy; "
          f"the other {1 - auto.mean():.1%} get the clarifying question.", ""]
    summary["C1"] = {"auto_share": float(auto.mean()), "auto_acc": float(ok_all[auto].mean())}
    print("E1 done", round(time.time() - t0), flush=True)

    # ------------------------------------------------ E2 new wording (strict groups, teams.csv-only rules)
    fam = d["last"].map(lambda s: re.sub(r"\d+", "#", s))
    clear_mask = ~d["vague"].values
    e2 = {k: np.zeros(len(d), bool) for k in list(VARIANTS) + ["B2 with embeddings (comparison)"]}
    detail = {}
    for a, b in GroupKFold(5).split(d, d["final_team_cur"], groups=fam):
        tr, va = d.iloc[a], d.iloc[b]
        for k in e2:
            if k.startswith("A1") or k.startswith("ALL"):
                feats = VARIANTS[k]
            elif k.startswith("B2 with"):
                feats = {"B2"}
            else:
                feats = VARIANTS[k]
            cls = EmbedNearest if k.startswith("B2 with") else KR.Router
            p = fit(tr, feats, cls, KR.TEAMS_ONLY_RULES).predict(va)
            e2[k][b] = p["team"].values == va["final_team_cur"].values
            if k == "B1+B2+B3+B4":
                detail.setdefault("route_detail", []).append(p.loc[~va["vague"].values, "route_detail"])
                detail.setdefault("ok", []).append(e2[k][b][~va["vague"].values])
    base = e2["R3 (round-3 router)"][clear_mask]
    rows = []
    for k, ok in e2.items():
        okc = ok[clear_mask]
        b_, c_, p = mcnemar(okc, base)
        rows.append({"variant": k, "clear requests right": f"{okc.mean():.2%}", "vs R3": f"{100 * (okc.mean() - base.mean()):+.2f}",
                     "wins/losses": f"{b_}/{c_}", "p": f"{p:.3g}"})
        summary.setdefault(k, {})["E2"] = okc.mean()
    L += ["## E2 New wording (strict: phrasings differing only in a number are one group; teams.csv-only rules)",
          f"", f"{clear_mask.sum():,} clear requests, every one scored by a router that never saw its phrasing.", ""]
    L += table(rows, list(rows[0])) + [""]
    rd = pd.DataFrame({"detail": pd.concat(detail["route_detail"]).values, "ok": np.concatenate(detail["ok"])})
    g = rd.groupby("detail")["ok"].agg(["size", "mean"])
    L += ["Which step decided, B1+B2+B3+B4 stack (E2):", "", "| step | share | right |", "|---|---|---|"]
    for k, r in g.sort_values("size", ascending=False).iterrows():
        L.append(f"| {k} | {r['size'] / len(rd):.1%} | {r['mean']:.1%} |")
    L.append("")
    print("E2 done", round(time.time() - t0), flush=True)

    # ------------------------------------------------ E3 typos, E4 Hinglish (synthetic)
    tr, va = d[d["created_at"] < "2026-04-01"], d[(d["created_at"] >= "2026-04-01") & ~d["vague"]]
    hin = pd.read_csv(ROOT / "out" / "synthetic_hinglish.csv")
    routers = {k: fit(tr, f) for k, f in VARIANTS.items()}
    for test_name, frames in [
        ("E3 Typos (synthetic): light = 15% of words corrupted, heavy = 30%",
         {"clean": va.assign(t=va["text"]),
          "light": va.assign(t=[corrupt(s, 0.15, random.Random(i)) for i, s in enumerate(va["text"])]),
          "heavy": va.assign(t=[corrupt(s, 0.30, random.Random(10_000 + i)) for i, s in enumerate(va["text"])])}),
        ("E4 Hinglish (synthetic, LLM-rewritten, 300 requests)",
         {"original English": hin.assign(t=hin["text"]), "Hinglish": hin.assign(t=hin["hinglish"])}),
    ]:
        res = {k: {c: r.predict(f, text_col="t")["team"].values == f["final_team_cur"].values for c, f in frames.items()}
               for k, r in routers.items()}
        rows = []
        for k in VARIANTS:
            row = {"variant": k}
            for c in frames:
                row[c] = f"{res[k][c].mean():.2%}"
            last_col = list(frames)[-1]
            b_, c_, p = mcnemar(res[k][last_col], res["R3 (round-3 router)"][last_col])
            row[f"{last_col} vs R3"] = f"{100 * (res[k][last_col].mean() - res['R3 (round-3 router)'][last_col].mean()):+.2f} (p {p:.3g})"
            rows.append(row)
            summary.setdefault(k, {})[test_name[:2]] = res[k][last_col].mean()
        L += [f"## {test_name}", ""] + table(rows, list(rows[0])) + [""]
        if test_name.startswith("E3"):
            ex = frames["heavy"]["t"].iloc[:3].tolist()
            L += ["Example heavy-typo inputs: " + " · ".join(f"“{T.ID_RE.sub('[no.]', e.lower())}”" for e in ex), ""]
        else:
            L += ["Example Hinglish inputs: " + " · ".join(f"“{T.ID_RE.sub('[no.]', e.lower())}”" for e in hin["hinglish"].iloc[:3]), ""]
    print("E3/E4 done", round(time.time() - t0), flush=True)

    # ------------------------------------------------ E5 corrections loop (B5)
    seen_acc = float(ok_base[kinds != "vague"].mean())  # known-wording accuracy on requests that state a need
    curve = {}
    for a, b in GroupKFold(5).split(d, d["final_team_cur"], groups=fam):
        tr, va = d.iloc[a], d.iloc[b]
        va = va[~va["vague"]].sort_values("created_at")
        va_fam = va["last"].map(lambda s: re.sub(r"\d+", "#", s))
        rank = va.groupby(va_fam).cumcount()
        for k in (0, 1, 3, 5, 10):
            add, rest = va[rank < k], va[rank >= 10]  # always score the same later requests (rank >= 10)
            r = KR.Router.fit(pd.concat([tr, add]), features=set())
            r.rules = list(KR.TEAMS_ONLY_RULES)
            ok = r.predict(rest)["team"].values == rest["final_team_cur"].values
            curve.setdefault(k, []).append(ok)
    L += ["## E5 Corrections loop (B5): accuracy on a new phrasing after k agent-confirmed examples are added", "",
          "| confirmed examples added per new phrasing | accuracy on its later requests |", "|---|---|"]
    for k, oks in curve.items():
        L.append(f"| {k} | {np.concatenate(oks).mean():.2%} |")
    L += ["", f"Reference: accuracy on known wording for requests that state a need = {seen_acc:.2%}; the protocol's "
          f"bar is 95% of that = {0.95 * seen_acc:.2%}.", ""]
    summary["B5"] = {k: float(np.concatenate(v).mean()) for k, v in curve.items()}
    print("E5 done", round(time.time() - t0), flush=True)

    # ------------------------------------------------ E6 clarifying-question option order (C2)
    cover = {"fixed order (today)": [], "pooled likelihood order": [], "product likelihood order (C2)": []}
    for name, start, end in FOLDS:
        tr, va = d[d["created_at"] < start], d[(d["created_at"] >= start) & (d["created_at"] < end)]
        va = va[va["vague"]]
        r = KR.Router.fit(tr, features={"C2"})
        fixed = [t for _, t in KR.CLARIFY_OPTIONS][:3]
        pooled = [t for t, _ in r.vague_counts.most_common(3)]
        for prod, tru in zip(va["product_family"], va["final_team_cur"]):
            cover["fixed order (today)"].append(tru in fixed)
            cover["pooled likelihood order"].append(tru in pooled)
            cover["product likelihood order (C2)"].append(tru in r._vague_posterior(prod).nlargest(3).index)
    L += ["## E6 Clarifying question: is the right team among the first 3 options? (vague requests, backtest)", "",
          "| option order | true team in first 3 |", "|---|---|"]
    for k, v in cover.items():
        L.append(f"| {k} | {np.mean(v):.2%} (n={len(v):,}) |")
    L.append("")
    summary["C2"] = {k: float(np.mean(v)) for k, v in cover.items()}

    (ROOT / "out" / "round4.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(summary).to_csv(ROOT / "out" / "round4_summary.csv")
    print("\n".join(L))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
