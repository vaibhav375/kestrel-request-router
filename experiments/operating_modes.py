"""Round 3: levers that change how the router is USED, not the model. Writes out/operating_modes.md.

Rounds 1-2 showed the text carries no more signal (oracle ceiling 86.25%). What is left:
  A. Selective routing  - auto-route only above a confidence threshold, send the rest to a person.
  B. Top-k suggestions  - how often the right team is among the first k the screen shows.
  C. Decision rule for vague requests, per metric - accuracy, macro-F1 and rupee cost can disagree.
Same rolling-origin backtest as experiments/challengers.py (3 quarters, 6,502 predictions, trained only on the past).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.challengers import FOLDS, ALPHA  # noqa: E402
from experiments.compare_models import build_frame  # noqa: E402
from kestrel.business import EXTRA_CONTACT_RS, TRANSFER_RS  # noqa: E402
from kestrel.data import TEAMS  # noqa: E402
from kestrel.router import Router  # noqa: E402

RNG = np.random.default_rng(0)


def vague_rules(tr: pd.DataFrame, va: pd.DataFrame) -> dict[str, np.ndarray]:
    """Alternative decisions for vague requests. Each returns one team per row of va (used only where vague)."""
    tv = tr[tr["vague"]]
    prior = tv["final_team_cur"].value_counts(normalize=True).reindex(TEAMS, fill_value=0)
    counts = tv.groupby("product_family")["final_team_cur"].value_counts().unstack(fill_value=0).reindex(columns=TEAMS, fill_value=0)

    def post(prod):
        c = counts.loc[prod] if prod in counts.index else pd.Series(0, index=TEAMS)
        return (c + ALPHA * prior) / (c.sum() + ALPHA)

    posts = pd.DataFrame([post(p) for p in va["product_family"]], index=va.index)
    rules = {"always the most common team (shipped)": np.array([prior.idxmax()] * len(va)),
             "most common team for the product (C1)": posts.idxmax(axis=1).values}
    for g in (0.5, 1.0):  # divide by prior^g: favours teams that are under-predicted -> balances recall
        rules[f"product, rebalanced (gamma={g})"] = posts.div(prior ** g).idxmax(axis=1).values
    rules["random in proportion to past outcomes"] = RNG.choice(TEAMS, size=len(va), p=prior.values)
    return rules


def main():
    d = build_frame()
    rows = []
    for name, start, end in FOLDS:
        tr = d[d["created_at"] < start]
        va = d[(d["created_at"] >= start) & (d["created_at"] < end)].copy()
        r = Router.fit(tr)
        p = r.predict(va)
        va["pred"], va["kind"], va["conf"] = p["team"].values, p["route_type"].values, p["confidence"].values
        va["alts"] = [[a["team"] for a in al] for al in p["alternatives"]]
        for rule, teams in vague_rules(tr, va).items():
            va[f"rule::{rule}"] = np.where(va["kind"] == "vague", teams, va["pred"])
        va["fold"] = name
        rows.append(va)
    v = pd.concat(rows)
    truth = v["final_team_cur"].values
    ok = v["pred"].values == truth
    cost_per_miss = None
    # historical cost of one misroute (transfers x Rs 305 + Rs 260), from the training history
    mis = d["first_team_cur"] != d["final_team_cur"]
    cost_per_miss = float((d.loc[mis, "transfers"] * TRANSFER_RS + EXTRA_CONTACT_RS).mean())

    L = ["# Round 3: operating levers (not model changes)", "",
         f"Rolling-origin backtest, {len(v):,} predictions over Oct 2025 – Jun 2026, router trained only on earlier data.",
         f"Router accuracy on this backtest: **{ok.mean():.2%}**.", ""]

    # ---- A. selective routing
    L += ["## A. Auto-route only when confident; send the rest to a person", "",
          "| auto-route when | share auto-routed | accuracy when auto-routed | share needing a person |", "|---|---|---|---|"]
    for label, m in [("always (today's design)", np.ones(len(v), bool)),
                     ("confidence ≥ 0.70 (high + medium)", v["conf"].values >= 0.70),
                     ("confidence ≥ 0.90 (high only)", v["conf"].values >= 0.90),
                     ("not vague", v["kind"].values != "vague")]:
        L.append(f"| {label} | {m.mean():.1%} | {ok[m].mean():.2%} | {1 - m.mean():.1%} |")
    L += ["", "Reading: auto-routing only the confident 84% is ~98% right; the other ~16% (almost all vague) go to a "
          "person or get the clarifying question. Overall accuracy then depends on how well that person or question "
          "does, which history cannot tell us.", ""]

    # ---- B. top-k
    topk = {}
    for k in (1, 2, 3):
        hit = np.array([t in (a[:k] if a else [p]) for t, a, p in zip(truth, v["alts"], v["pred"])])
        topk[k] = hit
    vm = v["kind"].values == "vague"
    L += ["## B. Is the right team among the first k shown?", "",
          "| k | all requests | clear requests | vague requests |", "|---|---|---|---|"]
    for k, hit in topk.items():
        L.append(f"| {k} | {hit.mean():.2%} | {hit[~vm].mean():.2%} | {hit[vm].mean():.2%} |")
    L += ["", "Reading: for clear requests the first answer is already right 98% of the time; showing more options adds "
          "little. For vague requests even three options cover only about half, so a list of options is no substitute "
          "for asking the customer.", ""]

    # ---- C. vague decision rule vs metric
    L += ["## C. Decision rule for vague requests, by metric", "",
          "Only vague requests change; everything else is the shipped router. Cost = misroutes × "
          f"Rs {cost_per_miss:.0f} (historical average per misroute), per 1,000 requests.", "",
          "| rule for vague requests | accuracy | macro-F1 | vague accuracy | misroute cost / 1,000 requests |",
          "|---|---|---|---|---|"]
    res = []
    for col in [c for c in v.columns if c.startswith("rule::")]:
        pr = v[col].values
        acc = np.mean(pr == truth)
        res.append((col[6:], acc, f1_score(truth, pr, average="macro"), np.mean((pr == truth)[vm]),
                    (1 - acc) * 1000 * cost_per_miss))
    for name, acc, mf1, vacc, cost in sorted(res, key=lambda x: -x[1]):
        L.append(f"| {name} | {acc:.2%} | {mf1:.2%} | {vacc:.2%} | Rs {cost:,.0f} |")
    best_acc = max(res, key=lambda x: x[1])
    best_f1 = max(res, key=lambda x: x[2])
    L += ["", f"Best for accuracy and cost: **{best_acc[0]}**. Best for macro-F1: **{best_f1[0]}**.", ""]
    (ROOT / "out" / "operating_modes.md").write_text("\n".join(L) + "\n")
    v.drop(columns=["alts"]).to_csv(ROOT / "out" / "operating_modes_predictions.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
