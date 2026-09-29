"""Protocol rule 3: 5-fold CV and held-out wording for the champion and the challengers that beat it in the backtest.

    python experiments/challenger_robustness.py C1 C3      # challenger ids to check (fast ones only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments import challengers as C  # noqa: E402
from experiments.compare_models import build_frame  # noqa: E402
import kestrel.router as KR  # noqa: E402


def predict(cid, tr, va, E):
    base, kinds = C.champion(tr, va)
    fns = {
        "C0": lambda: base,
        "C1": lambda: C.c1_product(tr, va, base, kinds),
        "C2": lambda: C.c2_recent(tr, va, base, kinds),
        "C3": lambda: C.c3_product_warranty(tr, va, base, kinds),
        "C4a": lambda: C.champion(tr, va, 1)[0],
        "C4b": lambda: C.champion(tr, va, 10)[0],
        "C5": lambda: C.c5_gbdt(tr, va),
        "C8": lambda: C.c8_noise_cleaned(tr, va)[0],
        "C9": lambda: C.c9_knn(tr, va, E),
    }
    return fns[cid]()


def main(ids):
    d = build_frame()
    E = C.embeddings(d) if "C9" in ids else None
    ids = ["C0"] + [i for i in ids if i != "C0"]
    rows = []
    schemes = {
        "5-fold CV": list(StratifiedKFold(5, shuffle=True, random_state=0).split(d, d["final_team_cur"])),
        "held-out wording": list(GroupKFold(5).split(d, d["final_team_cur"], groups=d["last"])),
    }
    for scheme, splits in schemes.items():
        # held-out wording: use the fair teams.csv-only rules, as in evaluate.py §3
        rules_backup = list(KR.KEYWORD_RULES)
        if scheme == "held-out wording":
            KR.KEYWORD_RULES[:] = KR.TEAMS_ONLY_RULES
        try:
            for cid in ids:
                accs = [np.mean(predict(cid, d.iloc[a], d.iloc[b], E) == d.iloc[b]["final_team_cur"].values)
                        for a, b in splits]
                rows.append({"scheme": scheme, "model": cid, "acc": np.mean(accs), "sd": np.std(accs, ddof=1)})
                print(rows[-1], flush=True)
        finally:
            KR.KEYWORD_RULES[:] = rules_backup
    res = pd.DataFrame(rows).pivot(index="model", columns="scheme", values="acc")
    champ = res.loc["C0"]
    lines = ["# Rule 3 check: 5-fold CV and held-out wording", "",
             "| model | 5-fold CV | vs champion | held-out wording (fair rules) | vs champion |", "|---|---|---|---|---|"]
    for m, r in res.iterrows():
        lines.append(f"| {m} | {r['5-fold CV']:.2%} | {100 * (r['5-fold CV'] - champ['5-fold CV']):+.2f} pts | "
                     f"{r['held-out wording']:.2%} | {100 * (r['held-out wording'] - champ['held-out wording']):+.2f} pts |")
    (ROOT / "out" / "challenger_robustness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
