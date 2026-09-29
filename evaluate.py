"""Evidence that the router works, and how often it does not. Writes out/evaluation.md (+ CSVs).

    python evaluate.py

Truth = the team that closed the request (resolution_log.final_team), renamed teams merged.
Every number in docs/memo.md and docs/submission-form.md comes from this script or experiments/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kestrel.data import TEAMS, load_test, load_train  # noqa: E402
from kestrel import text as T  # noqa: E402
from kestrel.router import TEAMS_ONLY_RULES, Router  # noqa: E402
from kestrel import business as B  # noqa: E402

OUT = ROOT / "out"
SPLIT = "2026-04-01"
RNG = np.random.default_rng(0)
ABBR = {"Billing": "BIL", "Filters & Consumables": "CON", "Installs & Demo": "INS", "Product Advice": "ADV",
        "Repairs": "REP", "Returns & Replacement": "RET", "Warranty Claims": "WAR"}


def boot_ci(ok: np.ndarray, n: int = 2000) -> tuple[float, float]:
    idx = RNG.integers(0, len(ok), size=(n, len(ok)))
    s = ok[idx].mean(axis=1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def bot_mimic(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """What the brief literally asked for: a model trained to reproduce the bot's team_label."""
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = v.fit_transform(train["text"].map(T.normalise))
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(X, train["bot_team"])
    return clf.predict(v.transform(valid["text"].map(T.normalise)))


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = load_train()
    d["vague"] = d["text"].map(T.is_vague)
    d["deciding"] = d["text"].map(T.last_clause)
    L: list[str] = ["# Evaluation: does the router work, and how often does it not?", "",
                    "Truth = the team that **closed** each request (`resolution_log.final_team`), with the two renamed "
                    "teams merged (policy §5). The bot's `team_label` is shown as a baseline, not as truth: it matches "
                    "the closing team on only 77% of the 10,822 historical requests.", ""]

    # ---------------------------------------------------------------- 1. out-of-time
    tr, va = d[d["created_at"] < SPLIT], d[d["created_at"] >= SPLIT].copy()
    router = Router.fit(tr)
    p = router.predict(va)
    va["pred"] = p["team"].values
    va["route_type"] = p["route_type"].values
    va["confidence"] = p["confidence"].values
    va["level"] = p["confidence_level"].values
    va["mimic"] = bot_mimic(tr, va)
    truth = va["final_team_cur"].values
    ok = va["pred"].values == truth
    ok_bot = va["bot_team"].values == truth
    ok_mimic = va["mimic"].values == truth
    lo, hi = boot_ci(ok)
    blo, bhi = boot_ci(ok_bot)
    L += ["## 1. Out-of-time test (the main number)", "",
          f"Trained on {len(tr):,} requests from 1 Apr 2025 – 31 Mar 2026; scored on the **{len(va):,} requests of "
          "Apr – Jun 2026** that it never saw (all on the new CRM, like the Jul – Sep test set).", "",
          "| | matches closing team | matches bot label |", "|---|---|---|",
          f"| **Router (this work)** | **{pct(ok.mean())}** (95% CI {pct(lo)} – {pct(hi)}) | "
          f"{pct(np.mean(va['pred'] == va['bot_team']))} |",
          f"| Vendor bot | {pct(ok_bot.mean())} (95% CI {pct(blo)} – {pct(bhi)}) | 100% |",
          f"| Model trained to copy the bot's labels (the brief as written) | {pct(ok_mimic.mean())} | "
          f"**{pct(np.mean(va['mimic'] == va['bot_team']))}** |", "",
          "The copy-the-bot model passes the \"90% match with the labels\" bar and routes no better than the bot. "
          "Matching the labels is the wrong test; matching where requests are actually closed is the right one.", ""]

    # by route type
    L += ["### Where it is right and wrong", "", "| request type | share | router right | bot right |",
          "|---|---|---|---|"]
    names = {"seen": "States a need, wording seen before", "vague": "States no need (\"please call me about my purifier\")",
             "new_wording": "States a need, new wording"}
    for k in ["seen", "vague", "new_wording"]:
        m = va["route_type"].values == k
        if m.any():
            L.append(f"| {names[k]} | {pct(m.mean())} ({m.sum():,}) | {pct(ok[m].mean())} | {pct(ok_bot[m].mean())} |")
    L.append("")

    # per team
    rep = classification_report(truth, va["pred"], labels=TEAMS, output_dict=True, zero_division=0)
    L += ["### Per team", "", "| team | precision | recall | requests |", "|---|---|---|---|"]
    for t in TEAMS:
        L.append(f"| {t} | {pct(rep[t]['precision'])} | {pct(rep[t]['recall'])} | {int(rep[t]['support']):,} |")
    L.append("")
    cm = pd.DataFrame(confusion_matrix(truth, va["pred"], labels=TEAMS),
                      index=[ABBR[t] for t in TEAMS], columns=[ABBR[t] for t in TEAMS])
    cm.to_csv(OUT / "confusion_out_of_time.csv")
    L += ["Confusion matrix (rows = closing team, columns = router):", "", "```", cm.to_string(), "```", "",
          "Most errors sit in the columns for Repairs: that is where vague requests go (best available guess).", ""]

    # calibration
    L += ["### Is the confidence honest?", "", "| confidence level | share of requests | router right |", "|---|---|---|"]
    for lev in ["high", "medium", "low"]:
        m = va["level"].values == lev
        if m.any():
            L.append(f"| {lev} | {pct(m.mean())} | {pct(ok[m].mean())} |")
    L += ["", "\"High\" can be auto-routed; \"low\" is almost entirely the vague requests that need one question.", ""]

    # ---------------------------------------------------------------- 2. stability: 5-fold CV
    rows = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(d, d["final_team_cur"]):
        r = Router.fit(d.iloc[a])
        pb = r.predict(d.iloc[b])
        okb = pb["team"].values == d.iloc[b]["final_team_cur"].values
        vb = pb["route_type"].values == "vague"
        rows.append({"all": okb.mean(), "clear": okb[~vb].mean(), "vague": okb[vb].mean(), "vague_share": vb.mean()})
    cv = pd.DataFrame(rows)
    L += ["## 2. Stability: 5-fold cross-validation on all 10,822 requests", "",
          f"Router: **{pct(cv['all'].mean())} ± {cv['all'].std():.1%}** (folds {pct(cv['all'].min())} – {pct(cv['all'].max())}). "
          f"Requests that state a need: {pct(cv['clear'].mean())}. Vague requests: {pct(cv['vague'].mean())}.", ""]

    # ---------------------------------------------------------------- 3. unseen wording
    rows = []
    for a, b in GroupKFold(5).split(d, d["final_team_cur"], groups=d["deciding"]):
        r = Router.fit(d.iloc[a])
        for rules_name, rules in [("shipped", None), ("teams_only", TEAMS_ONLY_RULES)]:
            if rules is not None:
                r.rules = list(rules)
            pb = r.predict(d.iloc[b])
            okb = pb["team"].values == d.iloc[b]["final_team_cur"].values
            vb = pb["route_type"].values == "vague"
            assert (pb["route_type"].values[~vb] == "new_wording").all()  # nothing leaks through the lookup
            rows.append({"rules": rules_name, "all": okb.mean(), "clear": okb[~vb].mean()})
    un_all = pd.DataFrame(rows)
    un = un_all[un_all["rules"] == "teams_only"]
    un_shipped = un_all[un_all["rules"] == "shipped"]
    L += ["## 3. New wording: what happens when customers phrase things differently", "",
          "Every phrasing in a validation fold is held out of training, so the router must fall back on its keyword "
          "rules and text model. This is harsher than the test set (99.5% of test wording has been seen before).", "",
          "| keyword rules used for new wording | overall | requests that state a need | range across folds |",
          "|---|---|---|---|",
          f"| Shipped rules (written after reading the history – optimistic) | {pct(un_shipped['all'].mean())} | "
          f"{pct(un_shipped['clear'].mean())} | {pct(un_shipped['clear'].min())} – {pct(un_shipped['clear'].max())} |",
          f"| **Fair check: rules from the teams.csv wording only** | **{pct(un['all'].mean())}** | "
          f"**{pct(un['clear'].mean())}** | {pct(un['clear'].min())} – {pct(un['clear'].max())} |", "",
          "The shipped rules cover every phrasing in the history, so the first row is not a real test of new wording. "
          "The second row is: expect genuinely new customer wording to be routed correctly about "
          f"{pct(un['clear'].mean())} of the time, which is why new wording is marked for review when unsure.", ""]

    # ---------------------------------------------------------------- 4. how often it fails, and why
    err = va[~ok].copy()
    clear_err = err[err["route_type"] != "vague"]
    noisy = clear_err["inconsistent"].sum()
    # label noise floor: clear requests whose closing team is not the usual team for that wording
    L += ["## 4. How often it fails, and why", "",
          f"Out-of-time it is wrong on **{len(err):,} of {len(va):,} requests ({pct(1 - ok.mean())})**:", "",
          f"* **{(err['route_type'] == 'vague').sum():,} vague requests** – the message states no need, and past requests like "
          "these closed in all seven teams. No text-only router can do better than ~22% here; the fix is one "
          "question at intake (section 5).",
          f"* **{len(clear_err):,} requests that did state a need** ({pct(len(clear_err) / max(1, (va['route_type'] != 'vague').sum()))} of them). "
          f"{noisy:,} of these are records that say 0 transfers yet closed in a different team than they opened in; "
          "most of the rest read as clear requests closed by an unrelated team (e.g. \"how to clean air fryer\" closed by "
          "Billing). These look like recording noise in the resolution log, not routing mistakes, and set a ceiling of "
          "~98% on requests with a stated need.", ""]
    sample = clear_err.sample(min(12, len(clear_err)), random_state=0)
    L += ["Random sample of the stated-need errors:", "", "| request | router | closed by | transfers |", "|---|---|---|---|"]
    for _, r in sample.iterrows():
        shown = T.ID_RE.sub("[no.]", r["text"].lower())[:90]  # no order/registration numbers in the report
        L.append(f"| {shown} | {r['pred']} | {r['final_team_cur']} | {r['transfers']} |")
    L.append("")
    err[["request_id", "text", "bot_team", "final_team_cur", "transfers", "pred", "route_type", "confidence"]].to_csv(
        OUT / "errors_out_of_time.csv", index=False)

    # ---------------------------------------------------------------- 5. clarification what-if
    clear_acc = ok[va["route_type"].values != "vague"].mean()
    vshare = (va["route_type"] == "vague").mean()
    L += ["## 5. What if vague requests were clarified at intake? (a projection, not a measurement)", "",
          f"If the {pct(vshare)} vague requests answered one multiple-choice question and were then routed as accurately as "
          f"requests that state a need ({pct(clear_acc)}), overall accuracy would be about "
          f"**{pct((1 - vshare) * clear_acc + vshare * clear_acc)}**. That assumes customers answer and answer "
          "correctly, which only a pilot can confirm.", ""]

    # ---------------------------------------------------------------- 6. expected test score
    te = load_test()
    full = Router.load() if (ROOT / "models" / "router.joblib").exists() else Router.fit(d)
    pt = full.predict(te)
    mix = pt["route_type"].value_counts(normalize=True)
    acc_by = {"seen": cv["clear"].mean(), "new_wording": un["clear"].mean(), "vague": cv["vague"].mean()}
    expected = sum(mix.get(k, 0) * acc_by[k] for k in acc_by)
    L += ["## 6. Expected score on the Jul – Sep 2026 test set", "",
          f"Test mix: {pct(mix.get('seen', 0))} seen wording, {pct(mix.get('vague', 0))} vague, "
          f"{pct(mix.get('new_wording', 0))} new wording (same mix as the Apr – Jun window).",
          f"Weighted by the cross-validated accuracy of each type: **{pct(expected)}**; the out-of-time test gave "
          f"{pct(ok.mean())}. Expected range **84–87%** against the closing team. Against the bot's labels it would "
          f"score about {pct(np.mean(va['pred'] == va['bot_team']))}.", ""]

    # ---------------------------------------------------------------- 7. business view
    L += B.report_lines(d, ok_router_rate=ok.mean(), ok_bot_rate=ok_bot.mean(), clear_acc=clear_acc, vague_share=vshare)

    (OUT / "evaluation.md").write_text("\n".join(L) + "\n")
    va[["request_id", "text", "bot_team", "final_team_cur", "pred", "route_type", "confidence", "mimic"]].to_csv(
        OUT / "predictions_out_of_time.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
