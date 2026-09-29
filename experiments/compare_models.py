"""Compare routing approaches on three validation schemes. Writes out/experiments.md.

Truth = final_team from resolution_log (the team that actually closed the request), current names.
Also reported: agreement with the bot's team_label, which is what the brief literally asked for.

  A. Out-of-time: train on created < 2026-04-01, score on Apr-Jun 2026 (closest to the Jul-Sep test).
  B. 5-fold stratified CV on all 10,822 rows (stability).
  C. Unseen wording: GroupKFold by the request's last clause, so every validation phrasing is new.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kestrel.data import load_train  # noqa: E402
from kestrel import text as T  # noqa: E402

SEED = 0


def build_frame():
    d = load_train()
    d["norm"] = d["text"].map(T.normalise)
    d["core"] = d["text"].map(T.core)
    d["last"] = d["text"].map(T.last_clause)
    d["vague"] = d["text"].map(T.is_vague)
    return d


# ---------- model factories: each returns fit(train_df, y) -> predict(df) ----------

class TfidfModel:
    def __init__(self, clf="lr", last=False, meta=False, char=False, field="norm", C=4.0):
        self.clf_name, self.last, self.meta, self.char, self.field, self.C = clf, last, meta, char, field, C

    def _feats(self, df, fit):
        mats = []
        if fit:
            self.v_full = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            mats.append(self.v_full.fit_transform(df[self.field]))
        else:
            mats.append(self.v_full.transform(df[self.field]))
        if self.char:
            if fit:
                self.v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
                mats.append(self.v_char.fit_transform(df[self.field]))
            else:
                mats.append(self.v_char.transform(df[self.field]))
        if self.last:
            if fit:
                self.v_last = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
                mats.append(self.v_last.fit_transform(df["last"]) * 1.5)
            else:
                mats.append(self.v_last.transform(df["last"]) * 1.5)
        if self.meta:
            cols = ["channel", "product_family", "warranty_status"]
            if fit:
                self.ohe = OneHotEncoder(handle_unknown="ignore")
                mats.append(self.ohe.fit_transform(df[cols]))
            else:
                mats.append(self.ohe.transform(df[cols]))
        return hstack(mats).tocsr()

    def fit(self, df, y):
        X = self._feats(df, True)
        if self.clf_name == "lr":
            self.clf = LogisticRegression(C=self.C, max_iter=3000)
        elif self.clf_name == "svm":
            self.clf = LinearSVC(C=0.5)
        elif self.clf_name == "nb":
            self.clf = ComplementNB(alpha=0.3)
        self.clf.fit(X, y)
        return self

    def predict(self, df):
        return self.clf.predict(self._feats(df, False))


class LookupModel:
    """Majority final team per last clause; fall back to a TF-IDF model for unseen clauses."""

    def fit(self, df, y):
        t = pd.DataFrame({"k": df["last"].values, "y": np.asarray(y)})
        self.table = t.groupby("k")["y"].agg(lambda s: s.value_counts().index[0]).to_dict()
        self.fallback = TfidfModel(last=True).fit(df, y)
        return self

    def predict(self, df):
        fb = self.fallback.predict(df)
        return np.array([self.table.get(k, f) for k, f in zip(df["last"], fb)])


class MajorityModel:
    def fit(self, df, y):
        self.c = pd.Series(y).value_counts().index[0]
        return self

    def predict(self, df):
        return np.array([self.c] * len(df))


class RulesModel:
    """Hand-written keyword rules on the last clause, written from teams.csv + policy only (no training).

    Tests whether a transparent rule list is enough. Order matters: first match wins.
    """
    RULES = [
        ("Installs & Demo", r"install|demo|wall mounting|installer"),
        ("Returns & Replacement", r"arrived damaged|missing parts|wrong model|received used|return|exchange|box was open|scratched|replacement$|need replacement"),
        ("Warranty Claims", r"warranty|shield|claim status|why$"),
        ("Billing", r"invoice|gst|charged twice|refund|emi conversion|coupon|payment deducted|bill\b"),
        ("Filters & Consumables", r"filter|candle|membrane|jar|brush roll|blade|amc|consumable"),
        ("Product Advice", r"how to|safe for kids|recipe|power consumption|best settings|which P|difference between|run on inverter"),
        ("Repairs", r"not working|not turning|leaking|noise|burnt|tripping|error code|gone blank|stopped working"),
    ]

    def fit(self, df, y):
        import re
        self.rx = [(t, re.compile(p)) for t, p in self.RULES]
        return self

    def predict(self, df):
        out = []
        for s in df["last"]:
            team = "Repairs"  # most common outcome, also the best guess for vague requests
            for t, r in self.rx:
                if r.search(s):
                    team = t
                    break
            out.append(team)
        return np.array(out)


class EmbedModel:
    """Frozen bge-small sentence embeddings (local, no API) of full text + last clause -> logistic regression."""
    _cache: dict = {}

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        if "m" not in EmbedModel._cache:
            EmbedModel._cache["m"] = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        self.m = EmbedModel._cache["m"]

    def _emb(self, texts):
        key = tuple(texts)
        if key not in EmbedModel._cache:
            EmbedModel._cache[key] = self.m.encode(list(texts), batch_size=128, normalize_embeddings=True,
                                                   show_progress_bar=False)
        return EmbedModel._cache[key]

    def _feats(self, df):
        return np.hstack([self._emb(df["text"].str.lower().tolist()), self._emb(df["last"].tolist())])

    def fit(self, df, y):
        self.clf = LogisticRegression(C=10, max_iter=3000).fit(self._feats(df), y)
        return self

    def predict(self, df):
        return self.clf.predict(self._feats(df))


def candidates(with_embed: bool):
    c = {
        "Majority class (always Repairs)": (MajorityModel, "final"),
        "Rules on last clause (no training)": (RulesModel, "final"),
        "TF-IDF+LR trained on BOT labels (as briefed)": (lambda: TfidfModel(), "bot"),
        "TF-IDF+LR on outcomes": (lambda: TfidfModel(), "final"),
        "TF-IDF+NB on outcomes": (lambda: TfidfModel(clf="nb"), "final"),
        "TF-IDF+SVM on outcomes": (lambda: TfidfModel(clf="svm"), "final"),
        "TF-IDF word+char+LR on outcomes": (lambda: TfidfModel(char=True), "final"),
        "TF-IDF+last-clause+LR on outcomes": (lambda: TfidfModel(last=True), "final"),
        "TF-IDF+last-clause+meta+LR on outcomes": (lambda: TfidfModel(last=True, meta=True), "final"),
        "TF-IDF+last-clause+LR, clean rows only": (lambda: TfidfModel(last=True), "final_clean"),
        "TF-IDF+last-clause+SVM on outcomes": (lambda: TfidfModel(clf="svm", last=True), "final"),
        "Lookup on last clause (+TF-IDF fallback)": (LookupModel, "final"),
    }
    if with_embed:
        c["bge-small embeddings+LR on outcomes"] = (EmbedModel, "final")
    return c


def run_split(d, tr_idx, va_idx, factory, target):
    tr, va = d.iloc[tr_idx], d.iloc[va_idx]
    if target == "bot":
        y = tr["bot_team"]
    elif target == "final_clean":
        tr = tr[~tr["inconsistent"]]
        y = tr["final_team_cur"]
    else:
        y = tr["final_team_cur"]
    m = factory().fit(tr, y.values)
    p = m.predict(va)
    return p


def score(p, va):
    return {
        "vs_outcome": float(np.mean(p == va["final_team_cur"].values)),
        "vs_bot": float(np.mean(p == va["bot_team"].values)),
        "vs_outcome_clear": float(np.mean((p == va["final_team_cur"].values)[~va["vague"].values])),
        "vs_outcome_vague": float(np.mean((p == va["final_team_cur"].values)[va["vague"].values])),
    }


def main(with_embed=True):
    d = build_frame()
    rows = []
    # A. out-of-time
    tr_idx = np.where(d["created_at"] < "2026-04-01")[0]
    va_idx = np.where(d["created_at"] >= "2026-04-01")[0]
    va = d.iloc[va_idx]
    bot = {"model": "Vendor bot (team_label as the prediction)", **score(va["bot_team"].values, va)}
    rows.append({"scheme": "A out-of-time", **bot, "secs": 0})
    for name, (f, tgt) in candidates(with_embed).items():
        t0 = time.time()
        p = run_split(d, tr_idx, va_idx, f, tgt)
        rows.append({"scheme": "A out-of-time", "model": name, **score(p, va), "secs": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)

    # B. 5-fold stratified CV, C. unseen wording (GroupKFold by last clause)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, d["final_team_cur"])
    gkf = GroupKFold(5).split(d, d["final_team_cur"], groups=d["last"])
    for scheme, splits in [("B 5-fold CV", list(skf)), ("C unseen wording", list(gkf))]:
        for name, (f, tgt) in candidates(with_embed).items():
            if scheme.startswith("C") and name.startswith("Lookup"):
                pass  # lookup degenerates to its fallback here; still informative
            accs = []
            t0 = time.time()
            for a, b in splits:
                accs.append(score(run_split(d, a, b, f, tgt), d.iloc[b]))
            m = pd.DataFrame(accs)
            rows.append({"scheme": scheme, "model": name, **m.mean().to_dict(),
                         "sd_outcome": m["vs_outcome"].std(), "secs": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
        # bot on the same folds
        rows.append({"scheme": scheme, "model": "Vendor bot (team_label as the prediction)",
                     **score(d["bot_team"].values, d), "secs": 0})

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "out" / "experiments.csv", index=False)
    lines = ["# Model comparison", "",
             "Truth = team that closed the request (resolution_log.final_team, renames merged).",
             "`vs_outcome_clear` = requests with a stated need; `vs_outcome_vague` = 'please call me about my X' requests.", ""]
    for scheme, g in res.groupby("scheme", sort=False):
        lines += [f"## {scheme}", "", "| model | vs outcome | vs bot label | clear requests | vague requests | sd | secs |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values("vs_outcome", ascending=False).iterrows():
            sd = f"{r['sd_outcome']:.3f}" if "sd_outcome" in r and pd.notna(r.get("sd_outcome")) else ""
            lines.append(f"| {r['model']} | {r['vs_outcome']:.3f} | {r['vs_bot']:.3f} | {r['vs_outcome_clear']:.3f} | "
                         f"{r['vs_outcome_vague']:.3f} | {sd} | {r['secs']} |")
        lines.append("")
    (ROOT / "out" / "experiments.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(with_embed="--no-embed" not in sys.argv)
