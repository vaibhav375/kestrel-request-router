"""The request router: one record in, a team plus reasons a Kestrel agent can read out.

How it decides (each step was chosen by the experiments in experiments/, see docs/FINDINGS.md):
  1. Find the customer's requests in the message, ignoring greetings, order numbers, sign-offs and
     payment remarks ("paid on upi" does not make a request Billing - policy §3).
  2. If the message has two requests, the LAST one decides: that is how 99% of past two-request
     messages were actually closed.
  3. If that request states no need ("please call back regarding my purifier"), nobody can route it
     from the text: past outcomes are spread over all seven teams. We give the most common outcome
     (Repairs) but flag it and suggest one clarifying question.
  4. If the wording has been seen before (>= MIN_SUPPORT past requests), use the team that closed
     most of those requests. This covers 99.9% of the Jul-Sep 2026 test set.
  5. New wording: keyword rules built from teams.csv plus the phrasings in the history; if no rule
     fires, a TF-IDF + logistic regression model trained on the same history.

Trained on where requests were actually closed (resolution_log.final_team), not on the bot's
label: the bot's label matched the closing team only 77% of the time.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import text as T
from .data import TEAMS, fix_mojibake, load_teams
from .repair import HINGLISH, repair

MODEL_VERSION = "2026-10-01.1"
MIN_SUPPORT = 3
PARAPHRASES_PATH = Path(__file__).resolve().parent / "resources" / "paraphrases.csv"
# Round 4 switches (docs/ROUND4_PROTOCOL.md). ALL_FEATURES lists every switch; DEFAULT_FEATURES the adopted ones.
ALL_FEATURES = frozenset({"A1", "B1", "B2", "B3", "B4", "C2"})
DEFAULT_FEATURES = frozenset({"A1", "B4"})  # adopted in round 4 (docs/ROUND4_RESULTS.md); C1 flag is always on
FUZZY_MIN_SIM = 0.60
VAGUE_ALPHA = 50
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "router.joblib"

# Keyword rules for wording never seen before. First match wins, checked on the deciding request.
# Words come from teams.csv 'handles' and from the phrasings customers used in the history.
KEYWORD_RULES: list[tuple[str, str, str]] = [
    ("Installs & Demo", r"install|demo\b|wall.?mount", "installation / demo / wall-mounting"),
    ("Returns & Replacement",
     r"arrived damaged|damaged|missing parts|incomplete|wrong model|wrong item|received used|return|exchange|"
     r"box was open|scratched|need replacement$|replace (?:the|my) P",
     "damaged, wrong or incomplete delivery, or a return/exchange"),
    ("Warranty Claims", r"warranty|shield|claim|guarantee|covered|coverage|registered|registration",
     "warranty or Kestrel Shield registration, coverage or claim"),
    ("Billing",
     r"invoice|gst|charged twice|double charge|charged|refund|emi conversion|emi\b|coupon|discount|"
     r"payment deducted|payment failed|\bbill\b",
     "a problem with the payment itself (invoice, GST, double charge, refund, EMI, coupon)"),
    ("Filters & Consumables",
     r"filter|candle|membrane|\bjar\b|brush|blade|amc|spare|consumable|cartridge",
     "buying or fitting filters, spares or consumables"),
    ("Repairs",
     r"not working|not turning|stopped working|won.?t start|dead|leak|noise|noisy|burnt|burning|smell|tripping|"
     r"mcb|error|e\d\b|gone blank|display|fault|broken|repair|overheat|spark|shock|not heating|not cooling|"
     r"crack|peel|rust|stuck|jammed",
     "a product fault or breakdown"),
    ("Product Advice",
     r"how to|how do|safe for|recipe|power consumption|best settings|which P|difference between|run on inverter|"
     r"can i|can P|should i|usage|use the",
     "a usage or buying question with no fault reported"),
]

# Fair-comparison rule set: words taken only from teams.csv 'handles' (used by evaluate.py to measure how
# the router copes with new wording without the benefit of rules written after reading the history).
TEAMS_ONLY_RULES: list[tuple[str, str, str]] = [
    ("Installs & Demo", r"install|demo|wall.?mount", "installation / demo / wall-mounting"),
    ("Returns & Replacement", r"damaged|wrong|incomplete|missing|return|exchange|replace(?!ment filter)",
     "damaged, wrong or incomplete delivery, or a return/exchange"),
    ("Warranty Claims", r"warranty|shield|registration|register|coverage|covered|claim", "warranty or Shield"),
    ("Billing", r"invoice|gst|double|charged|refund|payment|emi|coupon|bill", "a payment problem"),
    ("Filters & Consumables", r"filter|candle|membrane|jar|brush|blade|amc|spare|consumable", "filters or spares"),
    ("Repairs", r"fault|breakdown|error|noise|leak|not working|broken|repair|fix", "a product fault"),
    ("Product Advice", r"how|which|what|can |is P|usage|use|difference|best|safe", "a usage question"),
]

CLARIFY_OPTIONS = [
    ("A fault or breakdown (not working, noise, leak, error code)", "Repairs"),
    ("A damaged, wrong or incomplete delivery, or a return/exchange", "Returns & Replacement"),
    ("Installation, demo or wall-mounting", "Installs & Demo"),
    ("Warranty or Kestrel Shield", "Warranty Claims"),
    ("A payment problem: invoice, GST, double charge, refund, EMI, coupon", "Billing"),
    ("Filters, spares or consumables", "Filters & Consumables"),
    ("How to use or choose the product", "Product Advice"),
]
CLARIFY_QUESTION = "So we can send you to the right team, is this about:"


def _share(k: int, n: int) -> str:
    """Percent that never rounds up to 100% (or down to 0%) unless it really is."""
    p = k / n
    if 0.995 <= p < 1:
        return ">99%"
    if 0 < p < 0.005:
        return "<1%"
    return f"{p:.0%}"


@dataclass
class Router:
    version: str = MODEL_VERSION
    table: dict = field(default_factory=dict)          # masked deciding request -> Counter(closing team)
    vague_counts: Counter = field(default_factory=Counter)
    two_request_stats: tuple = (0, 0)                   # (closed by last request's team, total)
    payment_stats: tuple = (0, 0, 0)                    # (bot sent to Billing, closed by Billing, total)
    team_handles: dict = field(default_factory=dict)
    trained_rows: int = 0
    trained_until: str = ""
    rules: list = field(default_factory=lambda: list(KEYWORD_RULES))
    features: frozenset = DEFAULT_FEATURES
    vague_by_product: dict = field(default_factory=dict)  # A1/C2: product -> Counter(closing team) of vague requests
    vocab: set = field(default_factory=set)               # B4: known words for spelling correction
    para_table: dict = field(default_factory=dict)        # B3: paraphrase (masked) -> (team, source phrasing)
    _fuzzy_vec: TfidfVectorizer | None = None             # B1/B2: char n-grams over known phrasings
    _fuzzy_X = None
    _fuzzy_keys: list = field(default_factory=list)
    _vec_full: TfidfVectorizer | None = None
    _vec_last: TfidfVectorizer | None = None
    _clf: LogisticRegression | None = None

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, df: pd.DataFrame, target: str = "final_team_cur", features=None,
            paraphrases: pd.DataFrame | None = None) -> "Router":
        r = cls()
        r.features = frozenset(DEFAULT_FEATURES if features is None else features)
        df = df.copy()
        df["_norm"] = df["text"].map(T.normalise)
        df["_clauses"] = df["text"].map(T.clauses)
        df["_last"] = df["_clauses"].str[-1]
        df["_vague"] = df["text"].map(T.is_vague)
        y = df[target]

        clear = df[~df["_vague"]]
        r.vague_counts = Counter(y[df["_vague"]])
        for k, g in y[clear.index].groupby(clear["_last"]):
            if len(g) >= MIN_SUPPORT:
                r.table[k] = Counter(g)

        # Fallback text model (only used for wording never seen before).
        r._vec_full = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        r._vec_last = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        X = hstack([r._vec_full.fit_transform(clear["_norm"]), r._vec_last.fit_transform(clear["_last"]) * 1.5])
        r._clf = LogisticRegression(C=4.0, max_iter=3000).fit(X, y[clear.index])

        # Evidence quoted in reasons, measured on this training data.
        two = clear[clear["_clauses"].map(len) >= 2]
        first_team = two["_clauses"].map(lambda c: r._majority(c[0]))
        last_team = two["_last"].map(r._majority)
        distinct = first_team.notna() & last_team.notna() & (first_team != last_team)
        r.two_request_stats = (int((y[two.index][distinct] == last_team[distinct]).sum()), int(distinct.sum()))
        pay = df[df["text"].map(T.has_payment_remark) & ~df["_clauses"].map(lambda c: r._majority(c[-1]) == "Billing")]
        bot_billing = int((pay["bot_team"] == "Billing").sum()) if "bot_team" in pay else 0
        r.payment_stats = (bot_billing, int((y[pay.index] == "Billing").sum()), len(pay))

        # A1 / C2: vague outcomes per product (always stored; used only when the switch is on).
        if "product_family" in df:
            vg = df[df["_vague"]]
            for prod, g in y[vg.index].groupby(vg["product_family"]):
                r.vague_by_product[prod] = Counter(g)
        # B4: vocabulary of known words (training requests, product words, dictionary outputs).
        words = set(" ".join(df["text"].str.lower()).split()) | set(" ".join(rep for _, rep in HINGLISH).split())
        r.vocab = {w for w in words if w.isalpha()}
        # B3: paraphrases of known phrasings -> team of the source phrasing (only sources known to this training set).
        if paraphrases is None and "B3" in r.features and PARAPHRASES_PATH.exists():
            paraphrases = pd.read_csv(PARAPHRASES_PATH)
        if paraphrases is not None and "B3" in r.features:
            for src, para in zip(paraphrases["source"], paraphrases["paraphrase"]):
                team = r._majority(src)
                key = T.last_clause(str(para))
                if team and key and key not in r.table and not T.is_vague(str(para)):
                    r.para_table.setdefault(key, (team, src))
        # B1 / B2: character n-gram index over every known phrasing (and paraphrases when B3 is on).
        keys = list(r.table) + list(r.para_table)
        if keys and ({"B1", "B2"} & r.features):
            r._fuzzy_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            r._fuzzy_X = r._fuzzy_vec.fit_transform(keys)
            r._fuzzy_keys = keys

        teams = load_teams()
        for _, t in teams.iterrows():
            name = t["renamed_to"].split(" (")[0] if isinstance(t["renamed_to"], str) else t["team"]
            r.team_handles[name] = t["handles"]
        r.trained_rows = len(df)
        r.trained_until = str(df["created_at"].max().date()) if "created_at" in df else ""
        return r

    def _majority(self, clause: str):
        c = self.table.get(clause)
        return c.most_common(1)[0][0] if c else None

    def _key_team(self, key: str):
        return self._majority(key) or (self.para_table[key][0] if key in self.para_table else None)

    def _nearest(self, last: str):
        """B1/B2: closest known phrasing by character n-grams -> (phrasing, team, cosine similarity)."""
        if self._fuzzy_vec is None or not last:
            return None, None, 0.0
        sims = (self._fuzzy_X @ self._fuzzy_vec.transform([last]).T).toarray().ravel()
        i = int(sims.argmax())
        key = self._fuzzy_keys[i]
        return key, self._key_team(key), float(sims[i])

    def _vague_posterior(self, product) -> pd.Series:
        """A1/C2: P(team | vague, product), shrunk towards the pooled vague distribution."""
        pooled = pd.Series(self.vague_counts, dtype=float).reindex(TEAMS, fill_value=0)
        prior = pooled / pooled.sum()
        c = pd.Series(self.vague_by_product.get(product, {}), dtype=float).reindex(TEAMS, fill_value=0)
        return (c + VAGUE_ALPHA * prior) / (c.sum() + VAGUE_ALPHA)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = DEFAULT_MODEL_PATH) -> "Router":
        return joblib.load(path)

    # ------------------------------------------------------------------ prediction
    def _fallback(self, norm: str, last: str) -> tuple[str, float, list[tuple[str, float]]]:
        X = hstack([self._vec_full.transform([norm]), self._vec_last.transform([last]) * 1.5])
        p = self._clf.predict_proba(X)[0]
        order = np.argsort(p)[::-1]
        classes = self._clf.classes_
        return classes[order[0]], float(p[order[0]]), [(classes[i], float(p[i])) for i in order[:3]]

    def route(self, record: dict) -> dict:
        raw = str(record.get("request_text") or "")
        product = record.get("product_family")
        f = self.features
        text = fix_mojibake(raw)
        cl = T.clauses(text)
        cl_show = T.clauses(text, mask_products=False)
        last, last_show = cl[-1], cl_show[-1]
        vague = T.is_vague(text)
        reasons: list[str] = []
        needs_clarification = False
        alternatives: list[dict] = []
        detail = ""

        # B4: a message whose wording is unknown is repaired (Hinglish -> English, spelling) and re-read.
        repaired_note = None
        if "B4" in f and text.strip() and not vague and last not in self.table:
            fixed = repair(text, self.vocab)
            if fixed != text.lower():
                cl2 = T.clauses(fixed)
                if T.is_vague(fixed) or cl2[-1] in self.table or cl2[-1] in self.para_table:
                    cl, last, vague = cl2, cl2[-1], T.is_vague(fixed)
                    repaired_note = fixed
                elif cl2[-1]:
                    last = cl2[-1]  # keep the repaired wording for the later steps
                    repaired_note = fixed

        if not text.strip() or not last:
            team, conf, kind = self.vague_counts.most_common(1)[0][0], 0.0, "empty"
            needs_clarification = True
            reasons.append("The message is empty, so there is nothing to route on. Ask the customer what they need.")
        elif vague:
            kind = "vague"
            total = sum(self.vague_counts.values())
            team, n = self.vague_counts.most_common(1)[0]
            conf = n / total
            needs_clarification = True
            alternatives = [{"team": t, "share": round(c / total, 3)} for t, c in self.vague_counts.most_common(3)]
            reasons.append(f"The customer does not say what the problem is (“{last_show}”).")
            if "A1" in f and product in self.vague_by_product:
                post = self._vague_posterior(product)
                team, conf = post.idxmax(), float(post.max())
                n_p = sum(self.vague_by_product[product].values())
                alternatives = [{"team": t, "share": round(float(v), 3)} for t, v in post.nlargest(3).items()]
                reasons.append(
                    f"Best guess for a vague {product} request: {team} (about {conf:.0%}, from {n_p:,} past vague "
                    f"{product} requests). This is a guess: it was not proven better than the overall most common team "
                    f"({self.vague_counts.most_common(1)[0][0]}), and no one can route this reliably from the text.")
            else:
                reasons.append(
                    f"Past requests like this ({total:,}) were closed by all seven teams; the most common was "
                    f"{team} at only {conf:.0%}. Nobody can route this reliably from the text.")
            reasons.append("Ask the one question below before assigning, or let the first agent re-route after the call.")
        elif last in self.table:
            kind, detail = "seen", "repaired" if repaired_note else "exact"
            c = self.table[last]
            n = sum(c.values())
            team, k = c.most_common(1)[0]
            conf = k / n
            alternatives = [{"team": t, "share": round(v / n, 3)} for t, v in c.most_common(3)]
            reasons.append(f"Customer asks: “{last_show}”.")
            if repaired_note:
                reasons.append(f"Read as “{last.replace('P', 'product')}” after correcting spelling / Hinglish.")
            reasons.append(f"{n:,} past requests with this wording: {_share(k, n)} were closed by {team}.")
        else:
            kind = "new_wording"
            reasons.append(f"Customer asks: “{last_show}”.")
            if repaired_note:
                reasons.append(f"Read as “{last.replace('P', 'product')}” after correcting spelling / Hinglish.")
            fb_team, fb_conf, fb_alts = self._fallback(T.normalise(repaired_note or text), last)
            alternatives = [{"team": t, "share": round(p, 3)} for t, p in fb_alts]
            near_key, near_team, near_sim = self._nearest(last) if ({"B1", "B2"} & f) else (None, None, 0.0)
            hits = [(t, label) for t, rx, label in self.rules if re.search(rx, last)]
            rule_team, rule_label = hits[0] if hits else (None, None)
            other_teams = [t for t, _ in hits if t != rule_team]

            if "B3" in f and last in self.para_table:
                team, src = self.para_table[last]
                conf, detail = 0.85, "paraphrase"
                reasons.append(f"Same meaning as the known wording “{src.replace('P', 'product')}”, which {team} closes.")
            elif "B1" in f and near_key and near_sim >= FUZZY_MIN_SIM:
                team, conf, detail = near_team, 0.8, "similar"
                reasons.append(f"Closest known wording: “{near_key.replace('P', 'product')}” "
                               f"(similarity {near_sim:.2f}), usually closed by {team}.")
            elif "B2" in f:
                votes = [t for t in (rule_team, near_team, fb_team) if t]
                tally = Counter(votes).most_common()
                if tally and tally[0][1] >= 2:
                    team, conf, detail = tally[0][0], (0.85 if tally[0][1] == 3 else 0.75), "vote"
                    reasons.append(f"This wording is new. {tally[0][1]} of 3 checks agree on {team} "
                                   f"(keyword rules: {rule_team or 'no match'}; closest known wording: {near_team}; "
                                   f"text model: {fb_team}).")
                else:
                    team = rule_team or near_team or fb_team
                    conf, detail = 0.55, "vote_split"
                    reasons.append(f"This wording is new and the checks disagree (keyword rules: {rule_team or 'no match'}; "
                                   f"closest known wording: {near_team}; text model: {fb_team}). Check with the customer.")
            elif rule_team and not other_teams:
                team, conf, detail = rule_team, 0.8, "rules"
                reasons.append(f"This wording is new to the router. It reads as {rule_label}, which {team} handles.")
            elif rule_team:
                team, conf, detail = rule_team, 0.6, "rules_conflict"
                reasons.append(f"This wording is new to the router and reads two ways: as {rule_label} ({team}) "
                               f"or as a matter for {', '.join(dict.fromkeys(other_teams))}. Check with the customer.")
            else:
                team, conf, detail = fb_team, min(fb_conf, 0.6), "model"
                reasons.append("This wording is new to the router and no keyword matched; this is the text model's best guess.")
            if conf < 0.7:
                needs_clarification = True

        if len(cl) >= 2 and kind in ("seen", "new_wording"):
            k, n = self.two_request_stats
            share = f"{k / n:.0%} of {n:,}" if n else "most"
            first_show = cl_show[0] if len(cl_show) >= 2 else cl[0]
            reasons.append(
                f"The message has {len(cl)} requests (“{first_show}”, then “{last_show}”). "
                f"The last one decides: that is how {share} past two-request messages were closed. "
                f"Mention the earlier request to the {team} agent.")
        elif len(cl) >= 2 and kind == "vague":
            reasons.append(f"An earlier part of the message mentions “{cl_show[0]}”, but past outcomes show "
                           "that does not predict the closing team once the message ends vaguely.")

        remark = T.payment_remark(text)
        if remark and team != "Billing":
            bot_b, closed_b, n_pay = self.payment_stats
            hist = (f" The old bot sent {bot_b:,} of {n_pay:,} such past requests to Billing; Billing closed "
                    f"{closed_b:,} of them." if n_pay else "")
            reasons.append(f"Mentions “{remark}”. Saying they paid does not make it a Billing request "
                           f"(ops policy §3).{hist}")

        if team in self.team_handles and kind != "vague":
            reasons.append(f"{team} handles: {self.team_handles[team]}")

        clarify = None
        if needs_clarification:
            options = [{"label": l, "team": t, "likely": False} for l, t in CLARIFY_OPTIONS]
            if "C2" in f and product:
                post = self._vague_posterior(product)
                options.sort(key=lambda o: -post[o["team"]])
                for o in options[:3]:
                    o["likely"] = True
            clarify = {"question": CLARIFY_QUESTION, "options": options}

        level = "high" if conf >= 0.9 and not needs_clarification else ("medium" if conf >= 0.7 else "low")
        return {
            "request_id": record.get("request_id"),
            "team": team,
            "confidence": round(float(conf), 3),
            "confidence_level": level,
            "auto_route": bool(conf >= 0.7 and not needs_clarification),   # C1
            "route_type": kind,
            "route_detail": detail,
            "needs_clarification": needs_clarification,
            "clarifying_question": clarify,
            "alternatives": alternatives,
            "alternatives_basis": {"seen": "past requests with the same wording",
                                   "vague": "past requests that stated no need",
                                   "new_wording": "text model estimate (wording not seen before)"}.get(kind, ""),
            "reasons": reasons,
            "deciding_request": last_show,
            "model_version": self.version,
        }

    def predict(self, df: pd.DataFrame, text_col: str = "request_text") -> pd.DataFrame:
        has_prod = "product_family" in df
        out = [self.route({"request_id": getattr(r, "request_id", None), "request_text": getattr(r, text_col),
                           "product_family": getattr(r, "product_family", None) if has_prod else None})
               for r in df.itertuples()]
        return pd.DataFrame(out)

assert set(t for t, _, _ in KEYWORD_RULES) == set(TEAMS)
