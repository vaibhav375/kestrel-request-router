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

MODEL_VERSION = "2026-09-30.1"
MIN_SUPPORT = 3
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
    _vec_full: TfidfVectorizer | None = None
    _vec_last: TfidfVectorizer | None = None
    _clf: LogisticRegression | None = None

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, df: pd.DataFrame, target: str = "final_team_cur") -> "Router":
        r = cls()
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
        text = fix_mojibake(raw)
        cl = T.clauses(text)
        cl_show = T.clauses(text, mask_products=False)
        last, last_show = cl[-1], cl_show[-1]
        vague = T.is_vague(text)
        reasons: list[str] = []
        needs_clarification = False
        alternatives: list[dict] = []

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
            reasons.append(
                f"Past requests like this ({total:,}) were closed by all seven teams; the most common was "
                f"{team} at only {conf:.0%}. Nobody can route this reliably from the text.")
            reasons.append("Ask the one question below before assigning, or let the first agent re-route after the call.")
        elif last in self.table:
            kind = "seen"
            c = self.table[last]
            n = sum(c.values())
            team, k = c.most_common(1)[0]
            conf = k / n
            alternatives = [{"team": t, "share": round(v / n, 3)} for t, v in c.most_common(3)]
            reasons.append(f"Customer asks: “{last_show}”.")
            reasons.append(f"{n:,} past requests with this wording: {_share(k, n)} were closed by {team}.")
        else:
            kind = "new_wording"
            team, conf, rule_label = None, 0.0, None
            hits = [(t, label) for t, rx, label in self.rules if re.search(rx, last)]
            if hits:
                team, rule_label = hits[0]  # rule order decides
            other_teams = [t for t, _ in hits if t != team]
            fb_team, fb_conf, fb_alts = self._fallback(T.normalise(text), last)
            alternatives = [{"team": t, "share": round(p, 3)} for t, p in fb_alts]
            reasons.append(f"Customer asks: “{last_show}”.")
            if team and not other_teams:
                conf = 0.8  # fair new-wording test (evaluate.py §3): keyword rules right on ~87% of stated needs
                reasons.append(f"This wording is new to the router. It reads as {rule_label}, which {team} handles.")
            elif team:
                conf = 0.6
                reasons.append(f"This wording is new to the router and reads two ways: as {rule_label} ({team}) "
                               f"or as a matter for {', '.join(dict.fromkeys(other_teams))}. Check with the customer.")
            else:
                team, conf = fb_team, min(fb_conf, 0.6)
                reasons.append("This wording is new to the router and no keyword matched; this is the text model's best guess.")
            if conf < 0.7:
                needs_clarification = True

        if len(cl) >= 2 and kind in ("seen", "new_wording"):
            k, n = self.two_request_stats
            share = f"{k / n:.0%} of {n:,}" if n else "most"
            reasons.append(
                f"The message has {len(cl)} requests (“{cl_show[0]}”, then “{last_show}”). "
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

        level = "high" if conf >= 0.9 and not needs_clarification else ("medium" if conf >= 0.7 else "low")
        return {
            "request_id": record.get("request_id"),
            "team": team,
            "confidence": round(float(conf), 3),
            "confidence_level": level,
            "route_type": kind,
            "needs_clarification": needs_clarification,
            "clarifying_question": ({"question": CLARIFY_QUESTION,
                                     "options": [{"label": l, "team": t} for l, t in CLARIFY_OPTIONS]}
                                    if needs_clarification else None),
            "alternatives": alternatives,
            "alternatives_basis": {"seen": "past requests with the same wording",
                                   "vague": "past requests that stated no need",
                                   "new_wording": "text model estimate (wording not seen before)"}.get(kind, ""),
            "reasons": reasons,
            "deciding_request": last_show,
            "model_version": self.version,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = [self.route({"request_id": r.request_id, "request_text": r.request_text}) for r in df.itertuples()]
        return pd.DataFrame(out)


assert set(t for t, _, _ in KEYWORD_RULES) == set(TEAMS)
