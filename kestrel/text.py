"""Text normalisation shared by the model, the experiments and the reasons shown to agents.

What the data shows (scratch analysis, see docs/FINDINGS.md):
  * A request is a greeting + one or two "intent" phrases + filler (order/reg numbers, "pls call back",
    "thanks") and sometimes a payment remark ("paid on upi", "already paid in full").
  * Payment remarks do not change the team (policy §3). The bot sends them to Billing anyway.
  * When there are two intents, the request is resolved by the team of the LAST one (98.7% of 550 cases).
    So the model gets the last clause as a separate, position-aware input.
"""
from __future__ import annotations

import re

PRODUCT_WORDS = (
    r"water purifier|purifier|air fryer|fryer|mixer grinder|mixer|grinder|robot vacuum|vacuum|"
    r"induction cooktop|cooktop|room heater|heater|ceiling fan|fan|machine|product"
)
PRODUCT_RE = re.compile(rf"\b({PRODUCT_WORDS})\b")
ID_RE = re.compile(r"\b(order\s+ko\d+|reg\s+no\s+sr\d+|ko\d{5,}|sr\d{4,})\b")
GREETING_RE = re.compile(r"^\s*(hi|hello team|hello|good morning|namaste|sir|urgent:|pls help\s*-?|please help\s*-?)[\s,:-]*")
CLOSING = r"thanks|thank you|asap|pls call back|kindly resolve|very disappointed"
CLOSING_RE = re.compile(rf"\b({CLOSING})(?=\s|$|,)|\.{{2,}}|[-\u2013]+\s*$")
PAYMENT = r"paid on upi|already paid in full|payment done|paid by emi|paid via card|i paid extra for this"
PAYMENT_RE = re.compile(rf"\b({PAYMENT})\b")
VAGUE_RE = re.compile(
    r"^(someone contact me about P|complaint about P|not happy with P|help P|service request for P|"
    r"please call back regarding P|need help with my P|P problem|P query|issue with P)$"
)


def normalise(text: str, mask_products: bool = True) -> str:
    """Lower-case, mask ids and product names. Keeps greetings/closings (the model can ignore them)."""
    s = (text or "").lower()
    s = ID_RE.sub(" ", s)
    if mask_products:
        s = PRODUCT_RE.sub("P", s)
    return re.sub(r"\s+", " ", s).strip()


def core(text: str, mask_products: bool = True) -> str:
    """The request with greeting, ids, closings and payment remarks removed: only the intent phrases."""
    s = normalise(text, mask_products)
    s = GREETING_RE.sub("", s)
    s = PAYMENT_RE.sub(" ", s)
    s = CLOSING_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s).strip(" ,")
    return re.sub(r"(, )+", ", ", s)


# Second halves of single intents that contain a comma ("P arrived damaged, need replacement").
_CONTINUATIONS = {"need replacement", "want exchange", "why", "P scratched"}


def _split(text: str, mask_products: bool) -> list[str]:
    return [p.strip() for p in core(text, mask_products).split(",") if p.strip()]


def clauses(text: str, mask_products: bool = True) -> list[str]:
    """Split the core into intent clauses, re-joining the known two-part intents.

    With mask_products=False the same clauses are returned with the product words kept (for display).
    """
    masked = _split(text, True)
    parts = masked if mask_products else _split(text, False)
    if len(parts) != len(masked):  # cannot happen (product words hold no commas); stay safe
        parts = masked
    out: list[str] = []
    for m, p in zip(masked, parts):
        if out and m in _CONTINUATIONS:
            out[-1] = f"{out[-1]}, {p}"
        else:
            out.append(p)
    return out or [""]


def last_clause(text: str) -> str:
    return clauses(text)[-1]


def has_payment_remark(text: str) -> bool:
    return bool(PAYMENT_RE.search((text or "").lower()))


def payment_remark(text: str) -> str | None:
    m = PAYMENT_RE.search((text or "").lower())
    return m.group(1) if m else None


def is_vague(text: str) -> bool:
    """The deciding (last) clause states no need: "please call back regarding my purifier", "P query".

    Outcomes for these are spread over all seven teams even when an earlier clause is specific
    (120 train cases: the earlier clause's team was the closing team only 15% of the time).
    """
    return bool(VAGUE_RE.match(clauses(text)[-1]))
