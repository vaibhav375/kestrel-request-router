"""B4: repair messages before routing - Hinglish phrases to English, then spelling correction.

Written from general knowledge of how Indian customers type, BEFORE the synthetic Hinglish test set existed
(docs/ROUND4_PROTOCOL.md), so it is not tuned to that test. Applied only when the original wording is not
already known, so it can never change a request the router already routes from history.
"""
from __future__ import annotations

import difflib
import re

# Longest phrases first so "kaam nahi kar raha" wins over "nahi". Right side is plain English the router knows.
HINGLISH: list[tuple[str, str]] = [
    # faults
    (r"kaam nahi kar (?:raha|rahi|rahe)|kaam nahi karta|kaam nahi karti|chal nahi (?:raha|rahi)|nahi chal (?:raha|rahi)|"
     r"band (?:ho|pad) (?:gaya|gayi)|kharab ho (?:gaya|gayi)|kharab (?:hai|h)|kharab", "not working"),
    (r"(?:on|chalu|start) nahi ho (?:raha|rahi)|(?:on|chalu|start) nahi hota|(?:on|chalu) hi nahi", "not turning on"),
    (r"(?:paani|pani) (?:tapak|nikal|leak) (?:raha|rahi)(?: hai)?|leak ho (?:raha|rahi)|tapak (?:raha|rahi)", "leaking water"),
    (r"(?:bahut )?(?:awaaz|awaz|aawaz|avaaz|shor)(?: kar (?:raha|rahi))?", "making loud noise"),
    (r"jal (?:gaya|gayi)|jalne ki (?:smell|badbu|boo)|badbu", "burnt smell"),
    (r"garam nahi (?:ho )?(?:raha|rahi|hota)?", "not heating"),
    (r"display (?:band|gayab|off)|screen (?:band|gayab|off)", "display gone blank"),
    (r"mcb (?:gir|trip) (?:raha|rahi|jata|jati)|mcb trip ho", "tripping the mcb"),
    # delivery / returns
    (r"toot(?:a|i)? (?:hua|huyi|gaya|gayi)?|tut (?:gaya|gayi)|toota|tuta", "damaged"),
    (r"galat (?:model|product|item|saman|samaan)", "wrong model"),
    (r"(?:dabba|box|packet) (?:khula|khuli|khula hua)", "box was open"),
    (r"(?:parts?|saman|samaan) (?:missing|kam|nahi (?:tha|the|hai))", "missing parts in box"),
    (r"purana|use kiya hua|istemal kiya hua|used wala", "used"),
    (r"wapas (?:karna|karni|lena|le lo|kar do)|wapis|lautana|return karna", "return"),
    (r"badal (?:do|dijiye|na)|badalna|exchange karna", "want exchange"),
    # installation
    (r"(?:technician|mistri|banda|installer) nahi aaya", "installer did not turn up"),
    (r"lagana|lagwana|lagwani|fit karna|fitting|install karna|installation karna", "installation"),
    # billing
    (r"(?:bill|invoice) nahi mila", "invoice not received"),
    (r"(?:do|2) baar (?:paise|paisa|payment) (?:kat|cut)|double (?:kat|cut|payment)", "charged twice"),
    (r"paise (?:kat|cut) (?:gaye|gaya) (?:par|lekin) order nahi", "payment deducted but order not placed"),
    (r"(?:refund|paise|paisa) (?:wapas )?nahi (?:aaya|aaye|mila)", "refund not credited"),
    # warranty
    (r"guarantee|gaurantee|waranty|warenty", "warranty"),
    # advice
    (r"(?:kaise|kese) (?:saaf|clean) (?:kare|karein|karu|karna)|saaf kaise", "how to clean"),
    (r"(?:kaise|kese) (?:use|chalaye|chalayein|chalana|chalaun)", "how to use"),
    (r"kaun ?sa|konsa|kaunsa", "which"),
    (r"(?:bijli|light|current) kitni", "power consumption"),
    (r"(?:bacchon|bachon|bachhon) ke liye safe", "safe for kids"),
    # asks for contact only (vague)
    (r"(?:call|phone) (?:karo|kar do|kijiye|karna)|baat karni hai|contact karo", "please call back regarding"),
    (r"chahiye|chaiye", "want"),
]
_HINGLISH_RE = [(re.compile(rf"\b(?:{p})\b"), rep) for p, rep in HINGLISH]

# Hindi function words: never spelling-corrected (they would be "fixed" into unrelated English words).
KEEP = {"mera", "meri", "mere", "mujhe", "hai", "hain", "tha", "thi", "bhi", "aur", "kya", "nahi", "koi", "yeh",
        "woh", "abhi", "please", "bhai", "ji", "sir", "madam", "jaldi", "kab", "kyun", "kyon", "wala", "wali"}
WORD_RE = re.compile(r"[a-z]+")


def hinglish_to_english(text: str) -> str:
    s = (text or "").lower()
    for rx, rep in _HINGLISH_RE:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s).strip()


def spell_correct(text: str, vocab: set[str], cutoff: float = 0.8) -> str:
    """Replace unknown words (4+ letters) with the closest known word, if one is close enough."""
    vocab_list = sorted(vocab)

    def fix(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or w in vocab or w in KEEP:
            return w
        best = difflib.get_close_matches(w, vocab_list, n=1, cutoff=cutoff)
        return best[0] if best else w

    return WORD_RE.sub(fix, (text or "").lower())


def repair(text: str, vocab: set[str]) -> str:
    return spell_correct(hinglish_to_english(text), vocab)
