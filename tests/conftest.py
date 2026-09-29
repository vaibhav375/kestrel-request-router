"""Tests use a small synthetic history so they run without the (confidential) data pack.

Tests marked `pack` need the real data in data/ and are skipped when it is absent.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kestrel.router import Router  # noqa: E402

PHRASES = {
    "Repairs": ["{p} leaking water from bottom", "motor of {p} burnt smell", "{p} not turning on"],
    "Billing": ["invoice not received for {p}", "charged twice for my {p} order"],
    "Returns & Replacement": ["{p} arrived damaged, need replacement", "wrong model of {p} delivered"],
    "Installs & Demo": ["installer did not turn up for {p}", "want to reschedule installation of {p}"],
    "Warranty Claims": ["need warranty certificate {p}", "is {p} covered under shield plan"],
    "Filters & Consumables": ["need replacement filter for {p}", "want AMC filter kit for {p}"],
    "Product Advice": ["how to clean {p}", "is {p} safe for kids"],
}
VAGUE = ["please call back regarding {p}", "issue with {p}", "{p} query"]
PRODUCTS = ["water purifier", "air fryer", "ceiling fan"]


def synthetic_history() -> pd.DataFrame:
    rows, i = [], 0
    greetings = ["hi, ", "namaste, ", "", "urgent: "]
    for team, phrases in PHRASES.items():
        for ph in phrases:
            for p in PRODUCTS:
                for g in greetings:
                    text = g + ph.format(p=p)
                    bot = "Billing" if i % 5 == 0 else team
                    rows.append((f"SR{i:06d}", text + (" paid on upi" if i % 5 == 0 else ""), team, bot))
                    i += 1
    vague_teams = ["Repairs", "Repairs", "Installs & Demo", "Billing", "Warranty Claims"]
    for ph in VAGUE:
        for p in PRODUCTS:
            for k, t in enumerate(vague_teams):
                rows.append((f"SR{i:06d}", ph.format(p=p), t, "Repairs"))
                i += 1
    df = pd.DataFrame(rows, columns=["request_id", "request_text", "final_team_cur", "bot_team"])
    df["text"] = df["request_text"]
    df["created_at"] = pd.Timestamp("2026-01-01")
    return df


@pytest.fixture(scope="session")
def router() -> Router:
    return Router.fit(synthetic_history())


def pytest_configure(config):
    config.addinivalue_line("markers", "pack: needs the real Kestrel data pack in data/")


@pytest.fixture(scope="session")
def pack_available() -> bool:
    return any((ROOT / "data").glob("*train.csv"))
