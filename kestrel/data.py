"""Loading and cleaning the Kestrel data pack.

Every data fix lives here, with the evidence for it:
  * Team renames (policy §5): Installations -> Installs & Demo, Consumables -> Filters & Consumables
    from 15 Jan 2026. Same responsibilities, so old and new names are one team.
  * Legacy Zoho text (email from Tanmay): ~11% of legacy rows have mojibake ("Ã¢â‚¬Â¦", "urgÃ©nt").
  * Legacy Zoho resolved_at is UTC, not IST (policy §9). Only matters for resolution-time figures.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Current team names (the test period is Jul-Sep 2026, after the 15 Jan 2026 rename).
TEAMS = [
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims",
]
RENAMES = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
RENAME_DATE = pd.Timestamp("2026-01-15")


def current_team(name: str) -> str:
    return RENAMES.get(name, name)


def fix_mojibake(s: str) -> str:
    """Undo the UTF-8-read-as-cp1252 damage on legacy Zoho text, then drop stray accents."""
    if not isinstance(s, str):
        return ""
    s = s.replace("Ã¢â‚¬Â¦", "...").replace("Ã¢â‚¬â€œ", "-")  # double-encoded ellipsis / en dash
    try:
        s = s.encode("cp1252").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = s.replace("…", "...").replace("–", "-").replace("—", "-")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _find(pattern: str, data_dir: Path) -> Path:
    """Pack files may keep their <uuid>-name.csv download names."""
    exact = data_dir / pattern
    if exact.exists():
        return exact
    hits = sorted(data_dir.glob(f"*{pattern}"))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found in {data_dir}")
    return hits[0]


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """train.csv joined to resolution_log.csv, with current team names and cleaned text."""
    tr = pd.read_csv(_find("train.csv", data_dir))
    rl = pd.read_csv(_find("resolution_log.csv", data_dir))
    df = tr.merge(rl, on="request_id", how="left", validate="one_to_one")
    df["created_at"] = pd.to_datetime(df["created_at_ist"])
    df["bot_team"] = df["team_label"].map(current_team)
    df["first_team_cur"] = df["first_team"].map(current_team)
    df["final_team_cur"] = df["final_team"].map(current_team)
    df["text"] = df["request_text"].map(fix_mojibake)
    # Policy §9: legacy Zoho resolution events are UTC; convert to IST (+5:30).
    res = pd.to_datetime(df["resolved_at"])
    shift = pd.to_timedelta((df["source"] == "legacy_zoho").astype(int) * 330, unit="min")
    df["resolved_at_ist"] = res + shift
    df["hours_to_resolve"] = (df["resolved_at_ist"] - df["created_at"]).dt.total_seconds() / 3600
    # A record that says 0 transfers but closed in a different team than it started in is
    # internally inconsistent; we keep it for evaluation but can exclude it from training.
    df["inconsistent"] = (df["transfers"] == 0) & (df["first_team_cur"] != df["final_team_cur"])
    return df


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    te = pd.read_csv(_find("test_unlabelled.csv", data_dir))
    te["created_at"] = pd.to_datetime(te["created_at_ist"])
    te["text"] = te["request_text"].map(fix_mojibake)
    return te


def load_teams(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(_find("teams.csv", data_dir))
