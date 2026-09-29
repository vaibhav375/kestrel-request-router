"""Rupees and headcount, from the resolution log and the policy §4 cost rates.

Assumptions (all from the pack, none invented):
  * Transfer between teams: Rs 305 of handling time each (policy §4).
  * A misrouted request (opened in a team other than the one that closed it) costs one extra customer
    contact, Rs 260 on average (policy §4).
  * Bot licence: Rs 3.2 lakh a year (policy §4, Ritu's email).
  * Volume: the last 12 months of the export (Jul 2025 - Jun 2026). The Jul-Sep 2026 test month volume
    (726/month) matches it, so we treat the export as the full request flow.
"""
from __future__ import annotations

import pandas as pd

TRANSFER_RS = 305
EXTRA_CONTACT_RS = 260
LICENCE_RS = 320_000


def lakh(x: float) -> str:
    return f"Rs {x / 1e5:.1f} lakh"


def misroute_costs(d: pd.DataFrame) -> dict:
    last12 = d[d["created_at"] >= d["created_at"].max() - pd.DateOffset(months=12)]
    mis = last12["first_team_cur"] != last12["final_team_cur"]
    cost = last12["transfers"] * TRANSFER_RS + mis * EXTRA_CONTACT_RS
    return {
        "requests_12m": len(last12),
        "per_month": len(last12) / 12,
        "misroute_rate": mis.mean(),
        "transfers_12m": int(last12["transfers"].sum()),
        "cost_12m": float(cost.sum()),
        "cost_per_misroute": float(cost[mis].mean()),
        "hours_misrouted": float(last12.loc[mis, "hours_to_resolve"].median()),
        "hours_right_first": float(last12.loc[~mis, "hours_to_resolve"].median()),
    }


def team_volumes(d: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    recent = d[d["created_at"] >= d["created_at"].max() - pd.DateOffset(months=months)]
    v = pd.DataFrame({
        "bot_label_per_month": recent["bot_team"].value_counts() / months,
        "actual_per_month": recent["final_team_cur"].value_counts() / months,
    }).round(0)
    v["actual_share"] = (v["actual_per_month"] / v["actual_per_month"].sum()).round(3)
    v["bot_overstates_by"] = (v["bot_label_per_month"] - v["actual_per_month"]).astype(int)
    return v.sort_values("actual_per_month", ascending=False)


def report_lines(d: pd.DataFrame, ok_router_rate: float, ok_bot_rate: float, clear_acc: float,
                 vague_share: float) -> list[str]:
    c = misroute_costs(d)
    n = c["requests_12m"]
    router_mis = 1 - ok_router_rate
    bot_mis = 1 - ok_bot_rate
    saved_routing = (bot_mis - router_mis) * n * c["cost_per_misroute"]
    clarified_mis = 1 - clear_acc
    saved_clarified = (bot_mis - clarified_mis) * n * c["cost_per_misroute"]
    v = team_volumes(d)
    v.to_csv(Path_out() / "team_volumes.csv")
    L = ["## 7. What it is worth, and who is busiest", "",
         f"**Misrouting today** (last 12 months, {n:,} requests, ~{c['per_month']:.0f} a month): "
         f"{pct(c['misroute_rate'])} opened in the wrong team, {c['transfers_12m']:,} transfers. At Rs {TRANSFER_RS} per "
         f"transfer and Rs {EXTRA_CONTACT_RS} per extra contact that is **{lakh(c['cost_12m'])} a year**, about "
         f"Rs {c['cost_per_misroute']:.0f} per misrouted request. Misrouted requests also take twice as long to close "
         f"(median {c['hours_misrouted']:.1f} h vs {c['hours_right_first']:.1f} h).", "",
         "| scenario | wrong first team | cost of misrouting / year | licence | total vs today |", "|---|---|---|---|---|",
         f"| Today: vendor bot | {pct(bot_mis)} | {lakh(bot_mis * n * c['cost_per_misroute'])} | {lakh(LICENCE_RS)} | – |",
         f"| Router, bot switched off | {pct(router_mis)} | {lakh(router_mis * n * c['cost_per_misroute'])} | Rs 0 | "
         f"**saves {lakh(saved_routing + LICENCE_RS)}** |",
         f"| Router + one intake question for vague requests (projection) | ~{pct(clarified_mis)} | "
         f"{lakh(clarified_mis * n * c['cost_per_misroute'])} | Rs 0 | saves up to {lakh(saved_clarified + LICENCE_RS)} |", "",
         "Error rates are the out-of-time rates from section 1 applied to 12 months of volume; cost per misroute is the "
         "historical average (it includes that request's own transfers). Run cost of the router: **Rs 0 per request** – "
         "no API, no per-request fee; it answers in under 1 ms on a laptop CPU.", "",
         "**Who is busiest** (last 6 months, requests per month). Planning headcount on the bot's labels would over-staff "
         "Repairs, Billing and Filters & Consumables and under-staff Installs, Returns and Warranty:", "",
         "| team | bot label says | actually closed | share | bot over/under-states by |", "|---|---|---|---|---|"]
    for t, r in v.iterrows():
        L.append(f"| {t} | {r['bot_label_per_month']:.0f} | {r['actual_per_month']:.0f} | {pct(r['actual_share'])} | "
                 f"{int(r['bot_overstates_by']):+d} |")
    L.append("")
    return L


def pct(x: float) -> str:
    return f"{x:.1%}"


def Path_out():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "out"
    p.mkdir(exist_ok=True)
    return p
