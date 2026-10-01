"""Visual report (text + charts) for the video and for Ritu's team.

    python -m kestrel.report        -> out/report.html  (plotly is embedded, so it works offline)

Numbers for the router and the bot are computed live from data/. The testing-round charts read the saved outputs of
experiments/ (out/challengers.csv, out/round4_summary.csv, out/llm_benchmark_*.csv); if one is missing, that chart is
replaced by a note telling you which script to run.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from kestrel import business as B
from kestrel import text as T
from kestrel.data import load_train
from kestrel.router import Router

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
# Categorical slots validated with the dataviz validator (light surface #fcfcfb): blue / orange / aqua, worst adjacent
# CVD dE 9.2. Router = slot 1, bot = slot 2, copy-the-bot = slot 3. Gray is the de-emphasis neutral.
ROUTER, BOT, COPY, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7", "#1d1d1b", "#6b6a66"
SURFACE = "#fcfcfb"


def _fig(fig: go.Figure, height: int = 340, legend: bool = True) -> str:
    fig.update_layout(height=height, template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin=dict(l=10, r=20, t=10, b=10), font=dict(size=13, color=INK), bargap=0.35, bargroupgap=0.08,
                      showlegend=legend, legend=dict(orientation="h", y=1.08, x=0, font=dict(color=MUTED)),
                      hoverlabel=dict(bgcolor="white", font_size=13))
    fig.update_xaxes(showgrid=False, linecolor="#e3e1db", automargin=True)
    fig.update_yaxes(gridcolor="#eeede8", zeroline=False, automargin=True)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def _missing(script: str) -> str:
    return f'<p class="note">Chart not available: run <code>{html.escape(script)}</code> first.</p>'


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def compute():
    d = load_train()
    tr, va = d[d["created_at"] < "2026-04-01"], d[d["created_at"] >= "2026-04-01"].copy()
    r = Router.fit(tr)
    p = r.predict(va)
    va["pred"], va["kind"], va["level"], va["auto"] = (p["team"].values, p["route_type"].values,
                                                       p["confidence_level"].values, p["auto_route"].values)
    va["ok"] = va["pred"] == va["final_team_cur"]
    va["okb"] = va["bot_team"] == va["final_team_cur"]
    full = Router.fit(d)
    return d, va, full


# ---------------------------------------------------------------- charts
def chart_target(va) -> str:
    names = ["Router (this work)", "Vendor bot", "Copy of the bot's labels"]
    closing = [va["ok"].mean(), va["okb"].mean(), 0.770]
    labels = [(va["pred"] == va["bot_team"]).mean(), 1.0, 0.918]
    f = go.Figure([
        go.Bar(name="Sent to the team that closed it", x=names, y=[v * 100 for v in closing], marker_color=ROUTER,
               text=[_pct(v) for v in closing], textposition="outside", hovertemplate="%{x}: %{y:.1f}%<extra></extra>"),
        go.Bar(name="Agrees with the bot's label", x=names, y=[v * 100 for v in labels], marker_color=GRAY,
               text=["88.1%", "", "91.8%"], textposition="outside",
               hovertemplate="%{x}: %{y:.1f}%<extra></extra>")])
    f.add_annotation(x=names[2], y=104, text="passes the 90% ask, but routes like the bot", showarrow=False,
                     font=dict(color=MUTED, size=12))
    f.update_yaxes(title="% of the 2,135 Apr–Jun 2026 requests", range=[0, 112], ticksuffix="%")
    return _fig(f, 360)


def chart_bot_monthly(d) -> str:
    m = (d["bot_team"] == d["final_team_cur"]).groupby(d["created_at"].dt.to_period("M").astype(str)).mean() * 100
    f = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color=BOT, width=2),
                             marker=dict(size=8), name="Vendor bot", hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    f.add_annotation(x=m.index[-1], y=m.values[-1], text=f"{m.values[-1]:.0f}%", showarrow=False, xshift=22,
                     font=dict(color=MUTED))
    f.update_yaxes(title="% sent to the closing team", range=[50, 100], ticksuffix="%")
    return _fig(f, 300, legend=False)


def chart_bot_by_label(d) -> str:
    acc = ((d["bot_team"] == d["final_team_cur"]).groupby(d["bot_team"]).mean() * 100).sort_values()
    colors = [BOT if v < 80 else GRAY for v in acc.values]
    f = go.Figure(go.Bar(x=acc.values, y=acc.index, orientation="h", marker_color=colors,
                         text=[f"{v:.0f}%" for v in acc.values], textposition="outside",
                         hovertemplate="When the bot said %{y}: right %{x:.1f}%<extra></extra>"))
    f.update_xaxes(title="% of the bot's choices that were the closing team", range=[0, 108], ticksuffix="%")
    return _fig(f, 320, legend=False)


def chart_by_type(va) -> str:
    kinds = [("seen", "States a need"), ("vague", "States no need (vague)"), ("new_wording", "New wording")]
    share = {k: (va["kind"] == k).mean() for k, _ in kinds}
    x = [f"{lbl}<br><span style='font-size:11px'>{_pct(share[k], 1)} of requests</span>" for k, lbl in kinds]
    bars = []
    for name, col, color in [("Router", "ok", ROUTER), ("Vendor bot", "okb", BOT)]:
        ys = [va.loc[va["kind"] == k, col].mean() * 100 for k, _ in kinds]
        bars.append(go.Bar(name=name, x=x, y=ys, marker_color=color, text=[f"{v:.0f}%" for v in ys],
                           textposition="outside", hovertemplate=name + ": %{y:.1f}%<extra></extra>"))
    f = go.Figure(bars)
    f.update_yaxes(title="% right", range=[0, 112], ticksuffix="%")
    return _fig(f, 340)


def chart_errors(va) -> str:
    n = len(va)
    right = int(va["ok"].sum())
    vague_err = int((~va["ok"] & (va["kind"] == "vague")).sum())
    other_err = n - right - vague_err
    cats = ["Right", "Wrong: vague request (no need stated)", "Wrong: clear request (mostly log noise)"]
    vals = [right, vague_err, other_err]
    f = go.Figure(go.Bar(x=vals, y=cats, orientation="h", marker_color=[ROUTER, GRAY, GRAY],
                         text=[f"{v:,} ({v / n:.1%})" for v in vals], textposition="outside",
                         hovertemplate="%{y}: %{x:,}<extra></extra>"))
    f.update_xaxes(title="requests, Apr–Jun 2026", range=[0, n * 1.18])
    f.update_yaxes(autorange="reversed")
    return _fig(f, 240, legend=False)


def chart_confidence(va) -> str:
    rows = []
    for lev in ["high", "medium", "low"]:
        m = va["level"] == lev
        if m.any():
            rows.append((lev, m.mean(), va.loc[m, "ok"].mean()))
    f = go.Figure(go.Bar(x=[f"{l.capitalize()} confidence<br><span style='font-size:11px'>{s:.1%} of requests</span>"
                            for l, s, _ in rows], y=[a * 100 for _, _, a in rows], marker_color=ROUTER,
                         text=[f"{a:.0%} right" for _, _, a in rows], textposition="outside",
                         hovertemplate="%{x}<br>right: %{y:.1f}%<extra></extra>"))
    f.update_yaxes(title="% right", range=[0, 112], ticksuffix="%")
    return _fig(f, 300, legend=False)


def chart_challengers() -> str:
    path = OUT / "challengers.csv"
    if not path.exists():
        return _missing("python experiments/challengers.py")
    c = pd.read_csv(path).sort_values("pooled_acc")
    names = (c["model"].str.replace(r"^C\d+[ab]? ", "", regex=True)
             .str.replace("champion", "Router as of round 3")
             .replace({"vague prior by product": "vague guess by product (adopted in round 4)"}))
    is_champ = c["model"].str.startswith("C0") | c["model"].str.startswith("C1 ")
    f = go.Figure(go.Scatter(x=c["pooled_acc"] * 100, y=names, mode="markers",
                             marker=dict(size=[14 if b else 10 for b in is_champ],
                                         color=[ROUTER if b else GRAY for b in is_champ],
                                         line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    f.add_vline(x=86.79, line=dict(color=MUTED, width=1))
    f.add_annotation(x=86.79, y=1.02, yref="paper", text="ceiling 86.8%: an oracle that sees the answers",
                     showarrow=False, font=dict(color=MUTED, size=11), xanchor="right")
    f.update_xaxes(title="accuracy, rolling backtest (6,502 predictions)", range=[82.5, 87], ticksuffix="%")
    return _fig(f, 420, legend=False)


def chart_llm(full_d) -> str:
    path = OUT / "llm_benchmark_predictions.csv"
    tpath = OUT / "llm_benchmark_timing.csv"
    if not path.exists() or not tpath.exists():
        return _missing("python experiments/llm_benchmark.py")
    v = pd.read_csv(path)
    r = Router.fit(full_d[full_d["created_at"] < "2026-04-01"])
    router_ok = (r.predict(v.assign(request_text=v["text"]))["team"].values == v["final_team_cur"].values).mean()
    vals = [("Router", router_ok, ROUTER), ("Vendor bot", (v["bot_team"] == v["final_team_cur"]).mean(), BOT),
            ("Local LLM, zero-shot", (v["llm_zero-shot"] == v["final_team_cur"]).mean(), GRAY),
            ("Local LLM, with hints", (v["llm_hinted"] == v["final_team_cur"]).mean(), GRAY)]
    f = go.Figure(go.Bar(x=[n for n, _, _ in vals], y=[a * 100 for _, a, _ in vals], marker_color=[c for _, _, c in vals],
                         text=[_pct(a) for _, a, _ in vals], textposition="outside",
                         hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    f.update_yaxes(title="% right on the same 300 requests", range=[0, 100], ticksuffix="%")
    return _fig(f, 300, legend=False)


def chart_round4() -> tuple[str, str]:
    path = OUT / "round4_summary.csv"
    if not path.exists():
        return _missing("python experiments/round4.py"), _missing("python experiments/round4.py")
    s = pd.read_csv(path, index_col=0)
    before, b4 = s["R3 (round-3 router)"], s["B4 Hinglish + spelling repair"]
    tests = [("Heavy typos<br>(30% of words)", "E3"), ("Hinglish<br>(LLM-rewritten)", "E4")]
    f = go.Figure([go.Bar(name=name, x=[t for t, _ in tests], y=[float(col[k]) * 100 for _, k in tests],
                          marker_color=color, text=[_pct(float(col[k])) for _, k in tests], textposition="outside",
                          hovertemplate=name + ": %{y:.1f}%<extra></extra>")
                   for name, col, color in [("Before round 4", before, GRAY), ("With spelling + Hinglish repair", b4, ROUTER)]])
    f.update_yaxes(title="% right (synthetic stress tests)", range=[0, 112], ticksuffix="%")
    stress = _fig(f, 320)
    k = s["B5"].dropna()
    k = k[[i for i in k.index if str(i).isdigit()]]
    f2 = go.Figure(go.Scatter(x=[int(i) for i in k.index], y=k.values * 100, mode="lines+markers",
                              line=dict(color=ROUTER, width=2), marker=dict(size=9),
                              hovertemplate="after %{x} confirmed examples: %{y:.1f}%<extra></extra>"))
    f2.add_hline(y=98.39, line=dict(color=MUTED, width=1))
    f2.add_annotation(x=10, y=98.39, text="known-wording level 98.4%", showarrow=False, yshift=12, xanchor="right",
                      font=dict(color=MUTED, size=11))
    f2.update_xaxes(title="agent-confirmed examples of a new phrasing added at retrain", tickvals=[0, 1, 3, 5, 10])
    f2.update_yaxes(title="% right on its later requests", range=[85, 100], ticksuffix="%")
    return stress, _fig(f2, 300, legend=False)


def chart_money(d, va) -> tuple[str, dict]:
    c = B.misroute_costs(d)
    n, cpm = c["requests_12m"], c["cost_per_misroute"]
    rates = {"today": 1 - va["okb"].mean(), "router": 1 - va["ok"].mean(),
             "clarify": 1 - va.loc[va["kind"] != "vague", "ok"].mean()}
    sc = ["Today: vendor bot", "Router, bot switched off", "Router + intake question*"]
    mis = [rates[k] * n * cpm / 1e5 for k in ("today", "router", "clarify")]
    f = go.Figure([go.Bar(name="Cost of misrouting", x=sc, y=mis, marker_color=ROUTER,
                          hovertemplate="%{x}: Rs %{y:.1f} lakh<extra></extra>"),
                   go.Bar(name="Bot licence", x=sc, y=[3.2, 0, 0], marker_color=BOT,
                          hovertemplate="licence: Rs %{y:.1f} lakh<extra></extra>")])
    totals = [m + l for m, l in zip(mis, [3.2, 0, 0])]
    for x, t in zip(sc, totals):
        f.add_annotation(x=x, y=t, text=f"Rs {t:.1f} lakh", showarrow=False, yshift=12, font=dict(color=INK))
    f.update_layout(barmode="stack")
    f.update_yaxes(title="Rs lakh per year", range=[0, max(totals) * 1.18])
    saving = totals[0] - totals[1]
    return _fig(f, 340), {"cost": c, "saving": saving, "upside": totals[0] - totals[2]}


def chart_volumes(d) -> str:
    v = B.team_volumes(d).sort_values("actual_per_month")
    f = go.Figure([go.Bar(name="What the bot's labels say", y=v.index, x=v["bot_label_per_month"], orientation="h",
                          marker_color=BOT, hovertemplate="%{y}: %{x:.0f}/month (bot label)<extra></extra>"),
                   go.Bar(name="Actually closed by the team", y=v.index, x=v["actual_per_month"], orientation="h",
                          marker_color=ROUTER, hovertemplate="%{y}: %{x:.0f}/month (actual)<extra></extra>")])
    f.update_xaxes(title="requests per month, last 6 months")
    return _fig(f, 420)


# ---------------------------------------------------------------- page
CSS = """
:root { --bg:#f6f5f2; --panel:#fcfcfb; --ink:#1d1d1b; --muted:#6b6a66; --line:#e3e1db; --accent:#2a78d6; --bot:#eb6834; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 28px 20px 60px; }
header h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
header p.sub { color: var(--muted); margin: 0 0 22px; }
h2 { font-size: 21px; margin: 40px 0 8px; letter-spacing: -0.01em; }
h2 .n { color: var(--accent); margin-right: 8px; }
p { margin: 8px 0 12px; max-width: 74ch; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 8px 0 18px; }
.tile { background: var(--panel); border:1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.tile b { display:block; font-size: 30px; line-height:1.15; letter-spacing:-0.02em; }
.tile span { color: var(--muted); font-size: 14px; }
.card { background: var(--panel); border:1px solid var(--line); border-radius: 12px; padding: 14px 16px 6px; margin: 14px 0 20px; }
.card h3 { font-size: 15px; margin: 2px 0 2px; }
.card .cap { color: var(--muted); font-size: 13.5px; margin: 0 0 4px; }
.callout { border-left: 4px solid var(--accent); background: #eef4fb; padding: 12px 16px; border-radius: 0 10px 10px 0; margin: 14px 0; }
.callout.warn { border-color: var(--bot); background: #fdf1eb; }
ul, ol { margin: 6px 0 14px; padding-left: 22px; max-width: 74ch; } li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; margin: 8px 0 16px; background: var(--panel); }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
pre.flow { background: var(--panel); border:1px solid var(--line); border-radius: 12px; padding: 14px 16px; overflow-x:auto;
           font-size: 13.5px; line-height: 1.55; }
.note { color: var(--muted); font-style: italic; }
footer { color: var(--muted); font-size: 13px; margin-top: 40px; border-top:1px solid var(--line); padding-top: 12px; }
@media (max-width: 600px) { main { padding: 18px 16px 40px; } header h1 { font-size: 24px; } .tile b { font-size: 24px; } }
"""


def card(title: str, caption: str, chart: str) -> str:
    import re
    cid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]  # anchor, e.g. report.html#round-2-...
    return f'<div class="card" id="{cid}"><h3>{title}</h3><p class="cap">{caption}</p>{chart}</div>'


def main() -> None:
    from plotly.offline import get_plotlyjs
    d, va, full = compute()
    acc, bot = va["ok"].mean(), va["okb"].mean()
    fewer = 1 - (1 - acc) / (1 - bot)
    money_chart, money = chart_money(d, va)
    stress_chart, curve_chart = chart_round4()
    bot_b, closed_b, n_pay = full.payment_stats
    k2, n2 = full.two_request_stats
    vague_share = (va["kind"] == "vague").mean()
    clear_acc = va.loc[va["kind"] != "vague", "ok"].mean()
    c = money["cost"]

    body = f"""
<header>
  <h1>Kestrel Home: replacing the routing bot</h1>
  <p class="sub">Service-request router · results on {len(va):,} requests from Apr–Jun 2026 the model never saw ·
  truth = the team that actually closed each request</p>
</header>

<div class="tiles">
  <div class="tile"><b>{_pct(acc)}</b><span>sent to the right team (bot: {_pct(bot)})</span></div>
  <div class="tile"><b>{fewer:.0%}</b><span>fewer misrouted requests</span></div>
  <div class="tile"><b>Rs {money['saving']:.1f} lakh</b><span>saved per year, licence included</span></div>
  <div class="tile"><b>Rs 0</b><span>per request: no AI subscription, no per-request fee</span></div>
</div>

<p>Ritu asked for a model that matches the routing bot's labels at 90% or better. Those labels record where the bot
<em>sent</em> each request, not where it belonged. We trained the router on where requests were actually closed. It
routes <strong>{_pct(acc)}</strong> of new requests to the right team, against <strong>{_pct(bot)}</strong> for the bot,
and costs nothing to run.</p>

<h2><span class="n">1</span>The 90% target measures the wrong thing</h2>
<p>When we joined the export to the resolution log, the bot's label turned out to be the team that closed the request
only about <strong>77 times in 100</strong>. That figure barely moves from month to month. A model trained to copy those
labels reached 91.8% agreement, so it passes the 90% bar, but it routes no better than the bot because it learned the
bot's mistakes.</p>
{card("Matching the bot's labels is not the same as routing correctly",
      "Blue = sent to the team that closed it. Gray = agrees with the bot's label. The copy of the bot clears 90% on gray and is no better than the bot on blue.",
      chart_target(va))}
{card("The bot's accuracy, month by month", "Share of requests the bot sent to the team that finally closed them, Apr 2025 – Jun 2026.",
      chart_bot_monthly(d))}

<h2><span class="n">2</span>Where the bot goes wrong</h2>
<p>The bot is reliable when it picks Product Advice, Installs, Returns or Warranty, at about 96%. It is often wrong when
it picks Billing, Filters &amp; Consumables or Repairs. Three patterns explain most of it:</p>
<ul>
  <li><strong>Customers who mention paying.</strong> Of {n_pay:,} past requests with a remark like "paid on upi", the
  bot sent {bot_b:,} to Billing, and Billing closed <strong>{closed_b:,}</strong>. Ops policy §3 already says that
  mentioning payment does not make a request a billing request.</li>
  <li><strong>Purifier breakdowns</strong> ending in "…not working" went to Consumables and were closed by Repairs.</li>
  <li><strong>Vague requests</strong> ("please call me about my purifier") all went to Repairs, but were closed by every
  team.</li>
</ul>
{card("When the bot picked each team, how often was it right?",
      "Orange = the three teams where the bot is wrong a third of the time or more.", chart_bot_by_label(d))}

<h2><span class="n">3</span>How the router decides</h2>
<p>Customer messages follow a pattern: a greeting, one or two requests, filler such as order numbers and "pls call
back", and sometimes a payment remark. The router strips the filler and reads only the requests. When there are two
requests, the <strong>last one decides</strong>, which is how {k2:,} of {n2:,} past two-request messages were closed.</p>
<pre class="flow">message
 └► clean up garbled legacy text · merge the renamed teams
 └► drop greeting, order number, sign-off, payment remark → the LAST request decides
 └► unknown wording? fix Hinglish and spelling, then read it again
      ├─ states a need, wording seen before → the team that closed most past requests like it
      ├─ states a need, new wording         → keyword rules, then a text model
      └─ states no need ("call me")         → best guess (labelled as a guess) + one clarifying question
 └► answer: team · confidence · auto-route yes/no · reasons an agent can read
unsure answers → review queue → the agent confirms the team → learned at the next monthly retrain</pre>

<h2><span class="n">4</span>How well it works</h2>
<p>When the customer says what they need ({_pct(1 - vague_share)} of requests), the router is right
<strong>{_pct(clear_acc)}</strong> of the time. The other {_pct(vague_share)} say only "please call me about my
purifier". Past requests like that were closed by all seven teams, so nobody can route them from the words alone. That is
why no router, however clever, gets past about 86% against real outcomes from the message text.</p>
{card("Right, by type of request", "Router vs bot on the same requests.", chart_by_type(va))}
{card("Where the remaining errors come from",
      "Most errors are vague requests. The rest are clear requests closed by an unrelated team, which the log itself often marks as inconsistent.",
      chart_errors(va))}
{card("Can the router tell when it is sure?",
      "High-confidence answers can be sent automatically; low-confidence ones get the clarifying question.",
      chart_confidence(va))}
<div class="callout">Auto-routing only the confident answers covers <strong>{va['auto'].mean():.1%}</strong> of requests at
<strong>{va.loc[va['auto'], 'ok'].mean():.1%}</strong> accuracy. The rest get one question: "Is this about a fault, a delivery,
installation, warranty, payment, spares, or how to use it?"</div>

<h2><span class="n">5</span>Is this the best we can get? Four rounds of testing</h2>
<p>We tested about 40 alternatives in four rounds. For rounds 2 and 4, the rules for choosing a winner were written down
and committed <em>before</em> any result was seen. Alternatives included a fine-tuned neural network, gradient boosting,
nearest-neighbour matching, ensembles and a free local AI model. None did significantly better. An "oracle" allowed to
see the answers reaches only 86.8%, so there was almost no room left.</p>
{card("Round 2: eleven challengers vs the router",
      "Each quarter scored by models trained only on earlier data. Blue = the router before and after round 4. The line is the most any model could score.",
      chart_challengers())}
{card("A free local AI model (Qwen2.5-3B) on the same 300 requests",
      "Worse than the bot, and about 3 seconds per request; the router answers in milliseconds.", chart_llm(d))}

<h2><span class="n">6</span>Making it robust for real customers</h2>
<p>History reuses the same phrasings, but live customers won't. Round 4 added a Hinglish and spelling repair step,
and a corrections loop in which agents confirm the team for uncertain requests. Both were tested on stress tests we
generated (typos, and messages rewritten into Hinglish), so treat these as robustness checks, not live accuracy.</p>
{card("Typos and Hinglish, before and after the repair step",
      "30% of words corrupted with random typos; 300 messages rewritten into Hinglish by a local AI model.", stress_chart)}
{card("The corrections loop: how fast the router learns a new phrasing",
      "After 3 agent-confirmed examples, a phrasing it had never seen is routed as well as familiar wording.", curve_chart)}

<h2><span class="n">7</span>What it is worth</h2>
<p>Misrouting costs about <strong>Rs {c['cost_12m'] / 1e5:.1f} lakh a year</strong> today: {c['transfers_12m']:,}
transfers at Rs 305 each, plus Rs 260 for each repeat contact (ops policy §4). Misrouted requests also take twice as long
to close ({c['hours_misrouted']:.0f} hours against {c['hours_right_first']:.0f}). The router cuts that cost and removes the
Rs 3.2 lakh licence.</p>
{card("Yearly cost of routing", "* projection: assumes customers answer the clarifying question reliably, which needs a pilot.",
      money_chart)}
<div class="callout">Saving: about <strong>Rs {money['saving']:.1f} lakh a year</strong>, and up to Rs {money['upside']:.0f}
lakh if the intake question works. Running cost: Rs 0 per request, whatever the volume.</div>

<h2><span class="n">8</span>Headcount: plan on real volumes</h2>
<p>Staffing based on the bot's labels would over-staff Repairs, Billing and Filters &amp; Consumables, and under-staff
Installs, Returns and Warranty.</p>
{card("Requests per team per month", "What the bot's labels suggest vs what each team actually closed (last 6 months).",
      chart_volumes(d))}

<div class="callout warn"><strong>Limits.</strong> Tested on history, not live traffic. The recorded outcomes are about 2%
noisy. The best guess for vague requests is labelled as a guess: it was not proven better than the simpler rule. The typo
and Hinglish results come from generated test messages.</div>

<footer>Generated by <code>python -m kestrel.report</code> from the Kestrel data pack and the saved experiment outputs.
No customer data leaves the machine.</footer>
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kestrel Router Report</title><style>{CSS}</style>
<script>{get_plotlyjs()}</script></head>
<body><main>{body}</main></body></html>"""
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(page)
    print(f"wrote {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
