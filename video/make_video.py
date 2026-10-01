"""Build the walkthrough video (max 3 min, burned-in captions, no slides: real files, the report and the live app).

    python video/make_video.py            -> ../kestrel-walkthrough.mp4 (1920x1080, 30 fps)

Needs: the report (python -m kestrel.report), the service running on port 8766 for the app scenes
(uvicorn app.server:app --port 8766), Google Chrome and ffmpeg.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "out" / "video_frames"
OUTFILE = ROOT.parent / "kestrel-walkthrough.mp4"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
APP = "http://localhost:8766"
W, H = 1920, 1080
MAX_TOTAL = 178  # seconds; the brief allows three minutes

SECTIONS = {1: "What I tried", 2: "What I changed", 3: "What I threw away"}

# kind, section, source, caption ([[...]] = yellow), extra
SCENES = [
    ("page", 0, ("report.html", "out/report.html — Kestrel request router"),
     "Kestrel Home service-request routing. A walkthrough of [[what I tried]], [[what I changed]], and "
     "[[what I threw away]].", {}),
    ("file", 1, "data/email-thread.txt",
     "The ask: replace the routing bot with a model that [[matches its labels at 90%]]. First step: join those labels "
     "to where each request was actually closed.",
     {"anchor": "From: Ritu Deshpande", "hl": ["matches it at 90%+", "90% match is the bar"]}),
    ("page", 1, ("report.html#matching-the-bot-s-labels-is-not-the-same-as-rou", "out/report.html — section 1"),
     "The labels are the bot's guesses: [[right only 77% of the time]]. A copy of the bot hit [[91.8%]] and routed no "
     "better, so I trained on the [[team that closed each request]] instead.", {}),
    ("file", 1, "out/experiments.md",
     "About 20 approaches on three tests. Bag-of-words stalled at [[83%]]: when a message asks two things, [[the last "
     "one decides]] (98.7%), so the last request became its own input.",
     {"anchor": "## A out-of-time", "hl": ["| TF-IDF+LR on outcomes |", "| TF-IDF+last-clause+LR on outcomes |",
                                           "| Lookup on last clause", "trained on BOT labels"]}),
    ("file", 1, "out/llm_benchmark.md",
     "A free local LLM (Qwen2.5-3B) scored [[71%]], [[worse than the bot]], at 3 seconds a request. Giving it hints "
     "made it worse (68%).",
     {"anchor": "# Local LLM benchmark", "hl": ["| Router (this work)", "| LLM zero-shot", "| LLM with data-derived"]}),
    ("page", 1, ("report.html#round-2-eleven-challengers-vs-the-router", "out/report.html — section 5"),
     "Round 2: [[11 challengers]] under rules committed before running, including a fine-tuned transformer, gradient "
     "boosting and ensembles. [[None was significantly better]]; the ceiling is 86.8%.", {}),
    ("file", 2, "tests/test_text.py",
     "Reading the errors found a bug: [[\"please call back regarding my fan\"]] was stripped as a sign-off and sent to "
     "Installs. Fixed, with a regression test.",
     {"anchor": "def test_please_call_back_regarding_is_vague_not_a_closing", "hl": ["# Regression", "is_vague(\"sir"]}),
    ("file", 2, "kestrel/text.py",
     "\"Vague\" now means [[the last request states no need]]. That is 16% of requests, and no model can place them, "
     "so they get [[one clarifying question]].",
     {"anchor": "def is_vague", "hl": ["The deciding (last) clause", "Outcomes for these are spread", "earlier clause's team"]}),
    ("file", 2, "out/evaluation.md",
     "My first robustness score, [[98%]], used keyword rules I wrote after reading every phrasing. I replaced it with a "
     "fair check: [[87%]] on genuinely new wording.",
     {"anchor": "## 3. New wording", "hl": ["| Shipped rules", "| **Fair check"]}),
    ("file", 2, "docs/ROUND4_RESULTS.md",
     "Round 4 adopted [[Hinglish + spelling repair]] (heavy typos 94% → 97.6%), a [[corrections loop]] and an "
     "auto-route flag. The product-based vague guess is [[labelled as a guess]].",
     {"anchor": "## Short comparison", "hl": ["| A1 |", "| B4 |", "| B5 |", "| C1 |"]}),
    ("page", 2, (APP + "/?q=water%20purifier%20not%20turning%20on%20paid%20on%20upi%20pls%20call%20back"
                 "&product=Water%20Purifier", "localhost — Kestrel request router (POST /route)"),
     "The service: one endpoint, one screen. \"Paid on UPI\" goes to [[Repairs, not Billing]]. The bot sent 525 of "
     "626 of these to Billing, and Billing closed one.", {"scale": 1.5, "y": 20}),
    ("page", 2, (APP + "/?q=urgent%3A%20please%20call%20back%20regarding%20water%20purifier&product=Water%20Purifier",
                 "localhost — Kestrel request router (POST /route)"),
     "A vague request gets [[low confidence]], a guess clearly marked as one, and [[seven options]] to ask. The "
     "agent's answer is saved and learned at the next retrain.", {"scale": 1.25, "y": 95}),
    ("file", 3, "docs/FINDINGS.md",
     "Thrown away: [[training on the bot's labels]], [[an LLM in the routing path]], and sentence embeddings: same "
     "accuracy, about 100× slower, and a 2 GB install.",
     {"anchor": "## 7. What I threw away", "hl": ["Training on `team_label`", "A local LLM in the path",
                                                  "Sentence embeddings in the product"]}),
    ("file", 3, "docs/ROUND4_RESULTS.md",
     "Also dropped in round 4: a [[three-way vote that cost 10 points]], paraphrase lookup and fuzzy matching (no "
     "effect), and reordered options ([[worse]]).",
     {"anchor": "| B1 |", "hl": ["| B2 |", "| B3 |", "| C2 |", "| B1 |"]}),
    ("page", 3, ("report.html#yearly-cost-of-routing", "out/report.html — section 7"),
     "Result: [[86.0%]] sent to the right team against the bot's 76.7%, [[Rs 8.8 lakh a year]] saved, and "
     "[[Rs 0 per request]].", {}),
]

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 1920px; height: 1080px; overflow: hidden; }
body { background: radial-gradient(ellipse at 50% 35%, #34353a 0%, #232427 70%); color: #fff;
       font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
.badge { position: absolute; left: 60px; top: 26px; display: flex; align-items: center; gap: 14px; font-size: 24px; color: #e6e6e6; }
.badge b { background: #2a62d6; color: #fff; padding: 4px 12px; border-radius: 6px; font-size: 22px; letter-spacing: .02em; }
.win { position: absolute; left: 60px; right: 60px; top: 78px; height: 800px; border-radius: 14px; overflow: hidden;
       box-shadow: 0 24px 60px rgba(0,0,0,.45); background: #1e1f22; }
.win.nobadge { top: 78px; }
.bar { height: 44px; background: #2b2c30; display: flex; align-items: center; padding: 0 18px; position: relative; }
.dots span { display: inline-block; width: 13px; height: 13px; border-radius: 50%; margin-right: 8px; }
.bar .t { position: absolute; left: 0; right: 0; text-align: center; color: #c9c9cc; font-size: 18px; }
.code { padding: 22px 0; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 23px; line-height: 1.5; color: #e4e4e7; }
.ln { display: grid; grid-template-columns: 96px 1fr; padding-right: 40px; border-left: 5px solid transparent; }
.ln .n { color: #6e6f76; text-align: right; padding-right: 24px; }
.ln .x { white-space: pre-wrap; word-break: break-word; }
.ln.hl { background: rgba(212, 167, 25, .30); border-left-color: #f5c518; }
.frame { width: 1200px; height: 504px; border: 0; transform: scale(1.5); transform-origin: 0 0; background: #fcfcfb; }
.cap { position: absolute; left: 0; right: 0; bottom: 0; height: 190px; background: #111214; display: flex;
       align-items: center; justify-content: center; padding: 0 160px; text-align: center; font-size: 38px;
       line-height: 1.38; font-weight: 600; color: #fff; }
.cap em { font-style: normal; color: #f5c518; }
"""


def caption_html(text: str) -> str:
    out = html.escape(text)
    return re.sub(r"\[\[(.+?)\]\]", r"<em>\1</em>", out)


def file_lines(path: str, anchor: str, hl: list[str], budget: int = 20) -> str:
    lines = (ROOT / path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if anchor in l)
    start = max(0, start - 1)
    rows, used, i = [], 0, start
    while i < len(lines) and used < budget:
        text = lines[i]
        cost = max(1, -(-len(text) // 118))  # visual lines after wrapping (~118 chars per line at 23px)
        if used + cost > budget:
            break
        cls = " hl" if any(h in text for h in hl) else ""
        rows.append(f'<div class="ln{cls}"><span class="n">{i + 1}</span><span class="x">{html.escape(text) or " "}</span></div>')
        used += cost
        i += 1
    return f'<div class="code">{"".join(rows)}</div>'


def scene_html(kind, section, source, caption, extra) -> str:
    badge = (f'<div class="badge"><b>SECTION {section}</b>{SECTIONS[section]}</div>' if section else "")
    dots = ('<div class="dots"><span style="background:#ff5f57"></span><span style="background:#febc2e"></span>'
            '<span style="background:#28c840"></span></div>')
    if kind == "file":
        title, body = source, file_lines(source, extra["anchor"], extra["hl"])
    elif source[0].startswith("http"):
        # The live app is screenshotted directly (shoot_app) and shown at a readable scale, shifted to the result.
        title = source[1]
        sc, y = extra.get("scale", 1.5), extra.get("y", 0)
        body = (f'<div style="overflow:hidden;height:756px;background:#f6f5f2"><img src="{extra["png"]}" '
                f'style="width:1200px;transform-origin:0 0;transform:scale({sc}) translateY(-{y}px)"></div>')
    else:
        src, title = source
        body = f'<iframe class="frame" src="{html.escape("../" + src)}"></iframe>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{badge}'
            f'<div class="win"><div class="bar">{dots}<div class="t">{html.escape(title)}</div></div>{body}</div>'
            f'<div class="cap"><div>{caption_html(caption)}</div></div></body></html>')


def duration(caption: str) -> float:
    words = len(re.sub(r"\[\[|\]\]", "", caption).split())
    return max(8.0, words / 2.6 + 3.0)  # comfortable reading pace


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    durations = []
    for k, sc in enumerate(SCENES):
        if sc[0] == "page" and sc[2][0].startswith("http"):
            shot = FRAMES / f"app{k:02d}.png"
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1200,1400",
                            "--virtual-time-budget=12000", f"--screenshot={shot}", sc[2][0]], check=True, capture_output=True)
            sc[4]["png"] = shot.name
        page = FRAMES / f"scene{k:02d}.html"
        page.write_text(scene_html(*sc))
        png = FRAMES / f"scene{k:02d}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                        "--virtual-time-budget=12000", f"--screenshot={png}", page.as_uri()],
                       check=True, capture_output=True)
        durations.append(duration(sc[3]))
        print(f"scene {k:02d}  {durations[-1]:4.1f}s  {sc[2] if sc[0] == 'file' else sc[2][1]}", flush=True)
    scale = min(1.0, MAX_TOTAL / sum(durations))
    durations = [round(d * scale, 2) for d in durations]
    concat = FRAMES / "concat.txt"
    lines = []
    for k, d in enumerate(durations):
        lines += [f"file 'scene{k:02d}.png'", f"duration {d}"]
    lines.append(f"file 'scene{len(durations) - 1:02d}.png'")  # ffmpeg concat needs the last frame repeated
    concat.write_text("\n".join(lines) + "\n")
    total = sum(durations)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-vf", f"fps=30,scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-shortest", "-t", f"{total:.2f}", "-movflags", "+faststart", str(OUTFILE)],
                   check=True)
    print(f"wrote {OUTFILE}  ({total:.0f}s, {len(SCENES)} scenes)")


if __name__ == "__main__":
    sys.exit(main())
