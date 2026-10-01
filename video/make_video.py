"""Record the walkthrough video: real browser screen recording with motion, burned-in captions, max 3 minutes.

    python video/make_video.py        -> ../kestrel-walkthrough.mp4 (1920x1080, 30 fps, silent audio track)

Every scene is a live page recorded with Playwright driving the installed Google Chrome:
  * report scenes scroll through out/report.html,
  * app scenes type a request into the running service, press "Route this request" and show the answer
    (the vague one also clicks a clarifying answer, which saves a correction),
  * file scenes scroll through the real repo file like an editor while the key lines light up one by one.
The script starts the service (port 8766) and a static server for out/ (port 8790) itself, and removes the
correction/review files its own clicks create. Needs: pip install playwright (uses the system Chrome), ffmpeg,
and the report (python -m kestrel.report).
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "out" / "video_frames"
OUTFILE = ROOT.parent / "kestrel-walkthrough.mp4"
APP = "http://localhost:8766"
STATIC = "http://localhost:8790"          # serves ROOT/out, so scene pages and report.html share an origin
W, H = 1920, 1080
SECTIONS = {1: "What I tried", 2: "What I changed", 3: "What I threw away"}

PAY = "water purifier not turning on paid on upi pls call back"
VAGUE = "urgent: please call back regarding water purifier"

# (kind, section, seconds, caption with [[yellow]] parts, spec)
SCENES = [
    ("report", 0, 15, "Kestrel Home service-request routing. A walkthrough of [[what I tried]], [[what I changed]], "
     "and [[what I threw away]].", {"title": "out/report.html — Kestrel request router", "path": [("top", 0, 0), ("bottom", 0.8, 13.6)]}),
    ("file", 1, 11, "The ask: replace the routing bot with a model that [[matches its labels at 90%]]. First step: join "
     "those labels to where each request was actually closed.",
     {"path": "data/email-thread.txt", "anchor": "From: Ritu Deshpande", "hl": ["matches it at 90%+", "90% match is the bar"]}),
    ("report", 1, 13, "The labels are the bot's guesses: [[right only 77% of the time]]. A copy of the bot hit [[91.8%]] "
     "and routed no better, so I trained on the [[team that closed each request]] instead.",
     {"title": "out/report.html — section 1",
      "path": [("top", 0, 0), ("matching-the-bot-s-labels-is-not-the-same-as-rou", 0.8, 3.0),
               ("the-bot-s-accuracy-month-by-month", 8.0, 3.0)]}),
    ("file", 1, 13, "About 20 approaches on three tests. Bag-of-words stalled at [[83%]]: when a message asks two things, "
     "[[the last one decides]] (98.7%), so the last request became its own input.",
     {"path": "out/experiments.md", "anchor": "## A out-of-time",
      "hl": ["| TF-IDF+LR on outcomes |", "| TF-IDF+last-clause+LR on outcomes |", "| Lookup on last clause", "trained on BOT labels"]}),
    ("report", 1, 15, "Round 2: [[11 challengers]] under rules committed before running, including a fine-tuned transformer. "
     "[[None was significantly better]]. A free local LLM scored [[71%]], worse than the bot.",
     {"title": "out/report.html — section 5",
      "path": [("can-the-router-tell-when-it-is-sure", 0, 0), ("round-2-eleven-challengers-vs-the-router", 0.8, 2.5),
               ("a-free-local-ai-model-qwen2-5-3b-on-the-same-300", 8.5, 2.5)]}),
    ("file", 2, 11, "Reading the errors found a bug: [[\"please call back regarding my fan\"]] was stripped as a sign-off "
     "and sent to Installs. Fixed, with a regression test.",
     {"path": "tests/test_text.py", "anchor": "def test_please_call_back_regarding_is_vague_not_a_closing",
      "hl": ["# Regression", "is_vague(\"sir"]}),
    ("file", 2, 11, "\"Vague\" now means [[the last request states no need]]. That is 16% of requests, and no model can "
     "place them, so they get [[one clarifying question]].",
     {"path": "kestrel/text.py", "anchor": "def is_vague",
      "hl": ["The deciding (last) clause", "Outcomes for these are spread", "earlier clause's team"]}),
    ("file", 2, 11, "My first robustness score, [[98%]], used keyword rules I wrote after reading every phrasing. I "
     "replaced it with a fair check: [[87%]] on genuinely new wording.",
     {"path": "out/evaluation.md", "anchor": "## 3. New wording", "hl": ["| Shipped rules", "| **Fair check"]}),
    ("file", 2, 12, "Round 4 adopted [[Hinglish + spelling repair]] (heavy typos 94% → 97.6%), a [[corrections loop]] "
     "and an auto-route flag. The product-based vague guess is [[labelled as a guess]].",
     {"path": "docs/ROUND4_RESULTS.md", "anchor": "## Short comparison", "hl": ["| A1 |", "| B4 |", "| B5 |", "| C1 |"]}),
    ("app", 2, 15, "The live service: a request comes in and is routed in milliseconds. \"Paid on UPI\" goes to "
     "[[Repairs, not Billing]]; the bot sent 525 of 626 of these to Billing.",
     {"title": "localhost:8766 — Kestrel request router (POST /route)", "text": PAY, "product": "Water Purifier"}),
    ("app", 2, 16, "A vague request gets [[low confidence]], a guess marked as one, and [[seven options]] to ask. The "
     "agent's answer is saved and learned at the next retrain.",
     {"title": "localhost:8766 — Kestrel request router (POST /route)", "text": VAGUE, "product": "Water Purifier",
      "answer": "Warranty or Kestrel Shield"}),
    ("file", 3, 11, "Thrown away: [[training on the bot's labels]], [[an LLM in the routing path]], and sentence "
     "embeddings: same accuracy, about 100× slower, and a 2 GB install.",
     {"path": "docs/FINDINGS.md", "anchor": "## 7. What I threw away",
      "hl": ["Training on `team_label`", "A local LLM in the path", "Sentence embeddings in the product"]}),
    ("file", 3, 10, "Also dropped in round 4: a [[three-way vote that cost 10 points]], paraphrase lookup and fuzzy "
     "matching (no effect), and reordered options ([[worse]]).",
     {"path": "docs/ROUND4_RESULTS.md", "anchor": "| B1 |", "hl": ["| B2 |", "| B3 |", "| C2 |", "| B1 |"]}),
    ("report", 3, 12, "Result: [[86.0%]] sent to the right team against the bot's 76.7%, [[Rs 8.8 lakh a year]] saved, "
     "and [[Rs 0 per request]].",
     {"title": "out/report.html — section 7 and summary",
      "path": [("the-corrections-loop-how-fast-the-router-learns-", 0, 0), ("yearly-cost-of-routing", 0.8, 2.5),
               ("top", 7.5, 3.5)]}),
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
.bar { height: 44px; background: #2b2c30; display: flex; align-items: center; padding: 0 18px; position: relative; }
.dots span { display: inline-block; width: 13px; height: 13px; border-radius: 50%; margin-right: 8px; }
.bar .t { position: absolute; left: 0; right: 0; text-align: center; color: #c9c9cc; font-size: 18px; }
.code { height: 756px; overflow: hidden; padding: 22px 0 300px; font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 23px; line-height: 1.5; color: #e4e4e7; }
.ln { display: grid; grid-template-columns: 96px 1fr; padding-right: 40px; border-left: 5px solid transparent;
      transition: background-color .6s ease, border-color .6s ease; }
.ln .n { color: #6e6f76; text-align: right; padding-right: 24px; }
.ln .x { white-space: pre-wrap; word-break: break-word; }
.ln.on { background: rgba(212, 167, 25, .30); border-left-color: #f5c518; }
.frame { width: 1200px; height: 504px; border: 0; transform: scale(1.5); transform-origin: 0 0; background: #fcfcfb; }
.cap { position: absolute; left: 0; right: 0; bottom: 0; height: 190px; background: #111214; display: flex;
       align-items: center; justify-content: center; padding: 0 160px; text-align: center; font-size: 38px;
       line-height: 1.38; font-weight: 600; color: #fff; animation: fade .5s ease both; }
.cap em { font-style: normal; color: #f5c518; }
@keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
"""

# Page-side helpers: smooth scrolling of the code panel or the report iframe, and highlight reveal.
JS = """
function ease(t){ return t<.5 ? 2*t*t : 1-Math.pow(-2*t+2,2)/2; }
function animate(get, set, to, ms){ return new Promise(res=>{ const from=get(), t0=performance.now();
  function step(now){ const k=Math.min(1,(now-t0)/ms); set(from+(to-from)*ease(k)); k<1?requestAnimationFrame(step):res(); }
  requestAnimationFrame(step); }); }
window.playFile = async function(total){
  const code=document.querySelector('.code'); const hls=[...document.querySelectorAll('.ln.hl')];
  const first = hls.length ? hls[0].offsetTop : 0;
  const target = Math.max(0, Math.min(first-200, code.scrollHeight-code.clientHeight));
  await new Promise(r=>setTimeout(r,500));
  await animate(()=>code.scrollTop, v=>code.scrollTop=v, target, 1800);
  for (const el of hls){ el.classList.add('on'); await new Promise(r=>setTimeout(r,900)); }
  const rest = Math.max(1000, total*1000 - 2300 - hls.length*900 - 600);
  await animate(()=>code.scrollTop, v=>code.scrollTop=v, Math.min(code.scrollTop+90, code.scrollHeight-code.clientHeight), rest);
};
window.scrollReport = function(target, ms){
  const w=document.querySelector('iframe').contentWindow, d=w.document;
  let y = target==='top' ? 0 : target==='bottom' ? d.documentElement.scrollHeight - w.innerHeight
          : d.getElementById(target).getBoundingClientRect().top + w.scrollY - 18;
  return animate(()=>w.scrollY, v=>w.scrollTo(0,v), y, ms);
};
"""


def caption_html(text: str) -> str:
    return re.sub(r"\[\[(.+?)\]\]", r"<em>\1</em>", html.escape(text))


def code_html(path: str, anchor: str, hl: list[str]) -> str:
    lines = (ROOT / path).read_text().splitlines()
    a = next(i for i, l in enumerate(lines) if anchor in l)
    rows = []
    for i in range(max(0, a - 8), min(len(lines), a + 34)):
        text = lines[i]
        cls = " hl" if (i >= a and any(h in text for h in hl)) else ""
        rows.append(f'<div class="ln{cls}"><span class="n">{i + 1}</span><span class="x">{html.escape(text) or " "}</span></div>')
    return f'<div class="code">{"".join(rows)}</div>'


def scene_page(kind, section, caption, spec) -> str:
    badge = f'<div class="badge"><b>SECTION {section}</b>{SECTIONS[section]}</div>' if section else ""
    dots = ('<div class="dots"><span style="background:#ff5f57"></span><span style="background:#febc2e"></span>'
            '<span style="background:#28c840"></span></div>')
    if kind == "file":
        title, body = spec["path"], code_html(spec["path"], spec["anchor"], spec["hl"])
    elif kind == "report":
        title, body = spec["title"], f'<iframe class="frame" src="{STATIC}/report.html"></iframe>'
    else:
        title, body = spec["title"], f'<iframe class="frame" src="{APP}/"></iframe>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style><script>{JS}</script></head><body>'
            f'{badge}<div class="win"><div class="bar">{dots}<div class="t">{html.escape(title)}</div></div>{body}</div>'
            f'<div class="cap"><div>{caption_html(caption)}</div></div></body></html>')


def wait_url(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


def run_scene(page, kind, seconds, spec):
    """Perform the scene's motion; returns when the scene's time is used up."""
    t_end = time.time() + seconds
    if kind == "file":
        page.evaluate(f"playFile({seconds})")
    elif kind == "report":
        # path = [(target, start_s, duration_s), ...]: jump to the first target, then scroll smoothly to each next one
        start = time.time()
        steps = spec["path"]
        page.evaluate(f"scrollReport({json.dumps(steps[0][0])}, 1)")
        for target, at, dur in steps[1:]:
            time.sleep(max(0, start + at - time.time()))
            page.evaluate(f"scrollReport({json.dumps(target)}, {int(dur * 1000)})")
    else:  # app: type, route, show the answer
        app = next(f for f in page.frames if f.url.startswith(APP))
        app.wait_for_selector("#text")
        time.sleep(1.0)
        app.evaluate("document.getElementById('text').focus()")
        page.keyboard.type(spec["text"], delay=38)
        time.sleep(0.4)
        app.evaluate(f"document.getElementById('product').value = {json.dumps(spec['product'])}")
        time.sleep(0.5)
        app.evaluate("""() => { const b = document.getElementById('go'); b.style.filter = 'brightness(1.25)';
                        setTimeout(() => { b.style.filter = ''; b.click(); }, 250); }""")
        app.wait_for_selector("#out .team", timeout=10000)
        if spec.get("answer"):
            time.sleep(3.0)
            app.evaluate("""() => new Promise(res => { const y = document.querySelector('.clarify').getBoundingClientRect().top
                              + scrollY - 20; const y0 = scrollY, t0 = performance.now();
                              (function s(n){ const k = Math.min(1, (n - t0) / 1400); scrollTo(0, y0 + (y - y0) * k);
                              k < 1 ? requestAnimationFrame(s) : res(); })(t0); })""")
            time.sleep(2.0)
            app.evaluate(f"""() => [...document.querySelectorAll('.clarify .opts button')]
                             .find(b => b.textContent.startsWith({json.dumps(spec['answer'])})).click()""")
    time.sleep(max(0, t_end - time.time()))


def main():
    from playwright.sync_api import sync_playwright
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "raw").mkdir(parents=True)
    corrections, queue = ROOT / "data" / "corrections.csv", ROOT / "out" / "review_queue.csv"
    pre_existing = {p: p.exists() for p in (corrections, queue)}
    procs = [subprocess.Popen([str(ROOT / ".venv/bin/uvicorn"), "app.server:app", "--port", "8766"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, "-m", "http.server", "8790", "--bind", "127.0.0.1"], cwd=ROOT / "out",
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    segments = []
    try:
        wait_url(APP + "/health")
        wait_url(STATIC + "/report.html")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            for k, (kind, section, seconds, caption, spec) in enumerate(SCENES):
                (WORK / f"scene{k:02d}.html").write_text(scene_page(kind, section, caption, spec))
                ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(WORK / "raw"),
                                          record_video_size={"width": W, "height": H})
                page = ctx.new_page()
                t_created = time.time()
                page.goto(f"{STATIC}/video_frames/scene{k:02d}.html", wait_until="load")
                if kind == "report":
                    page.frames[1].wait_for_load_state("load")
                    page.wait_for_timeout(1500)  # plotly draws its charts after load
                elif kind == "app":
                    page.wait_for_timeout(800)
                offset = time.time() - t_created
                run_scene(page, kind, seconds, spec)
                video = page.video.path()
                ctx.close()
                seg = WORK / f"seg{k:02d}.mp4"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.2f}", "-i", str(video), "-t", str(seconds),
                                "-vf", f"fps=30,scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                                "-crf", "18", "-an", str(seg)], check=True)
                segments.append(seg)
                print(f"scene {k:02d} {kind:6s} {seconds:4.1f}s (trimmed {offset:.1f}s of loading)", flush=True)
            browser.close()
    finally:
        for pr in procs:
            pr.terminate()
        for path, existed in pre_existing.items():  # the vague scene's click saves a correction: undo it
            if not existed and path.exists():
                path.unlink()
    lst = WORK / "segments.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in segments))
    total = sum(s[2] for s in SCENES)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", "-t", str(total), "-movflags", "+faststart", str(OUTFILE)],
                   check=True)
    print(f"wrote {OUTFILE}  ({total}s, {len(SCENES)} scenes)")


if __name__ == "__main__":
    main()
