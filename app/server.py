"""Kestrel request router - HTTP service and screen.

    uvicorn app.server:app --port 8000      then open http://localhost:8000

POST /route     one request record as JSON -> team, confidence, reasons an agent can read, auto_route flag
POST /feedback  the team an agent confirmed (saved to data/corrections.csv, used by the next python train.py)
GET  /health  is a model loaded, which version, trained on what
GET  /teams   the seven teams and what each handles
GET  /        the screen (app/static/index.html), which calls POST /route

No model API is used, so there is no key to configure and nothing to fail without one. If the trained
model file is missing the service trains it from data/ on startup (~2 s); if the data pack is missing
too, the service still starts and /route answers 503 with instructions.
"""
from __future__ import annotations

import csv
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kestrel.router import DEFAULT_MODEL_PATH, Router  # noqa: E402

log = logging.getLogger("kestrel")
STATE: dict = {"router": None, "error": None}


def load_router() -> None:
    try:
        if DEFAULT_MODEL_PATH.exists():
            STATE["router"] = Router.load(DEFAULT_MODEL_PATH)
            return
        from kestrel.data import load_train
        log.warning("No trained model at %s - training from data/ now", DEFAULT_MODEL_PATH)
        r = Router.fit(load_train())
        r.save(DEFAULT_MODEL_PATH)
        STATE["router"] = r
    except FileNotFoundError as e:
        STATE["error"] = (f"The router is not trained yet and the data pack was not found ({e}). "
                          "Copy the Kestrel data pack into data/ and run: python train.py")
    except Exception as e:  # keep the service up and say why
        STATE["error"] = f"The router could not be loaded: {type(e).__name__}: {e}. Re-run: python train.py"
    if STATE["error"]:
        log.error(STATE["error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_router()
    yield


app = FastAPI(title="Kestrel request router", version="1.0", lifespan=lifespan)


class RequestRecord(BaseModel):
    """One service request, as in test_unlabelled.csv. Only request_text is required."""
    request_id: Optional[str] = Field(None, examples=["SR510822"])
    request_text: str = Field(..., min_length=1, max_length=2000,
                              examples=["display of air fryer gone blank pls call back"])
    channel: Optional[Literal["ivr", "chat", "whatsapp", "email"]] = None
    product_family: Optional[str] = Field(None, examples=["Air Fryer"])
    warranty_status: Optional[Literal["in_warranty", "shield", "out_of_warranty"]] = None
    created_at_ist: Optional[str] = None
    source: Optional[str] = None


def _router() -> Router:
    if STATE["router"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "The router is still loading; try again.")
    return STATE["router"]


REVIEW_QUEUE = ROOT / "out" / "review_queue.csv"        # B5: requests the router was unsure about
CORRECTIONS = ROOT / "data" / "corrections.csv"          # B5: agent-confirmed teams, read by train.py


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": datetime.now().isoformat(timespec="seconds"), **row}
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class Feedback(BaseModel):
    """B5: the team an agent confirmed for a request (e.g. after asking the clarifying question)."""
    request_id: Optional[str] = None
    request_text: str = Field(..., min_length=1, max_length=2000)
    product_family: Optional[str] = None
    team: str


@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    from kestrel.data import TEAMS
    if fb.team not in TEAMS:
        raise HTTPException(status_code=422, detail=f"team must be one of: {', '.join(TEAMS)}")
    _append_csv(CORRECTIONS, {"request_id": fb.request_id, "request_text": fb.request_text,
                              "product_family": fb.product_family, "team": fb.team,
                              "confirmed_at": datetime.now().isoformat(timespec="seconds")})
    return {"saved": True, "used_at_next": "python train.py"}


@app.post("/route")
def route(record: RequestRecord) -> dict:
    r = _router()
    out = r.route(record.model_dump())
    if not out.get("auto_route"):
        _append_csv(REVIEW_QUEUE, {"request_id": record.request_id, "request_text": record.request_text,
                                   "product_family": record.product_family, "router_team": out["team"],
                                   "route_type": out["route_type"], "confidence": out["confidence"]})
    # product_family is used only for the best guess on vague requests (A1, labelled a guess). channel and
    # warranty are accepted but do not change the decision: tested, they add nothing (+0.0 to +0.6 points).
    out["input"] = record.model_dump(exclude_none=True)
    return out


@app.get("/health")
def health() -> dict:
    r = STATE["router"]
    return {
        "status": "ok" if r else "not_ready",
        "model_version": getattr(r, "version", None),
        "trained_on_requests": getattr(r, "trained_rows", None),
        "trained_until": getattr(r, "trained_until", None),
        "uses_paid_api": False,
        "message": None if r else STATE["error"],
    }


@app.get("/teams")
def teams() -> dict:
    return {"teams": _router().team_handles}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")
