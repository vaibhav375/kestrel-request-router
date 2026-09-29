"""Kestrel request router - HTTP service and screen.

    uvicorn app.server:app --port 8000      then open http://localhost:8000

POST /route   one request record as JSON -> team, confidence, reasons an agent can read
GET  /health  is a model loaded, which version, trained on what
GET  /teams   the seven teams and what each handles
GET  /        the screen (app/static/index.html), which calls POST /route

No model API is used, so there is no key to configure and nothing to fail without one. If the trained
model file is missing the service trains it from data/ on startup (~2 s); if the data pack is missing
too, the service still starts and /route answers 503 with instructions.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
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


@app.post("/route")
def route(record: RequestRecord) -> dict:
    r = _router()
    out = r.route(record.model_dump())
    # channel / product / warranty are accepted but do not change the decision: tested, they add nothing
    # (experiments: +0.0 to +0.6 points, within noise). They are echoed so the screen can show them.
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
