"""The HTTP service: one endpoint, one page, a health check, a rate limit.

CPU-only by design. The CUDA wheel alone is ~2.5 GB; the CPU wheel keeps the
container near 1 GB, and Cloud Run bills CPU. The service runs the exact
20-pass chain the evaluation measured -- about 4 s per request on a desktop
CPU -- rather than a faster variant the published numbers do not describe.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from vinspect.serve.predictor import Predictor

BUNDLE_DIR = Path(os.environ.get("VINSPECT_BUNDLES", "bundles"))
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
RATE_LIMIT = int(os.environ.get("VINSPECT_RATE_LIMIT", "10"))  # per minute per IP

app = FastAPI(title="Visual Inspection That Refuses", docs_url="/docs")

_predictors: Dict[str, Predictor] = {}
_predictor_lock = threading.Lock()
_requests: Dict[str, deque] = defaultdict(deque)


def _get_predictor(category: str) -> Predictor:
    with _predictor_lock:
        if category not in _predictors:
            bundle = BUNDLE_DIR / category
            if not (bundle / "chain.json").is_file():
                available = sorted(
                    p.name for p in BUNDLE_DIR.iterdir() if (p / "chain.json").is_file()
                ) if BUNDLE_DIR.is_dir() else []
                raise HTTPException(
                    404,
                    f"unknown category {category!r}; available: {available}",
                )
            _predictors[category] = Predictor(bundle)
        return _predictors[category]


def _rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _requests[client]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, f"rate limit: {RATE_LIMIT} requests/minute")
    window.append(now)


# Both paths serve the same check. /healthz is the conventional name and works
# locally and in Docker, but Google's Front End reserves that exact path on
# run.app domains and answers its own 404 before the request reaches the
# container -- so the deployed health check lives at /health.
@app.get("/health")
@app.get("/healthz")
def healthz() -> Dict:
    loaded = sorted(_predictors)
    available = (
        sorted(p.name for p in BUNDLE_DIR.iterdir() if (p / "chain.json").is_file())
        if BUNDLE_DIR.is_dir()
        else []
    )
    return {"status": "ok", "bundles": available, "loaded": loaded}


@app.post("/inspect/{category}")
async def inspect(category: str, request: Request, file: UploadFile = File(...)):
    _rate_limit(request)
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload over {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    if not payload:
        raise HTTPException(400, "empty upload")

    predictor = _get_predictor(category)
    try:
        result = predictor.inspect(payload)
    except Exception:  # malformed image, truncated file, wrong format
        raise HTTPException(
            422,
            "could not decode the upload as an image; send a PNG or JPEG of a "
            "single part",
        )
    return JSONResponse(result)


INDEX_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")



