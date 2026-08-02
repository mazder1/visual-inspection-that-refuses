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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Inspection That Refuses</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; color: #111; }
  h1 { font-size: 1.4rem; } .muted { color: #666; font-size: .9rem; }
  select, input, button { font-size: 1rem; margin: .3rem 0; }
  #result { white-space: pre-wrap; background: #f6f6f4; padding: 1rem; border-radius: 6px; font-family: monospace; font-size: .8rem; overflow-x: auto; }
  #mask { image-rendering: pixelated; border: 1px solid #ccc; max-width: 256px; }
  .verdict { font-size: 1.6rem; font-weight: 700; margin: .5rem 0; }
</style></head><body>
<h1>Visual Inspection That Refuses</h1>
<p class="muted">Drop a photo of a part. The system returns a defect mask, a calibrated
probability, and one of three verdicts: <b>fail</b>, <b>pass</b>, or <b>no-call</b> —
which means “route this part to a human”. Decision support only; not qualified for
production use. Expect ~5–10 s: the model runs 20 times per image to measure its own
uncertainty.</p>
<label>Category:
  <select id="category"><option>bottle</option><option>carpet</option><option selected>hazelnut</option></select>
</label><br>
<input type="file" id="file" accept="image/*">
<button id="go">Inspect</button>
<div class="verdict" id="verdict"></div>
<img id="mask" style="display:none">
<div id="result"></div>
<script>
document.getElementById('go').onclick = async () => {
  const file = document.getElementById('file').files[0];
  if (!file) { alert('pick an image first'); return; }
  const category = document.getElementById('category').value;
  document.getElementById('verdict').textContent = 'inspecting…';
  const body = new FormData(); body.append('file', file);
  const started = performance.now();
  const response = await fetch(`/inspect/${category}`, { method: 'POST', body });
  const data = await response.json();
  const total = Math.round(performance.now() - started);
  if (!response.ok) {
    document.getElementById('verdict').textContent = 'error';
    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
    return;
  }
  const colours = { fail: '#c0392b', pass: '#1e8449', 'no-call': '#b7791f' };
  const v = document.getElementById('verdict');
  v.textContent = `${data.verdict.toUpperCase()}  (p=${data.probability_defective}, round-trip ${total} ms)`;
  v.style.color = colours[data.verdict] || '#111';
  const mask = document.getElementById('mask');
  mask.src = 'data:image/png;base64,' + data.mask_png_base64;
  mask.style.display = 'block';
  const { mask_png_base64, ...rest } = data;
  document.getElementById('result').textContent = JSON.stringify(rest, null, 2);
};
</script></body></html>"""
