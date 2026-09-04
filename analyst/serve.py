"""Live Analyst panel — localhost FastAPI UI + capture control.

    python -m analyst.serve
    open http://127.0.0.1:8765
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyst.capture.worker import CaptureWorker
from analyst.pipeline import AnalystPipeline
from analyst.store.db import AnalystStore
from analyst.store.persist import persist_result
from analyst.whitebox.trace import trace_from_run_row

PANEL_DIR = Path(__file__).resolve().parent / "panel"
HOST = "127.0.0.1"
PORT = 8765

store = AnalystStore()
# Retention runs at startup as well as amortised on insert, so thumbnails from
# a previous session cannot outlive their TTL just because nothing was captured.
try:
    _pruned = store.prune()
except Exception:
    _pruned = {"thumbs_cleared": 0, "rows_deleted": 0}
pipeline = AnalystPipeline()
worker = CaptureWorker(store=store, pipeline=pipeline)

app = FastAPI(title="Guardian Analyst", version="0.2.0")


class StartBody(BaseModel):
    age: int = Field(10, ge=5, le=17)
    # A check measures 6-9s typically (38s worst seen), so anything under
    # ~10s can only overlap and be skipped. See CaptureWorker.start().
    interval_s: float = Field(30.0, ge=10.0, le=600.0)


def _thumb_b64(thumb: Optional[bytes]) -> Optional[str]:
    if not thumb:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii")


def _public_run(row: Dict[str, Any], *, full: bool = True) -> Dict[str, Any]:
    """Serialise a stored run for the panel.

    `full=False` is the list view. Three fields dominate a run's size and none
    of them are drawn in a list: the preview image, the rebuilt trace, and the
    per-region detections. Serving all forty runs complete, every three
    seconds, moved 3.16 MB per poll — 63 MB a minute — to render 0.07 MB of
    visible headlines. The panel now fetches the whole run only for the one it
    has selected, via /api/runs/{id}.
    """
    out = dict(row)
    thumb = out.pop("thumb_jpeg", None)
    if not full:
        out.pop("trace", None)
        evidence = out.get("evidence")
        if isinstance(evidence, dict):
            # `escalated` is the only evidence field the list reads.
            out["evidence"] = {k: v for k, v in evidence.items() if k != "detections"}
        return out
    out["thumb_data_url"] = _thumb_b64(thumb)
    if not out.get("trace"):
        out["trace"] = trace_from_run_row(row)
    return out


@app.get("/")
def index() -> FileResponse:
    path = PANEL_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(500, "panel/index.html missing")
    return FileResponse(path, media_type="text/html")


@app.get("/api/health")
def api_health() -> dict:
    st = worker.status()
    backends = st.get("backends") or {}
    degraded = [
        k for k, v in backends.items()
        if str(v) in ("none", "deferred", "clip_failed", "whisper_failed")
    ]
    # Surface *why* a backend is down, so a teammate with a broken native
    # install can tell it apart from simply not having installed it.
    backend_errors = {}
    ocr_err = getattr(pipeline.ocr, "last_error", "")
    if ocr_err:
        backend_errors["ocr"] = ocr_err
    audio_err = getattr(worker.audio, "last_error", "")
    if audio_err:
        backend_errors["audio"] = audio_err
    # The meme reader is the one branch that goes quiet without failing: with
    # ANALYST_VLM_URL unset it never runs, reports "none", and logs nothing.
    # That reads as "no memes on screen" rather than "the meme channel is off",
    # which is exactly how a demo of captioned memes can score 0.08 and look
    # like a broken classifier. Say so explicitly.
    vlm = getattr(pipeline, "vision_meaning", None)
    if vlm is not None and not getattr(vlm, "enabled", False):
        backend_errors["vision_meaning"] = (
            "disabled: ANALYST_VLM_URL not set - the meme reader (picture "
            "meaning + text inside pictures) will not run"
        )
    elif vlm is not None and getattr(vlm, "last_error", ""):
        backend_errors["vision_meaning"] = vlm.last_error

    return {
        "ok": True,
        "capturing": st.get("capturing"),
        "protection": st.get("protection"),
        "degraded_backends": degraded,
        "backend_errors": backend_errors,
        "last_error": st.get("last_error") or "",
        "stats": st.get("stats"),
    }


@app.post("/api/retention/prune")
def api_prune() -> dict:
    """Apply the retention policy now."""
    return {"ok": True, **store.prune()}


@app.post("/api/retention/purge")
def api_purge() -> dict:
    """Delete every recorded run, including all stored thumbnails."""
    return {"ok": True, "rows_deleted": store.purge_all()}


@app.get("/api/retention")
def api_retention() -> dict:
    from analyst.store import db as _db

    return {
        "thumb_ttl_hours": _db.THUMB_TTL_HOURS,
        "run_ttl_days": _db.RUN_TTL_DAYS,
        "max_runs": _db.MAX_RUNS,
        "pruned_at_startup": _pruned,
        "note": "Blurred thumbnails expire first; telemetry rows outlive them.",
    }


@app.get("/api/whitebox")
def api_whitebox() -> dict:
    """Live pipeline snapshot for the White box tab.

    The worker's trace ring is in memory, so it is empty after a restart and
    stays empty for anyone who only ever uses "Test a message" — that path
    persists a run but never ticks the worker. Falling back to the newest
    stored run means the diagram has something real to show in both cases,
    rather than an empty pipeline that looks broken.
    """
    wb = worker.whitebox()
    if not wb.get("last_trace"):
        rows = store.list_runs(limit=1)
        if rows:
            wb["last_trace"] = trace_from_run_row(rows[0])
            wb["trace_source"] = "stored run"
    return wb


class PreviewBody(BaseModel):
    blurred: bool = True
    width: Optional[int] = None


@app.post("/api/preview")
def api_preview(body: PreviewBody) -> dict:
    """Flip preview privacy without restarting.

    Only affects frames captured from now on — rows already stored keep the
    preview written at the time, because the source frame was wiped after that
    tick and cannot be re-rendered.
    """
    from analyst.store import persist as _persist

    return _persist.set_preview_mode(body.blurred, body.width)


@app.get("/api/status")
def api_status() -> dict:
    from analyst.store import persist as _persist

    st = worker.status()
    # The panel must be able to say whether it is showing a privacy-preserving
    # preview or a sharp demo capture. Leaving that ambiguous is how a reviewer
    # ends up believing the wrong thing about what the database holds.
    st["preview"] = {
        "blurred": _persist.previews_are_blurred(),
        "width": _persist.THUMB_WIDTH,
    }
    return st


@app.post("/api/capture/start")
def api_capture_start(body: StartBody) -> dict:
    return worker.start(child_age=body.age, interval_s=body.interval_s)


@app.post("/api/capture/stop")
def api_capture_stop() -> dict:
    return worker.stop()


@app.get("/api/runs")
def api_runs(limit: int = 50) -> dict:
    rows = [_public_run(r, full=False) for r in store.list_runs(limit=limit)]
    return {"runs": rows, "stats": store.stats()}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    row = store.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    return _public_run(row)


@app.post("/api/analyze")
async def api_analyze(
    age: int = Form(10),
    text: str = Form(""),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
) -> dict:
    image_bytes = await image.read() if image is not None else None
    audio_bytes = await audio.read() if audio is not None else None
    if not text.strip() and not image_bytes and not audio_bytes:
        raise HTTPException(400, "Provide text and/or image and/or audio")
    result = pipeline.analyze(
        child_age=age,
        overlay_text=text,
        image_bytes=image_bytes or None,
        audio_bytes=audio_bytes or None,
        app_exe="upload",
        app_category="manual",
    )
    run_id = persist_result(
        store,
        result,
        child_age=age,
        app_exe="upload",
        frame_bytes=image_bytes,
    )
    row = store.get_run(run_id)
    return {
        "run_id": run_id,
        "decision": result.decision,
        "run": _public_run(row) if row else None,
        "result": result.model_dump(),
    }


def main() -> None:
    import uvicorn

    print(f"Analyst live panel -> http://{HOST}:{PORT}")
    print("Temporary C2 capture - C1 will own capture later.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
