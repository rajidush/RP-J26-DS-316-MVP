"""Pipeline step trace — human-readable blackbox → whitebox log."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from analyst.schemas import AnalystRunResult


def _step(step_id: str, label: str, status: str, ms: float = 0.0, detail: str = "") -> dict:
    return {
        "id": step_id,
        "label": label,
        "status": status,  # ok | skip | fail | warn
        "ms": round(float(ms), 1),
        "detail": detail,
    }


def _status_from_notes(notes: List[str], prefix: str) -> str:
    for n in notes:
        if n.startswith(prefix):
            return "fail"
    return "ok"


def build_trace_from_result(
    *,
    ts: str,
    tick: int,
    run_id: str,
    app: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    had_frame: bool,
    had_audio: bool,
    screen_ok: bool,
    screen_error: str,
    capture_ms: float,
    result: AnalystRunResult,
) -> dict:
    lat = result.latency_ms or {}
    s1 = result.stage1 or {}
    s2 = result.stage2 or {}
    notes = list(result.notes or [])

    steps: List[dict] = []

    # No frame is only a failure when a frame was actually expected. A manual
    # text check has nothing to capture, and reporting that as "fail" made a
    # perfectly healthy run look broken in the panel.
    if had_frame:
        cap_status = "ok" if screen_ok else "warn"
        cap_detail = f"{frame_w}×{frame_h}" if frame_w else "—"
    elif screen_error:
        cap_status, cap_detail = "fail", screen_error[:80]
    else:
        cap_status, cap_detail = "skip", "not a screen check"
    steps.append(_step("capture", "Screen grab", cap_status, capture_ms, cap_detail))

    audio_status = "ok" if had_audio else "skip"
    audio_detail = f"{len(result.transcript or '')} chars" if had_audio else "silent"
    steps.append(_step("audio", "Loopback audio", audio_status, lat.get("asr_ms", 0) if had_audio else 0, audio_detail))

    ocr_status = _status_from_notes(notes, "ocr_failed")
    if not result.ocr_text and not any(n.startswith("ocr_failed") for n in notes):
        ocr_status = "skip" if not had_frame else "ok"
    ocr_detail = f"{len(result.ocr_text)} chars" if result.ocr_text else ("empty" if had_frame else "—")
    steps.append(_step("ocr", "OCR text", ocr_status, lat.get("ocr_ms", 0), ocr_detail))

    clip_status = _status_from_notes(notes, "clip_failed")
    if not had_frame:
        clip_status = "skip"
    steps.append(_step("clip", "Image embed (CLIP)", clip_status, lat.get("clip_ms", 0), result.backends.get("clip", "—")))

    # Vision meaning (local VLM). Only present when the branch is enabled and
    # the gate opened, so the panel can show whether the meme path actually ran.
    vlm_backend = result.backends.get("vision_meaning", "none")
    if vlm_backend and vlm_backend != "none":
        if "vision_meaning_used" in notes:
            vlm_status, vlm_detail = "ok", vlm_backend.replace("lmstudio:", "")
        elif "vision_meaning_unavailable" in notes:
            vlm_status, vlm_detail = "fail", "model unreachable"
        elif "vlm_ms" in lat:
            vlm_status, vlm_detail = "ok", "no added signal"
        else:
            vlm_status, vlm_detail = "skip", "gate closed"
        steps.append(
            _step("vlm", "Image meaning", vlm_status, lat.get("vlm_ms", 0), vlm_detail)
        )

    ts1 = s1.get("text_score", 0.0)
    vs1 = s1.get("vision_score", 0.0)
    steps.append(
        _step(
            "stage1",
            "Stage-1 scan",
            "ok",
            lat.get("stage1_ms", 0),
            f"text {ts1:.2f} · vision {vs1:.2f}",
        )
    )

    fused = (s2 or {}).get("fused", 0.0)
    escalated = "stopped_at_stage1" not in notes
    s2_label = "Fusion (Stage-2)" if escalated else "Fusion preview"
    steps.append(
        _step(
            "stage2",
            s2_label,
            "ok",
            lat.get("stage2_ms", 0),
            f"fused {fused:.2f}" if s2 else "—",
        )
    )

    decision_label = "Alert" if result.decision == "hate" else "Clear"
    steps.append(
        _step(
            "decide",
            "Decision",
            "warn" if result.decision == "hate" else "ok",
            0,
            f"{decision_label} · risk {result.risk_score:.2f}",
        )
    )

    return {
        "ts": ts,
        "tick": tick,
        "run_id": run_id,
        "outcome": "analyzed",
        "app": dict(app),
        "decision": result.decision,
        "risk_score": result.risk_score,
        "explanation": result.explanation,
        "steps": steps,
        "total_ms": lat.get("total_ms", 0),
    }


def build_trace_skipped(
    *,
    ts: str,
    tick: int,
    app: Dict[str, Any],
    reason: str = "duplicate frame, no new audio",
) -> dict:
    return {
        "ts": ts,
        "tick": tick,
        "run_id": None,
        "outcome": "skipped",
        "app": dict(app),
        "decision": None,
        "risk_score": None,
        "explanation": reason,
        "steps": [
            _step("capture", "Screen grab", "ok", 0, "same frame as last tick"),
            _step("audio", "Loopback audio", "skip", 0, "silent"),
            _step("ocr", "OCR text", "skip", 0, "skipped — no change"),
            _step("clip", "Image embed (CLIP)", "skip", 0, "—"),
            _step("vlm", "Image meaning", "skip", 0, "—"),
            _step("stage1", "Stage-1 scan", "skip", 0, "—"),
            _step("stage2", "Fusion", "skip", 0, "—"),
            _step("decide", "Decision", "skip", 0, reason),
        ],
        "total_ms": 0,
    }


def build_trace_failed(
    *,
    ts: str,
    tick: int,
    app: Dict[str, Any],
    error: str,
    capture_ms: float = 0.0,
) -> dict:
    return {
        "ts": ts,
        "tick": tick,
        "run_id": None,
        "outcome": "capture_failed",
        "app": dict(app),
        "decision": None,
        "risk_score": None,
        "explanation": error,
        "steps": [
            _step("capture", "Screen grab", "fail", capture_ms, error[:80]),
            _step("audio", "Loopback audio", "skip", 0, "—"),
            _step("ocr", "OCR text", "skip", 0, "no input"),
            _step("clip", "Image embed (CLIP)", "skip", 0, "—"),
            _step("stage1", "Stage-1 scan", "skip", 0, "—"),
            _step("stage2", "Fusion", "skip", 0, "—"),
            _step("decide", "Decision", "skip", 0, "waiting for capture"),
        ],
        "total_ms": capture_ms,
    }


def trace_from_run_row(row: Dict[str, Any]) -> dict:
    """Rebuild whitebox trace from a stored DB run (for log replay)."""
    app = {"exe": row.get("app_exe") or "unknown", "title": "", "title_hash": ""}
    lat = row.get("latency_ms") or {}
    s1 = row.get("stage1") or {}
    s2 = row.get("stage2") or {}
    notes = row.get("notes") or []
    decision = row.get("decision") or "not-hate"
    score = float(row.get("score") or 0.0)
    ocr = row.get("ocr_snippet") or ""
    asr = row.get("transcript_snippet") or ""
    expl = ""
    for n in notes:
        if str(n).startswith("explanation:"):
            expl = str(n).replace("explanation:", "", 1)
            break

    fake = AnalystRunResult(
        decision=decision,  # type: ignore[arg-type]
        ocr_text=ocr,
        transcript=asr,
        stage1=s1,
        stage2=s2,
        backends=row.get("backends") or {},
        notes=notes,
        latency_ms=lat,
        risk_score=score,
        explanation=expl,
        lexicon_hits=row.get("lexicon_hits") or [],
    )
    return build_trace_from_result(
        ts=row.get("ts") or "",
        tick=0,
        run_id=row.get("id") or "",
        app=app,
        frame_w=0,
        frame_h=0,
        had_frame=bool(ocr or row.get("thumb_jpeg")),
        had_audio=bool(asr),
        screen_ok=True,
        screen_error="",
        capture_ms=0,
        result=fake,
    )


class TraceBuffer:
    """Thread-safe ring buffer of recent pipeline traces."""

    def __init__(self, maxlen: int = 30) -> None:
        self._maxlen = max(5, int(maxlen))
        self._items: Deque[dict] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def push(self, trace: dict) -> None:
        with self._lock:
            self._items.appendleft(trace)

    def list(self, limit: int = 20) -> List[dict]:
        limit = max(1, min(int(limit), self._maxlen))
        with self._lock:
            return list(self._items)[:limit]

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._items[0] if self._items else None

    def recent_apps(self, limit: int = 8) -> List[dict]:
        """Unique foreground apps from recent traces (newest first)."""
        seen: set[str] = set()
        out: List[dict] = []
        with self._lock:
            for tr in self._items:
                app = tr.get("app") or {}
                exe = str(app.get("exe") or "unknown")
                if exe in seen:
                    continue
                seen.add(exe)
                out.append(
                    {
                        "exe": exe,
                        "title": app.get("title") or "",
                        "last_ts": tr.get("ts"),
                        "last_outcome": tr.get("outcome"),
                        "last_decision": tr.get("decision"),
                    }
                )
                if len(out) >= limit:
                    break
        return out
