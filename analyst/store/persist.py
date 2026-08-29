"""Map AnalystRunResult → SQLite row (+ optional blurred thumbnail)."""

from __future__ import annotations

import io
import os
import re
from typing import Optional

from PIL import Image, ImageFilter

from analyst.schemas import AnalystRunResult, new_id
from analyst.store.db import AnalystStore

_SENSITIVE = re.compile(
    r"(?i)((?:password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*)\S+"
)
_SUDO_PW = re.compile(
    r"(?i)\[sudo\]\s*password\s*for\s+\S+\s*:?\s*\S*"
)
_PASSWORD_FOR = re.compile(
    r"(?i)(password\s+for\s+\S+\s*[:=]\s*)\S+"
)


def redact_snippet(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    out = _SUDO_PW.sub("[sudo] password: [redacted]", text)
    out = _PASSWORD_FOR.sub(r"\1[redacted]", out)
    out = _SENSITIVE.sub(r"\1[redacted]", out)
    return out[:limit]


# Preview policy. The defaults are the privacy-preserving ones and should stay
# that way: a 320 px blurred JPEG shows *where* the engine looked without
# storing anything readable from a child's screen.
#
# Demo mode (ANALYST_THUMB_BLUR=0) writes a sharp, larger preview so a reviewer
# can actually read the screen behind the detection boxes. That is a real
# weakening — the database then holds legible screenshots — so it is opt-in,
# never the default, and the panel labels any run captured under it.
THUMB_WIDTH = int(os.environ.get("ANALYST_THUMB_WIDTH", "320"))
THUMB_BLUR = float(os.environ.get("ANALYST_THUMB_BLUR", "8"))

# Blur radius used when previews are switched back on. Kept separate from
# THUMB_BLUR so toggling to visible and back restores the original strength
# rather than a hardcoded guess.
_BLUR_STRENGTH = THUMB_BLUR if THUMB_BLUR > 0 else 8.0
_SHARP_WIDTH = 1100
_BLURRED_WIDTH = 320


def previews_are_blurred() -> bool:
    return THUMB_BLUR > 0


def set_preview_mode(blurred: bool, width: Optional[int] = None) -> dict:
    """Switch preview privacy at runtime.

    The env vars set the *starting* mode; this lets the panel flip it without a
    restart, which is what the demo actually needs. Only frames captured after
    the switch are affected — an existing row keeps whatever preview was written
    at the time, because the original frame is long gone from memory.
    """
    global THUMB_BLUR, THUMB_WIDTH
    THUMB_BLUR = _BLUR_STRENGTH if blurred else 0.0
    THUMB_WIDTH = int(width) if width else (_BLURRED_WIDTH if blurred else _SHARP_WIDTH)
    return {"blurred": previews_are_blurred(), "width": THUMB_WIDTH}


def make_blurred_thumb(
    frame_bytes: Optional[bytes],
    max_w: Optional[int] = None,
) -> Optional[bytes]:
    """Preview image for the panel. Never stores the full frame."""
    if not frame_bytes:
        return None
    width = max_w or THUMB_WIDTH
    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        w, h = img.size
        if w > width:
            nh = max(1, int(h * (width / w)))
            img = img.resize((width, nh), Image.Resampling.BILINEAR)
        if THUMB_BLUR > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=THUMB_BLUR))
        # A sharp preview only earns its size if the text is legible, so give
        # demo mode more quality and a proportionally larger ceiling.
        quality = 45 if THUMB_BLUR > 0 else 72
        ceiling = 48_000 if THUMB_BLUR > 0 else 220_000
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()
        if len(data) > ceiling:
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=max(30, quality - 20), optimize=True)
            data = out.getvalue()
        return data
    except Exception:
        return None


def persist_result(
    store: AnalystStore,
    result: AnalystRunResult,
    *,
    child_age: int = 10,
    app_exe: str = "unknown",
    frame_bytes: Optional[bytes] = None,
    trace: Optional[dict] = None,
) -> str:
    payload = result.payload
    score = float(result.risk_score)
    category = "none"
    summary = ""
    action = ""
    modalities = dict(result.modalities or {})
    lexicon_hits = list(result.lexicon_hits or [])

    if payload is not None:
        score = float(payload.score)
        category = payload.category
        summary = payload.child_safe_summary
        action = payload.recommended_action
        modalities = payload.modalities.model_dump()
        lexicon_hits = list(payload.evidence.lexicon_hits or lexicon_hits)
    elif result.stage1 and score <= 0:
        score = float(
            max(result.stage1.get("text_score", 0.0), result.stage1.get("vision_score", 0.0))
        )

    thumb = make_blurred_thumb(frame_bytes)
    run_id = new_id()
    store.insert_run(
        {
            "id": run_id,
            "decision": result.decision,
            "category": category,
            "score": score,
            "child_age": child_age,
            "ocr_snippet": redact_snippet(result.ocr_text or ""),
            "transcript_snippet": redact_snippet(result.transcript or ""),
            "image_caption": redact_snippet(result.image_caption or "", 300),
            "image_text": redact_snippet(result.image_text or "", 300),
            "lexicon_hits": lexicon_hits,
            "stage1": result.stage1,
            "stage2": result.stage2,
            "backends": result.backends,
            "latency_ms": result.latency_ms,
            "notes": list(result.notes) + (
                [f"explanation:{result.explanation}"] if result.explanation else []
            ),
            "app_exe": app_exe,
            "modalities": modalities,
            "child_safe_summary": summary or (
                "Monitoring looks clear right now." if result.decision == "not-hate" else ""
            ),
            "recommended_action": action,
            "thumb_jpeg": thumb,
            "envelope": result.envelope.model_dump() if result.envelope else None,
            "trace": trace,
            "evidence": build_evidence(result),
        }
    )
    return run_id


def build_evidence(result: AnalystRunResult) -> dict:
    """Why this run scored what it did — the audit trail behind the number.

    Kept structured rather than folded into `notes` so the panel can render it
    and a reviewer can answer "which detector fired, and was the score adjusted"
    without parsing prose.
    """
    return {
        "lexicon_score": round(float(result.lexicon_score or 0.0), 4),
        "model_score": (
            round(float(result.model_score), 4) if result.model_score is not None else None
        ),
        "model_labels": dict(result.model_labels or {}),
        "framing_reason": result.framing_reason or "",
        "score_before_framing": (
            round(float(result.score_before_framing), 4)
            if result.score_before_framing is not None
            else None
        ),
        "vision_calibrated": bool(result.vision_calibrated),
        "fusion_mode": result.fusion_mode or "idle",
        "escalated": bool(result.escalated),
        "explanation": result.explanation or "",
        # Capped: a busy desktop can yield 50+ text lines and this rides along
        # in every stored row.
        "detections": list(result.detections or [])[:40],
    }
