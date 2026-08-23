"""Map AnalystRunResult → SQLite row (+ optional blurred thumbnail)."""

from __future__ import annotations

import io
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


def make_blurred_thumb(frame_bytes: Optional[bytes], max_w: int = 320) -> Optional[bytes]:
    """Privacy-safe preview: small + blurred JPEG. Never stores full frame."""
    if not frame_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        w, h = img.size
        if w > max_w:
            nh = max(1, int(h * (max_w / w)))
            img = img.resize((max_w, nh), Image.Resampling.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=8))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=45, optimize=True)
        data = out.getvalue()
        if len(data) > 48_000:
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=30, optimize=True)
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
        }
    )
    return run_id
