"""ASR plug. faster-whisper tiny if installed; otherwise skip audio (score text from OCR)."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

_MODEL = None
_MODEL_NAME = "none"


def asr_status() -> str:
    return _MODEL_NAME


def transcribe(audio_bytes: Optional[bytes]) -> str:
    global _MODEL, _MODEL_NAME
    if not audio_bytes:
        return ""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        _MODEL_NAME = "none"
        return ""

    try:
        if _MODEL is None:
            _MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
            _MODEL_NAME = "faster-whisper-tiny"
        # faster-whisper wants a path or numpy PCM; use a private temp and unlink.
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open(path, "wb") as handle:
                handle.write(audio_bytes)
            segments, _ = _MODEL.transcribe(path, beam_size=1)
            return " ".join(seg.text for seg in segments).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        _MODEL_NAME = "whisper_failed"
        return ""
