"""ASR plug. faster-whisper tiny on CPU if installed; otherwise skip audio."""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

_WHISPER_IMPORT_OK = False
try:
    from faster_whisper import WhisperModel  # noqa: F401

    _WHISPER_IMPORT_OK = True
except Exception:
    WhisperModel = None  # type: ignore[misc, assignment]


class AsrEngine:
    """Lazy-loads Whisper tiny on first clip so FastAPI startup stays fast."""

    def __init__(self, transcribe_fn: Optional[Callable[[bytes], str]] = None) -> None:
        self._override = transcribe_fn
        self._model = None
        if self._override is not None:
            self.name = "injected"
        elif _WHISPER_IMPORT_OK:
            self.name = "faster-whisper-tiny"
        else:
            self.name = "none"

    def transcribe(self, audio_bytes: Optional[bytes]) -> str:
        if not audio_bytes:
            return ""
        if self._override is not None:
            try:
                return (self._override(audio_bytes) or "").strip()
            except Exception:
                return ""
        if not _WHISPER_IMPORT_OK:
            return ""
        try:
            text = self._whisper(audio_bytes)
            return (text or "").strip()
        except Exception:
            self.name = "whisper_failed"
            return ""

    def _whisper(self, audio_bytes: bytes) -> str:
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self.name = "faster-whisper-tiny"

        suffix = _guess_suffix(audio_bytes)
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with open(path, "wb") as handle:
                handle.write(audio_bytes)
            segments, _ = self._model.transcribe(path, beam_size=1, vad_filter=True)
            return " ".join(seg.text for seg in segments)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def _guess_suffix(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    return ".wav"
