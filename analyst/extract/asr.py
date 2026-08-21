"""ASR — faster-whisper tiny/base.en on CPU. Missing package = skip, no crash."""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

_WHISPER_OK = False
try:
    from faster_whisper import WhisperModel  # noqa: F401

    _WHISPER_OK = True
except Exception:
    WhisperModel = None  # type: ignore


class AsrEngine:
    def __init__(
        self,
        model_size: str = "tiny",
        transcribe_fn: Optional[Callable[[bytes], str]] = None,
    ) -> None:
        self.model_size = model_size
        self._override = transcribe_fn
        self._model = None
        if self._override is not None:
            self.name = "injected"
        elif _WHISPER_OK:
            self.name = f"faster-whisper-{model_size}"
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
        if not _WHISPER_OK:
            return ""
        try:
            return (self._whisper(audio_bytes) or "").strip()
        except Exception:
            self.name = "whisper_failed"
            return ""

    def _whisper(self, audio_bytes: bytes) -> str:
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.name = f"faster-whisper-{self.model_size}"

        suffix = ".wav" if audio_bytes[:4] == b"RIFF" else ".webm"
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
