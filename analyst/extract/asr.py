"""ASR — faster-whisper tiny/base.en on CPU. Missing package = skip, no crash.

Privacy rule (Engineering Plan): raw audio never touches disk. faster-whisper
accepts a binary file-like object, so we decode straight from RAM.
"""

from __future__ import annotations

import io
from typing import Callable, Optional

# Segment-level confidence floors. Whisper reports both; silence typically
# yields no_speech_prob near 1.0 and a very negative avg_logprob.
NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -1.0

_WHISPER_OK = False
try:
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
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

        # RAM-only handle; closed (and dropped) before we return.
        stream = io.BytesIO(audio_bytes)
        try:
            # Whisper hallucinates fluent text out of silence and room tone —
            # on a quiet desktop it was emitting Japanese ("私は"), which then
            # scored as real speech. Three guards:
            #   language="en"                 v1 ships English classifiers only,
            #                                 so never let it drift to another one
            #   condition_on_previous_text    stops one hallucination seeding the next
            #   no_speech_prob per segment    drop segments the model itself
            #                                 believes are not speech
            segments, _info = self._model.transcribe(
                stream,
                beam_size=1,
                vad_filter=True,
                language="en",
                condition_on_previous_text=False,
            )
            kept = []
            for seg in segments:
                if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_MAX:
                    continue
                if getattr(seg, "avg_logprob", 0.0) < AVG_LOGPROB_MIN:
                    continue
                text = (seg.text or "").strip()
                if text:
                    kept.append(text)
            return " ".join(kept)
        finally:
            stream.close()
