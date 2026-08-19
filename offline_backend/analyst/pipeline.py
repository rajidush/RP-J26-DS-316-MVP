"""Chained Analyst: hold media → Stage 1 → Stage 2 fusion → JSON → delete."""

from __future__ import annotations

import io
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image

from .asr import AsrEngine
from .buffer import TransientMediaBuffer
from .capture import grab_screen_jpeg
from .ocr import OcrEngine
from .schema import AnalystHealth, AnalystResult, BackendStatus
from .text_classifier import TextClassifier
from .vision_classifier import VisionClassifier

STAGE1_TEXT = 0.40
STAGE1_VISION = 0.40
HATE_GATE = 0.85
TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40


def _image_from_bytes(data: Optional[bytes]) -> Optional[Image.Image]:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def fuse(text_score: float, vision_score: float) -> float:
    """Late fusion. Mid+mid on both modalities gets a meme bump."""
    fused = TEXT_WEIGHT * text_score + VISION_WEIGHT * vision_score
    if 0.35 <= text_score < HATE_GATE and 0.35 <= vision_score < HATE_GATE:
        fused = min(1.0, fused + 0.15)
    return round(min(1.0, max(0.0, fused)), 4)


class AnalystPipeline:
    def __init__(self) -> None:
        self.buffer = TransientMediaBuffer(max_slots=2)
        self.ocr = OcrEngine()
        self.asr = AsrEngine()
        self.text = TextClassifier()
        self.vision = VisionClassifier()

    def health(self) -> AnalystHealth:
        capture_name = "pillow_grab"
        notes = []
        if self.ocr.name == "none":
            notes.append(
                "OCR not installed. Paste overlay text or add RapidOCR/Tesseract; pipeline still runs."
            )
        if self.vision.name == "deferred":
            notes.append(
                "Vision ONNX not loaded. Image-only hate waits for models/vision_stage1.onnx."
            )
        if self.asr.name == "none":
            notes.append(
                "Whisper not installed. Voice is skipped; OCR + overlay text still score."
            )
        return AnalystHealth(
            backends=self._backends(),
            notes=notes,
        )

    def _backends(self) -> BackendStatus:
        return BackendStatus(
            ocr=self.ocr.name,
            asr=self.asr.name,
            text=self.text.name,
            vision=self.vision.name,
            capture="pillow_grab",
        )

    def analyze(
        self,
        *,
        child_age: int = 10,
        overlay_text: str = "",
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        capture_screen: bool = False,
    ) -> AnalystResult:
        notes: list[str] = []
        if capture_screen and not image_bytes:
            grabbed, how = grab_screen_jpeg()
            if grabbed:
                image_bytes = grabbed
                notes.append(f"screen_capture:{how}")
            else:
                notes.append("screen_capture_unavailable")

        if not image_bytes and not (overlay_text or "").strip() and not audio_bytes:
            # Still return a valid not-hate result so the UI is not stuck.
            notes.append("no_media_or_text")
            return self._empty_result(child_age, notes)

        trigger_id = str(uuid.uuid4())
        with self.buffer.hold(trigger_id, frame=image_bytes, audio=audio_bytes) as slot:
            result = self._run_cascade(
                child_age=child_age,
                overlay_text=overlay_text or "",
                frame=slot["frame"],
                audio=slot["audio"],
                trigger_id=trigger_id,
                notes=notes,
            )
        result.media_deleted = True
        return result

    def _empty_result(self, child_age: int, notes: list[str]) -> AnalystResult:
        return AnalystResult(
            decision="not-hate",
            risk_score=0.05,
            category="none",
            child_age=child_age,
            source={"ocr": False, "asr": False, "vision": False, "overlay": False},
            session_hint="",
            stage1={"text_score": 0.05, "vision_score": 0.0},
            escalated=False,
            backends=self._backends(),
            notes=notes,
        )

    def _run_cascade(
        self,
        *,
        child_age: int,
        overlay_text: str,
        frame: Optional[bytes],
        audio: Optional[bytes],
        trigger_id: str,
        notes: list[str],
    ) -> AnalystResult:
        image = _image_from_bytes(frame)
        with ThreadPoolExecutor(max_workers=2) as pool:
            ocr_future = pool.submit(self.ocr.extract, image)
            asr_future = pool.submit(self.asr.transcribe, audio)
            ocr_text = ocr_future.result()
            transcript = asr_future.result()

        if audio and not transcript:
            notes.append("asr_empty_or_unavailable")
        if audio and not transcript and self.asr.name == "none":
            notes.append("install faster-whisper for voice")
        combined_text = " ".join(
            part for part in (overlay_text, ocr_text, transcript) if part
        ).strip()

        text_score, category, _ = self.text.score(combined_text)
        vision_score = self.vision.score(image)

        if not combined_text and image is not None and self.vision.name == "deferred":
            notes.append("image_without_text_vision_deferred")

        escalated = text_score >= STAGE1_TEXT or vision_score >= STAGE1_VISION
        stage1 = {
            "text_score": round(text_score, 4),
            "vision_score": round(vision_score, 4),
        }

        stage2 = None
        if escalated:
            fused = fuse(text_score, vision_score)
            # If vision is deferred, text carries the decision (no silent drop).
            if self.vision.name == "deferred":
                fused = max(fused, text_score)
            stage2 = {
                "text_score": round(text_score, 4),
                "vision_score": round(vision_score, 4),
                "fused": fused,
            }
            risk = fused
        else:
            risk = round(max(text_score, vision_score), 4)
            notes.append("stopped_at_stage1")

        decision = "hate" if risk > HATE_GATE else "not-hate"
        if decision == "not-hate":
            category = "none"

        return AnalystResult(
            decision=decision,
            risk_score=risk,
            category=category if decision == "hate" else "none",
            threshold=HATE_GATE,
            child_age=child_age,
            source={
                "ocr": bool(ocr_text),
                "asr": bool(transcript),
                "vision": self.vision.name not in ("deferred", "none"),
                "overlay": bool(overlay_text.strip()),
            },
            session_hint=trigger_id,
            ocr_text=ocr_text[:500],
            transcript=transcript[:500],
            overlay_text=overlay_text[:500],
            stage1=stage1,
            stage2=stage2,
            escalated=escalated,
            backends=self._backends(),
            notes=notes,
        )


_PIPELINE: Optional[AnalystPipeline] = None


def get_pipeline() -> AnalystPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = AnalystPipeline()
    return _PIPELINE
