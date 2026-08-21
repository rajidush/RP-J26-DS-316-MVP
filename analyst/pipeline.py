"""Core Analyst cascade — Engineering Plan §4.2 decision flow."""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image

from .buffer import TransientMediaBuffer
from .decide import STAGE1_THETA, decide
from .extract.asr import AsrEngine
from .extract.embed import ImageEmbedder
from .extract.ocr import OcrEngine
from .schemas import AnalystRunResult, new_id
from .stage1.image_fast import ImageFast
from .stage1.text_fast import TextFast
from .stage2.fusion import TextFull, fuse, fusion_detail


def _image_from_bytes(data: Optional[bytes]) -> Optional[Image.Image]:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


class AnalystPipeline:
    def __init__(self) -> None:
        self.buffer = TransientMediaBuffer(max_slots=2)
        self.ocr = OcrEngine()
        self.asr = AsrEngine(model_size="tiny")
        self.embed = ImageEmbedder()
        self.text_fast = TextFast()
        self.image_fast = ImageFast(embedder=self.embed)
        self.text_full = TextFull()

    def backends(self) -> dict:
        return {
            "ocr": self.ocr.name,
            "asr": self.asr.name,
            "text_fast": self.text_fast.name,
            "image_fast": self.image_fast.name,
            "clip": self.embed.name,
            "text_full": self.text_full.name,
        }

    def analyze(
        self,
        *,
        child_age: int = 10,
        overlay_text: str = "",
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        app_exe: str = "unknown",
        app_category: str = "other",
        corr: Optional[str] = None,
    ) -> AnalystRunResult:
        notes: list[str] = []
        t0 = time.perf_counter()

        if not image_bytes and not (overlay_text or "").strip() and not audio_bytes:
            notes.append("no_media_or_text")
            return AnalystRunResult(
                decision="not-hate",
                backends=self.backends(),
                notes=notes,
            )

        trigger_id = corr or new_id()
        with self.buffer.hold(trigger_id, frame=image_bytes, audio=audio_bytes) as slot:
            result = self._run(
                child_age=child_age,
                overlay_text=overlay_text or "",
                frame=slot["frame"],
                audio=slot["audio"],
                app_exe=app_exe,
                app_category=app_category,
                corr=trigger_id,
                notes=notes,
                t0=t0,
            )
        result.media_deleted = True
        return result

    def _run(
        self,
        *,
        child_age: int,
        overlay_text: str,
        frame: Optional[bytes],
        audio: Optional[bytes],
        app_exe: str,
        app_category: str,
        corr: str,
        notes: list[str],
        t0: float,
    ) -> AnalystRunResult:
        image = _image_from_bytes(frame)
        latency: dict[str, float] = {}

        def _timed_ocr():
            t = time.perf_counter()
            text = self.ocr.extract(image)
            return text, round((time.perf_counter() - t) * 1000, 1)

        def _timed_asr():
            t = time.perf_counter()
            text = self.asr.transcribe(audio)
            return text, round((time.perf_counter() - t) * 1000, 1)

        t_extract = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            ocr_f = pool.submit(_timed_ocr)
            asr_f = pool.submit(_timed_asr)
            ocr_text, ocr_ms = ocr_f.result()
            transcript, asr_ms = asr_f.result()
        latency["ocr_ms"] = ocr_ms
        latency["asr_ms"] = asr_ms
        latency["extract_ms"] = round((time.perf_counter() - t_extract) * 1000, 1)

        combined = " ".join(p for p in (overlay_text, ocr_text, transcript) if p).strip()
        t_clip = time.perf_counter()
        emb = self.embed.embed(image)
        latency["clip_ms"] = round((time.perf_counter() - t_clip) * 1000, 1)

        t_s1 = time.perf_counter()
        text_score, category, hits = self.text_fast.score(combined)
        vision_score = self.image_fast.score(embedding=emb, image=image)
        latency["stage1_ms"] = round((time.perf_counter() - t_s1) * 1000, 1)

        if not combined and image is not None and self.image_fast.name == "deferred":
            notes.append("image_without_text_vision_deferred")
        elif not combined and vision_score >= STAGE1_THETA:
            notes.append("vision_only_escalation")
            if category == "none":
                category = "hate_identity"

        escalated = text_score >= STAGE1_THETA or vision_score >= STAGE1_THETA
        stage1 = {
            "text_score": round(text_score, 4),
            "vision_score": round(vision_score, 4),
        }

        stage2 = None
        if escalated:
            t_s2 = time.perf_counter()
            full = self.text_full.score(combined, text_score)
            fused = fuse(full, vision_score)
            # Do not dilute a strong text hit when vision is inactive / deferred.
            if self.image_fast.name == "deferred" or vision_score <= 0.0:
                fused = max(fused, full)
            # Vision-only: text_full may be ~0; keep vision signal.
            if not combined and vision_score >= STAGE1_THETA:
                fused = max(fused, vision_score)
            latency["stage2_ms"] = round((time.perf_counter() - t_s2) * 1000, 1)
            detail = fusion_detail(full, vision_score)
            stage2 = {
                "text_full": round(full, 4),
                "vision_score": round(vision_score, 4),
                "fused": fused,
                "text_weight": float(detail["text_weight"]),
                "vision_weight": float(detail["vision_weight"]),
                "weighted": float(detail["weighted"]),
                "meme_bump": 1.0 if detail["meme_bump"] else 0.0,
            }
            risk = fused
        else:
            risk = round(max(text_score, vision_score), 4)
            notes.append("stopped_at_stage1")

        decision, envelope, payload, cleared = decide(
            text_score=text_score,
            vision_score=vision_score,
            fused_score=risk,
            category=category if escalated else "none",
            escalated=escalated,
            child_age=child_age,
            ocr_text=ocr_text,
            transcript=transcript,
            lexicon_hits=hits,
            app_exe=app_exe,
            app_category=app_category,
            corr=corr,
        )

        latency["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return AnalystRunResult(
            decision=decision,
            envelope=envelope,
            payload=payload,
            cleared=cleared,
            ocr_text=ocr_text[:500],
            transcript=transcript[:500],
            stage1=stage1,
            stage2=stage2,
            backends=self.backends(),
            media_deleted=True,
            notes=notes,
            latency_ms=latency,
        )
