"""Core Analyst cascade — Engineering Plan §4.2 decision flow.

Channel model
-------------
Text and vision are separate *channels*, and what belongs to each is decided by
how the evidence was produced, not by where it came from:

    text channel    overlay text, OCR, speech transcript, and any words a
                    vision-language model read out of a picture. All of it is
                    language, so all of it is scored by the same text scorer.

    vision channel  judgements made about pixels themselves (image_fast).
                    Currently uncalibrated, so it is shown but does not move
                    the score — see stage1/image_fast.py and stage2/fusion.py.

The previous version filed VLM-read text under "vision", which mixed a
calibrated text score with an uncalibrated image score in the same number and
made the fused result impossible to reason about.

Hardened for live capture: every branch isolates its failures, so a broken OCR
or model install degrades the run instead of killing the tick.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from PIL import Image

from .buffer import TransientMediaBuffer
from .decide import STAGE1_THETA, decide
from .extract.asr import AsrEngine
from .extract.embed import ImageEmbedder
from .extract.ocr import OcrEngine
from .extract.vision_meaning import VisionMeaning
from .schemas import AnalystRunResult, new_id
from .stage1.image_fast import ImageFast
from .stage1.text_fast import ScoreDetail, TextFast
from .stage2.fusion import Signal, TextFull, explain_fusion, fuse_signals


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
        # Stage 2 reuses the Stage-1 scorer rather than loading a third model:
        # its job is to re-read text Stage 1 truncated, not to hold an opinion
        # of its own. See stage2.fusion.TextFull.
        self.text_full.bind(lambda chunk: self.text_fast.score(chunk)[0])
        self.vision_meaning = VisionMeaning()

    def backends(self) -> dict:
        return {
            "ocr": self.ocr.name,
            "asr": self.asr.name,
            "text_fast": self.text_fast.name,
            "image_fast": self.image_fast.name,
            "clip": self.embed.name,
            "text_full": self.text_full.name,
            "vision_meaning": self.vision_meaning.name,
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
                protection_state="protected",
                explanation="Nothing to analyse — no text, image or audio in this tick.",
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

    # -- extraction -----------------------------------------------------------

    def _extract(self, image, audio, notes: list, latency: dict):
        """OCR and ASR concurrently; neither may take the tick down with it."""

        def _timed_ocr():
            t = time.perf_counter()
            try:
                text = self.ocr.extract(image) if image is not None else ""
            except Exception as exc:
                notes.append(f"ocr_failed:{type(exc).__name__}")
                text = ""
            return text, round((time.perf_counter() - t) * 1000, 1)

        def _timed_asr():
            t = time.perf_counter()
            try:
                text = self.asr.transcribe(audio) if audio else ""
            except Exception as exc:
                notes.append(f"asr_failed:{type(exc).__name__}")
                text = ""
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
        return ocr_text, transcript

    def _read_picture(self, image, notes: list, latency: dict):
        """Vision-language reading of a meme/poster, if the branch is enabled.

        Not gated on escalation: a meme is precisely the case where Stage 1 sees
        nothing, so gating would close the branch exactly when it is needed. The
        cost control is the crop (extract/region.py), which also stops a small
        VLM inventing content about browser chrome.
        """
        if not (self.vision_meaning.enabled and image is not None):
            return None
        try:
            reading = self.vision_meaning.read(image)
        except Exception as exc:
            notes.append(f"vision_meaning_failed:{type(exc).__name__}")
            return None
        latency["vlm_ms"] = reading.ms
        if reading.error:
            notes.append("vision_meaning_unavailable")
        elif "no_image_region" in reading.notes:
            notes.append("no_image_region")
        elif reading.any_text:
            notes.append("vision_meaning_read")
        return reading

    # -- main ------------------------------------------------------------------

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

        ocr_text, transcript = self._extract(image, audio, notes, latency)
        reading = self._read_picture(image, notes, latency)

        # --- text channel: every source of language, scored by one scorer ----
        picture_words = reading.combined() if reading else ""
        text_parts = [p for p in (overlay_text, ocr_text, transcript, picture_words) if p]
        text_blob = " ".join(text_parts).strip()

        t_s1 = time.perf_counter()
        try:
            detail = self.text_fast.score_detailed(text_blob)
        except Exception as exc:
            notes.append(f"text_fast_failed:{type(exc).__name__}")
            detail = ScoreDetail(score=0.0, category="none")
        text_score, category, hits = detail.score, detail.category, list(detail.hits)

        # --- vision channel: judgements about pixels --------------------------
        t_clip = time.perf_counter()
        try:
            emb = self.embed.embed(image) if image is not None else []
        except Exception as exc:
            notes.append(f"clip_failed:{type(exc).__name__}")
            emb = []
        latency["clip_ms"] = round((time.perf_counter() - t_clip) * 1000, 1)

        try:
            vision_score = self.image_fast.score(embedding=emb, image=image)
        except Exception as exc:
            notes.append(f"image_fast_failed:{type(exc).__name__}")
            vision_score = 0.0
        vision_calibrated = bool(getattr(self.image_fast, "calibrated", False))
        latency["stage1_ms"] = round((time.perf_counter() - t_s1) * 1000, 1)

        if vision_score > 0.0 and not vision_calibrated:
            notes.append("vision_uncalibrated_excluded")

        self._note_channels(ocr_text, transcript, picture_words, image, notes)

        stage1 = {
            "text_score": round(text_score, 4),
            "lexicon_score": round(detail.lexicon_score, 4),
            "model_score": round(detail.model_score, 4) if detail.model_score is not None else 0.0,
            "vision_score": round(vision_score, 4),
        }

        # --- escalation: only calibrated evidence may escalate ---------------
        vision_escalates = vision_calibrated and vision_score >= STAGE1_THETA
        escalated = text_score >= STAGE1_THETA or vision_escalates

        # Vision-only: a picture carried the signal and there were no words to
        # read. The category cannot come from the text scorer, so it comes from
        # what the vision branch is trained to look for (hateful imagery).
        # Uncalibrated vision can never reach here — it does not escalate.
        if vision_escalates and not text_blob:
            notes.append("vision_only_escalation")
            if category == "none":
                category = "hate_identity"

        if not escalated:
            notes.append("stopped_at_stage1")

        # --- stage 2 ----------------------------------------------------------
        t_s2 = time.perf_counter()
        full_text_score = text_score
        if escalated:
            try:
                full_text_score = self.text_full.score(text_blob, text_score)
                if full_text_score > text_score:
                    notes.append("stage2_found_more_in_full_text")
            except Exception as exc:
                notes.append(f"text_full_failed:{type(exc).__name__}")

        fusion = fuse_signals(
            [
                Signal("text", full_text_score, True, "ocr/asr/overlay/vlm"),
                Signal("vision", vision_score, vision_calibrated, self.image_fast.name),
            ]
        )
        latency["stage2_ms"] = round((time.perf_counter() - t_s2) * 1000, 1)

        # stage2 stays numeric — the mode is carried by result.fusion_mode.
        stage2 = dict(fusion.as_dict())
        stage2["text_full"] = round(full_text_score, 4)
        if not escalated:
            stage2["preview"] = 1.0

        risk = fusion.fused if escalated else round(max(text_score, 0.0), 4)

        decision, envelope, payload, cleared, mods = decide(
            text_score=text_score,
            vision_score=vision_score if vision_calibrated else 0.0,
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

        risk_out = float(payload.score) if payload is not None else round(text_score, 4)

        if decision == "hate":
            protection = "threat"
        elif cleared is not None:
            protection = "reviewing"
        elif any("failed:" in n for n in notes):
            protection = "degraded"
        else:
            protection = "protected"

        explanation = explain_fusion(
            text_score=text_score,
            vision_score=vision_score,
            fused=risk,
            escalated=escalated,
            decision=decision,
            has_ocr=bool(ocr_text.strip()),
            has_asr=bool(transcript.strip()),
            lexicon_hits=hits,
            vision_calibrated=vision_calibrated,
            framing_reason=detail.framing_reason,
            model_score=detail.model_score,
        )

        latency["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return AnalystRunResult(
            decision=decision,
            envelope=envelope,
            payload=payload,
            cleared=cleared,
            ocr_text=ocr_text[:500],
            image_caption=(reading.caption[:400] if reading else ""),
            image_text=(reading.image_text[:400] if reading else ""),
            image_region=(list(reading.box) if reading and reading.box else None),
            transcript=transcript[:500],
            stage1=stage1,
            stage2=stage2,
            backends=self.backends(),
            media_deleted=True,
            notes=notes,
            latency_ms=latency,
            modalities=mods.model_dump(),
            risk_score=risk_out,
            explanation=explanation,
            protection_state=protection,  # type: ignore[arg-type]
            lexicon_hits=list(hits[:10]),
            lexicon_score=round(detail.lexicon_score, 4),
            model_score=detail.model_score,
            model_labels=dict(detail.model_labels or {}),
            framing_reason=detail.framing_reason,
            score_before_framing=detail.discounted_from,
            vision_calibrated=vision_calibrated,
            fusion_mode=fusion.mode,
            escalated=escalated,
        )

    @staticmethod
    def _note_channels(
        ocr_text: str,
        transcript: str,
        picture_words: str,
        image,
        notes: List[str],
    ) -> None:
        if ocr_text and transcript:
            notes.append("multimodal_ocr_asr")
        elif transcript and not ocr_text:
            notes.append("asr_primary")
        elif ocr_text and not transcript:
            notes.append("ocr_primary")
        if picture_words:
            notes.append("picture_text_read")
        if image is not None and not ocr_text and not picture_words:
            notes.append("image_without_readable_text")
