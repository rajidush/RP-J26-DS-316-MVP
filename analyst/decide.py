"""Map scores → hate.detected / hate.cleared + child_safe_summary."""

from __future__ import annotations

from typing import Optional, Tuple

from .schemas import (
    CHILD_SAFE_SUMMARIES,
    Category,
    Envelope,
    Evidence,
    HateClearedPayload,
    HateDetectedPayload,
    Modalities,
    RecommendedAction,
    new_id,
)

# Persona thresholds from eng plan §4.4
PERSONA_THETA2 = {
    "8-10": 0.55,
    "11-13": 0.65,
    "14-15": 0.75,
}

STAGE1_THETA = 0.35


def persona_for_age(age: int) -> str:
    if age <= 10:
        return "8-10"
    if age <= 13:
        return "11-13"
    return "14-15"


def map_category(raw: str) -> Category:
    allowed = {
        "bullying",
        "hate_identity",
        "threat",
        "sexual_harassment",
        "profanity",
        "none",
    }
    if raw in allowed:
        return raw  # type: ignore[return-value]
    legacy = {
        "cyberbullying": "bullying",
        "identity_attack": "hate_identity",
        "hate_speech": "hate_identity",
    }
    return legacy.get(raw, "bullying")  # type: ignore[return-value]


def recommended_action(category: Category, score: float) -> RecommendedAction:
    if category == "profanity":
        return "notify_only"
    if category == "threat" or score >= 0.9:
        return "block"
    if score >= 0.7:
        return "blur_region"
    return "notify_only"


def modality_weights(
    text_score: float,
    vision_score: float,
    ocr_text: str,
    transcript: str,
) -> Modalities:
    """Split contribution across text / image / audio for the panel.

    Prior bug: OCR+ASR together zeroed audio weight. Fixed by splitting the
    text-channel score across OCR and ASR when both are present.
    """
    t = max(0.0, float(text_score))
    v = max(0.0, float(vision_score))
    has_ocr = bool((ocr_text or "").strip())
    has_asr = bool((transcript or "").strip())

    if has_asr and has_ocr:
        text_part, audio_part, image_part = 0.55 * t, 0.45 * t, v
    elif has_asr and not has_ocr:
        text_part, audio_part, image_part = 0.0, t, v
    else:
        text_part, audio_part, image_part = t, 0.0, v

    total = text_part + audio_part + image_part
    if total < 1e-6:
        return Modalities(text=0.0, image=0.0, audio=0.0)
    return Modalities(
        text=round(text_part / total, 3),
        image=round(image_part / total, 3),
        audio=round(audio_part / total, 3),
    )


def decide(
    *,
    text_score: float,
    vision_score: float,
    fused_score: float,
    category: str,
    escalated: bool,
    child_age: int,
    ocr_text: str,
    transcript: str,
    lexicon_hits: list,
    app_exe: str = "unknown",
    app_category: str = "other",
    corr: Optional[str] = None,
) -> Tuple[str, Optional[Envelope], Optional[HateDetectedPayload], Optional[HateClearedPayload], Modalities]:
    persona = persona_for_age(child_age)
    theta2 = PERSONA_THETA2[persona]
    cat = map_category(category)
    mods = modality_weights(text_score, vision_score, ocr_text, transcript)

    if escalated and fused_score >= theta2 and cat != "none":
        payload = HateDetectedPayload(
            score=round(fused_score, 4),
            category=cat,
            stage=2,
            modalities=mods,
            app={"exe": app_exe, "category": app_category},
            evidence=Evidence(
                ocr_snippet=(ocr_text or "")[:200],
                transcript_snippet=(transcript or "")[:200],
                lexicon_hits=lexicon_hits[:10],
            ),
            child_safe_summary=CHILD_SAFE_SUMMARIES.get(cat, CHILD_SAFE_SUMMARIES["bullying"]),
            recommended_action=recommended_action(cat, fused_score),
            persona_threshold=theta2,
        )
        env = Envelope(
            topic="hate.detected",
            corr=corr or new_id(),
            payload=payload.model_dump(),
        )
        return "hate", env, payload, None, mods

    if escalated:
        cleared = HateClearedPayload(
            stage1_score=round(max(text_score, vision_score), 4),
            stage2_score=round(fused_score, 4),
            reason="below_persona_threshold" if fused_score < theta2 else "category_none",
        )
        env = Envelope(
            topic="hate.cleared",
            corr=corr or new_id(),
            payload=cleared.model_dump(),
        )
        return "not-hate", env, None, cleared, mods

    # Stopped at stage 1: nothing escalated, so there is no false positive to report.
    return "not-hate", None, None, None, mods
