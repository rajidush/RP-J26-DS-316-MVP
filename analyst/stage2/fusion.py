"""Late fusion head. Full MLP ONNX later; rule-based fusion works now.

Failure modes handled here:
- vision inactive (0 / deferred) → do not dilute strong text
- text-only / vision-only → keep the active modality
- both mid → meme bump (hateful-meme case)
"""

from __future__ import annotations

from typing import Dict, List

HATE_GATE = 0.65  # default persona mid; decide.py applies persona thresholds
TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40
MEME_BUMP = 0.15
MEME_LO = 0.35
MEME_HI = 0.85


def fuse(text_score: float, vision_score: float) -> float:
    """Safe late fusion — never silently zeros out a strong single modality."""
    t = max(0.0, min(1.0, float(text_score)))
    v = max(0.0, min(1.0, float(vision_score)))

    if v <= 0.0 and t <= 0.0:
        return 0.0
    if v <= 0.0:
        return round(t, 4)
    if t <= 0.0:
        return round(v, 4)

    fused = TEXT_WEIGHT * t + VISION_WEIGHT * v
    if MEME_LO <= t < MEME_HI and MEME_LO <= v < MEME_HI:
        fused = min(1.0, fused + MEME_BUMP)
    # Strong text should never be pulled below itself by mild vision noise
    if t >= 0.75 and v < 0.35:
        fused = max(fused, t)
    # Strong vision with empty/weak text (vision-only path)
    if v >= 0.75 and t < 0.35:
        fused = max(fused, v)
    return round(min(1.0, max(0.0, fused)), 4)


def fusion_detail(text_score: float, vision_score: float) -> Dict[str, float | bool | str]:
    t = max(0.0, min(1.0, float(text_score)))
    v = max(0.0, min(1.0, float(vision_score)))
    weighted = TEXT_WEIGHT * t + VISION_WEIGHT * v if (t > 0 and v > 0) else max(t, v)
    meme = MEME_LO <= t < MEME_HI and MEME_LO <= v < MEME_HI
    if v <= 0.0 and t > 0.0:
        mode = "text_only"
    elif t <= 0.0 and v > 0.0:
        mode = "vision_only"
    elif meme:
        mode = "multimodal_meme_bump"
    elif t > 0.0 and v > 0.0:
        mode = "multimodal"
    else:
        mode = "idle"
    return {
        "text_weight": TEXT_WEIGHT,
        "vision_weight": VISION_WEIGHT,
        "weighted": round(min(1.0, max(0.0, weighted)), 4),
        "meme_bump": meme,
        "fused": fuse(t, v),
        "mode": mode,
    }


def explain_fusion(
    *,
    text_score: float,
    vision_score: float,
    fused: float,
    escalated: bool,
    decision: str,
    has_ocr: bool,
    has_asr: bool,
    lexicon_hits: List[str],
) -> str:
    parts: List[str] = []
    detail = fusion_detail(text_score, vision_score)
    mode = str(detail["mode"])
    if decision == "hate":
        parts.append("Threat confirmed after multimodal scoring.")
    elif escalated:
        parts.append("Escalated to Stage-2 but below age persona threshold.")
    else:
        parts.append("Stage-1 screen — no escalation.")

    if mode == "text_only":
        parts.append("Signal from text (OCR/ASR/overlay).")
    elif mode == "vision_only":
        parts.append("Signal from vision only (no readable text).")
    elif mode == "multimodal_meme_bump":
        parts.append("Text + image both mid — meme bump applied.")
    elif mode == "multimodal":
        parts.append("Text + image fused.")

    channels = []
    if has_ocr:
        channels.append("OCR")
    if has_asr:
        channels.append("ASR")
    if vision_score > 0:
        channels.append("vision")
    if channels:
        parts.append("Channels: " + ", ".join(channels) + ".")
    if lexicon_hits:
        parts.append("Lexicon: " + ", ".join(lexicon_hits[:5]) + ".")
    parts.append(f"Risk {fused:.2f} (text {text_score:.2f}, vision {vision_score:.2f}).")
    return " ".join(parts)


class TextFull:
    """Stage-2 text confirmer. Until text_full.onnx exists, reuse stage-1 score."""

    def __init__(self) -> None:
        self.name = "passthrough"

    def score(self, text: str, stage1_score: float) -> float:
        return stage1_score
