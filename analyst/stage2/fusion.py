"""Late fusion head. Full MLP ONNX later; rule-based fusion works now."""

from __future__ import annotations

from typing import Dict

HATE_GATE = 0.65  # default persona mid; decide.py applies persona thresholds
TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40
MEME_BUMP = 0.15
MEME_LO = 0.35
MEME_HI = 0.85


def fuse(text_score: float, vision_score: float) -> float:
    fused = TEXT_WEIGHT * text_score + VISION_WEIGHT * vision_score
    # Meme bump: both modalities mid → combination may be hate
    if MEME_LO <= text_score < MEME_HI and MEME_LO <= vision_score < MEME_HI:
        fused = min(1.0, fused + MEME_BUMP)
    return round(min(1.0, max(0.0, fused)), 4)


def fusion_detail(text_score: float, vision_score: float) -> Dict[str, float | bool]:
    """Explain weights for the demo table (not used as a second score)."""
    weighted = TEXT_WEIGHT * text_score + VISION_WEIGHT * vision_score
    meme = MEME_LO <= text_score < MEME_HI and MEME_LO <= vision_score < MEME_HI
    return {
        "text_weight": TEXT_WEIGHT,
        "vision_weight": VISION_WEIGHT,
        "weighted": round(min(1.0, max(0.0, weighted)), 4),
        "meme_bump": meme,
        "fused": fuse(text_score, vision_score),
    }


class TextFull:
    """Stage-2 text confirmer. Until text_full.onnx exists, reuse stage-1 score."""

    def __init__(self) -> None:
        self.name = "passthrough"

    def score(self, text: str, stage1_score: float) -> float:
        return stage1_score
