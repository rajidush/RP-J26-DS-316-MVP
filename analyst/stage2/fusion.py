"""Late fusion head. Full MLP ONNX later; rule-based fusion works now."""

from __future__ import annotations

HATE_GATE = 0.65  # default persona mid; decide.py applies persona thresholds
TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40


def fuse(text_score: float, vision_score: float) -> float:
    fused = TEXT_WEIGHT * text_score + VISION_WEIGHT * vision_score
    # Meme bump: both modalities mid → combination may be hate
    if 0.35 <= text_score < 0.85 and 0.35 <= vision_score < 0.85:
        fused = min(1.0, fused + 0.15)
    return round(min(1.0, max(0.0, fused)), 4)


class TextFull:
    """Stage-2 text confirmer. Until text_full.onnx exists, reuse stage-1 score."""

    def __init__(self) -> None:
        self.name = "passthrough"

    def score(self, text: str, stage1_score: float) -> float:
        return stage1_score
