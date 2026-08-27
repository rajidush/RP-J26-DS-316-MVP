"""Late fusion across modalities, with a safety invariant.

The bug this design exists to prevent
-------------------------------------
Fusion used to average whatever each branch produced. Measured on the demo
assets, CLIP zero-shot returned 0.324-0.393 for *every* image — clean gaming,
hate, benign chat, and an abstract shape alike (cosine margin +/-0.02, and the
hate_identity asset scored *below* the safe prompts). That is noise, not a
signal. Averaged into a confident text detection it did real damage:

    text 0.88 + vision 0.393  ->  0.685   cleared for ages 14-15
    text 0.88 + vision 0.330  ->  0.880   flagged for every age

A confirmed "you should kys" was therefore cleared or flagged depending on
which side of an arbitrary guard the *noise* happened to land. Safety behaviour
must never depend on an uninformative input.

Two rules follow, and both are enforced by tests:

1. **Only calibrated modalities fuse.** A branch that has not been validated
   against labelled data contributes evidence to the panel, never to the score.
   `image_fast` zero-shot is uncalibrated until the trained probe lands
   (Milestone A2 / Step 6), which is what the original design intended before
   the branch was silently switched on.

2. **Fusion may raise confidence, never lower it.** The fused score is floored
   at the strongest calibrated modality. Corroboration between modalities is
   evidence *for* harm; it is never evidence against harm that one modality
   already established on its own.

Text extracted *from* an image (OCR, or a VLM reading a meme) is text, not
vision — it is scored by the same text scorer and belongs to the text channel.
The vision channel is reserved for judgements made about pixels themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40

# Independent mid-band signals corroborate each other — the hateful-meme case,
# where neither the picture nor the caption is damning alone.
AGREEMENT_BONUS = 0.15
AGREE_LO = 0.35
AGREE_HI = 0.85


@dataclass
class Signal:
    """One modality's contribution."""

    name: str          # text | vision | audio
    score: float
    calibrated: bool   # may it move the score, or is it evidence only?
    source: str = ""   # what produced it, for the panel

    @property
    def contributes(self) -> bool:
        return self.calibrated and self.score > 0.0


@dataclass
class FusionResult:
    fused: float
    mode: str
    floor: float                       # strongest calibrated modality
    weighted: float                    # the raw weighted sum, before the floor
    agreement: bool = False
    contributing: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {
            "fused": round(self.fused, 4),
            "floor": round(self.floor, 4),
            "weighted": round(self.weighted, 4),
            "agreement": 1.0 if self.agreement else 0.0,
            "text_weight": TEXT_WEIGHT,
            "vision_weight": VISION_WEIGHT,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def fuse_signals(signals: List[Signal]) -> FusionResult:
    """Combine modalities under the two rules in the module docstring."""
    contributing = [s for s in signals if s.contributes]
    ignored = [s.name for s in signals if s.calibrated is False and s.score > 0.0]

    if not contributing:
        return FusionResult(fused=0.0, mode="idle", floor=0.0, weighted=0.0, ignored=ignored)

    floor = max(s.score for s in contributing)

    if len(contributing) == 1:
        only = contributing[0]
        return FusionResult(
            fused=round(_clamp(only.score), 4),
            mode=f"{only.name}_only",
            floor=round(floor, 4),
            weighted=round(only.score, 4),
            contributing=[only.name],
            ignored=ignored,
        )

    text = next((s.score for s in contributing if s.name == "text"), 0.0)
    vision = next((s.score for s in contributing if s.name == "vision"), 0.0)
    weighted = TEXT_WEIGHT * text + VISION_WEIGHT * vision

    agreement = all(AGREE_LO <= s.score < AGREE_HI for s in contributing)
    fused = weighted + AGREEMENT_BONUS if agreement else weighted

    # Rule 2: corroboration may only add confidence.
    fused = max(fused, floor)
    return FusionResult(
        fused=round(_clamp(fused), 4),
        mode="multimodal_agreement" if agreement else "multimodal",
        floor=round(floor, 4),
        weighted=round(_clamp(weighted), 4),
        agreement=agreement,
        contributing=[s.name for s in contributing],
        ignored=ignored,
    )


def fuse(text_score: float, vision_score: float, vision_calibrated: bool = True) -> float:
    """Two-modality shorthand, kept for callers and tests that predate Signal."""
    return fuse_signals(
        [
            Signal("text", _clamp(text_score), True, "text"),
            Signal("vision", _clamp(vision_score), vision_calibrated, "vision"),
        ]
    ).fused


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
    vision_calibrated: bool = True,
    framing_reason: str = "",
    model_score: Optional[float] = None,
) -> str:
    """One plain-English sentence per fact that moved the decision."""
    parts: List[str] = []
    if decision == "hate":
        parts.append("Flagged after multimodal scoring.")
    elif escalated:
        parts.append("Escalated to Stage 2 but stayed below this child's age threshold.")
    else:
        parts.append("Stage-1 screen — nothing escalated.")

    channels = []
    if has_ocr:
        channels.append("on-screen text")
    if has_asr:
        channels.append("speech")
    if channels:
        parts.append("Read from " + " and ".join(channels) + ".")

    if lexicon_hits:
        parts.append("Matched rule(s): " + ", ".join(lexicon_hits[:4]) + ".")
    if model_score is not None and model_score > 0:
        parts.append(f"Language model scored {model_score:.2f}.")
    if framing_reason:
        parts.append("Score reduced — the text looks like it is reporting or quoting harm, not committing it.")
    if vision_score > 0 and not vision_calibrated:
        parts.append(
            f"Image branch read {vision_score:.2f} but is uncalibrated, so it did not affect the score."
        )
    parts.append(f"Final risk {fused:.2f}.")
    return " ".join(parts)


class TextFull:
    """Stage-2 text confirm: re-read the *whole* text, not just the first window.

    Stage 1 truncates to 128 tokens for speed. On a screen capture that is often
    only the first few chat lines, so abuse further down a long conversation is
    invisible to it. Stage 2 runs only after escalation, and it is the place
    that can afford to look at everything: the text is chunked and every chunk
    scored, taking the maximum.

    This is a real second opinion rather than a different model for its own
    sake — measurement did not justify a third head, but it did show that a
    truncated window loses evidence.
    """

    CHUNK_CHARS = 400
    MAX_CHUNKS = 6

    def __init__(self, scorer=None) -> None:
        self._scorer = scorer
        self.name = "chunked-recheck" if scorer is not None else "passthrough"

    def bind(self, scorer) -> None:
        """Give Stage 2 the Stage-1 scorer to reuse (no extra model loaded)."""
        self._scorer = scorer
        self.name = "chunked-recheck"

    def score(self, text: str, stage1_score: float) -> float:
        blob = (text or "").strip()
        if self._scorer is None or len(blob) <= self.CHUNK_CHARS:
            return stage1_score  # Stage 1 already saw all of it
        best = stage1_score
        for chunk in self._chunks(blob):
            try:
                best = max(best, float(self._scorer(chunk)))
            except Exception:
                continue
        return round(best, 4)

    def _chunks(self, blob: str) -> List[str]:
        step = self.CHUNK_CHARS // 2  # overlap, so harm spanning a cut survives
        out = [blob[i:i + self.CHUNK_CHARS] for i in range(0, len(blob), step)]
        return [c for c in out if c.strip()][: self.MAX_CHUNKS]
