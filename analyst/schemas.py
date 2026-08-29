"""Message contracts aligned with J26-DS-316 Engineering Plan v0.1 §2.4 / §4.7."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

Category = Literal[
    "bullying",
    "hate_identity",
    "threat",
    "sexual_harassment",
    "profanity",
    "none",
]
RecommendedAction = Literal["blur_region", "block", "notify_only", "none"]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid4().hex


class Envelope(BaseModel):
    v: int = 1
    topic: str
    ts: str = Field(default_factory=_now_iso)
    src: str = "analyst"
    id: str = Field(default_factory=new_id)
    corr: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class FrameCaptured(BaseModel):
    """C1 → C2 bus contract (Engineering Plan §2.4).

    NOT YET WIRED. C1 does not exist, so nothing in this repo constructs or
    consumes it — `capture/worker.py` produces JPEG bytes directly and
    `--replay` reads from disk. It is kept because it is the *interface spec*
    C1 will be built against; delete it only if that integration is dropped.
    """

    monitor: int = 0
    w: int = 0
    h: int = 0
    jpeg_bytes: Optional[bytes] = None  # in-memory only; never written to disk
    change_score: float = 1.0
    text_likely: bool = True
    fg_rect: Optional[List[int]] = None
    fg_exe: str = "unknown"
    reason: Literal["scene_change", "heartbeat", "replay", "manual"] = "manual"


class Modalities(BaseModel):
    text: float = 0.0
    image: float = 0.0
    audio: float = 0.0


class Evidence(BaseModel):
    ocr_snippet: str = ""
    transcript_snippet: str = ""
    lexicon_hits: List[str] = Field(default_factory=list)
    top_tokens: List[str] = Field(default_factory=list)


class HateDetectedPayload(BaseModel):
    score: float
    category: Category
    stage: int = Field(..., description="1 = stage-1 only, 2 = fusion/confirm")
    modalities: Modalities
    app: Dict[str, str] = Field(default_factory=lambda: {"exe": "unknown", "category": "other"})
    window_title_hash: str = ""
    evidence: Evidence
    child_safe_summary: str
    recommended_action: RecommendedAction = "notify_only"
    persona_threshold: float = 0.65


class HateClearedPayload(BaseModel):
    stage1_score: float
    stage2_score: float
    reason: str = "below_threshold"


class AnalystRunResult(BaseModel):
    """Internal result returned by the pipeline (also used by CLI / tests)."""

    decision: Literal["hate", "not-hate"]
    envelope: Optional[Envelope] = None
    payload: Optional[HateDetectedPayload] = None
    cleared: Optional[HateClearedPayload] = None
    ocr_text: str = ""
    transcript: str = ""
    # What the vision branch saw. Kept separate from ocr_text so the panel
    # can answer "what does the picture mean" and "what words are in it".
    image_caption: str = ""
    image_text: str = ""
    image_region: Optional[List[int]] = None
    # Where on the frame each detection came from, normalised to 0..1 so the
    # panel can draw an overlay on the 320px blurred thumbnail. Presentation
    # only — the decision does not depend on these.
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    stage1: Dict[str, float] = Field(default_factory=dict)
    stage2: Optional[Dict[str, float]] = None
    backends: Dict[str, str] = Field(default_factory=dict)
    media_deleted: bool = True
    notes: List[str] = Field(default_factory=list)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    modalities: Dict[str, float] = Field(default_factory=dict)
    risk_score: float = 0.0
    explanation: str = ""
    protection_state: Literal["protected", "reviewing", "threat", "degraded"] = "protected"
    lexicon_hits: List[str] = Field(default_factory=list)

    # --- Why the score is what it is (panel + audit trail) -------------------
    # Split the two Stage-1 detectors so a reviewer can see which one fired.
    lexicon_score: float = 0.0
    model_score: Optional[float] = None
    model_labels: Dict[str, float] = Field(default_factory=dict)
    # Set when the framing guard reduced the score because the text reads as
    # reporting/quoting harm rather than committing it. Never silent.
    framing_reason: str = ""
    score_before_framing: Optional[float] = None
    # False while the image branch is uncalibrated: it is shown but excluded
    # from the fused score (see stage2/fusion.py).
    vision_calibrated: bool = False
    fusion_mode: str = "idle"
    escalated: bool = False


CHILD_SAFE_SUMMARIES: Dict[str, str] = {
    "threat": "Someone in this chat is using language that sounds like a threat or harm.",
    "hate_identity": "Someone in this chat is using hurtful language about a group of people.",
    "bullying": "Someone in this chat is using words that can bully or put someone down.",
    "sexual_harassment": "This content has language that is not appropriate for you.",
    "profanity": "This chat has strong language. Your parent can decide what to do.",
    "none": "We checked this content and it looks okay for now.",
}
