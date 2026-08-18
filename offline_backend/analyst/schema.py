from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["hate", "not-hate"]
Category = Literal[
    "hate_speech",
    "cyberbullying",
    "identity_attack",
    "threat",
    "none",
]


class BackendStatus(BaseModel):
    ocr: str = "none"
    asr: str = "none"
    text: str = "lexicon"
    vision: str = "deferred"
    capture: str = "pillow_grab"


class AnalystHealth(BaseModel):
    component: str = "analyst"
    ready: bool = True
    backends: BackendStatus
    notes: list[str] = Field(default_factory=list)


class AnalystResult(BaseModel):
    component: str = "analyst"
    threat_type: str = "hate_speech"
    decision: Decision
    risk_score: float = Field(..., ge=0.0, le=1.0)
    category: Category
    threshold: float = 0.85
    child_age: int
    source: Dict[str, bool]
    session_hint: str
    ocr_text: str = ""
    transcript: str = ""
    overlay_text: str = ""
    stage1: Dict[str, float]
    stage2: Optional[Dict[str, float]] = None
    escalated: bool
    media_deleted: bool = True
    backends: BackendStatus
    notes: list[str] = Field(default_factory=list)
