"""Stage-1 image screen on CLIP embedding. Stub until image_fast probe is trained."""

from __future__ import annotations

from typing import List


class ImageFast:
    def __init__(self) -> None:
        self.name = "deferred"

    def score(self, embedding: List[float]) -> float:
        if not embedding:
            return 0.0
        # Milestone A2: logistic probe on CLIP emb.
        return 0.0
