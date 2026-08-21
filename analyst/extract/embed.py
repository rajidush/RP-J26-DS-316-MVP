"""Image embedding plug (MobileCLIP / CLIP ONNX). Deferred until model is registered."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image


class ImageEmbedder:
    """Returns empty embedding until models/analyst.clip_img.onnx is present."""

    def __init__(self) -> None:
        self.name = "deferred"
        self.dim = 0

    def embed(self, image: Optional[Image.Image]) -> List[float]:
        if image is None:
            return []
        # Phase A1/A3: load ONNX CLIP tower here. Until then, image branch
        # relies on image_fast stub (score 0) so text path still works.
        return []
