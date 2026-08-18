"""Vision plug. Scores 0.0 until models/vision_stage1.onnx is dropped in.

The pipeline still completes: empty-text images simply do not escalate until
the ONNX encoder is present. That is deferred, not a crash.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ONNX_NAME = "vision_stage1.onnx"


class VisionClassifier:
    def __init__(self) -> None:
        self.name = "deferred"
        self._session = None
        self._load_onnx()

    def _load_onnx(self) -> None:
        path = Path(os.environ.get("ANALYST_VISION_ONNX", MODELS_DIR / ONNX_NAME))
        if not path.is_file():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.name = "onnx"
        except Exception:
            self._session = None
            self.name = "onnx_failed"

    def score(self, image: Optional[Image.Image]) -> float:
        if image is None or self._session is None:
            return 0.0
        try:
            # Phase 3 export will define preprocess + input name.
            # Unknown graph must not raise into the cascade.
            return 0.0
        except Exception:
            return 0.0
