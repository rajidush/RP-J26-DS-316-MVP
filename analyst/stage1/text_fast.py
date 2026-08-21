"""Stage-1 text screen: lexicon always + optional ONNX text_fast later."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from .lexicon import score_text as lexicon_score

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ONNX_NAME = "text_fast.onnx"


class TextFast:
    def __init__(self) -> None:
        self.name = "lexicon"
        self._session = None
        path = Path(os.environ.get("ANALYST_TEXT_FAST_ONNX", MODELS_DIR / ONNX_NAME))
        if path.is_file():
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                self.name = "lexicon+onnx"
            except Exception:
                self._session = None

    def score(self, text: str) -> Tuple[float, str, List[str]]:
        score, category, hits = lexicon_score(text)
        # ONNX graph wiring comes with Milestone A2 export; lexicon never blocks.
        return score, category, hits
