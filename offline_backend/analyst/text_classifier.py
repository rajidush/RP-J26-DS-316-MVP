"""Text hate scorer: lexicon always, optional ONNX/transformers later via models/text_hate.onnx."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from .lexicon import score_text as lexicon_score

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ONNX_NAME = "text_hate.onnx"


class TextClassifier:
    def __init__(self) -> None:
        self.name = "lexicon"
        self._session = None
        self._load_onnx()

    def _load_onnx(self) -> None:
        path = Path(os.environ.get("ANALYST_TEXT_ONNX", MODELS_DIR / ONNX_NAME))
        if not path.is_file():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.name = "lexicon+onnx"
        except Exception:
            self._session = None

    def score(self, text: str) -> Tuple[float, str, str]:
        lex_score, category = lexicon_score(text)
        model_score = self._onnx_score(text)
        if model_score is None:
            return lex_score, category, self.name
        # High-recall Stage 1: either source may escalate.
        combined = max(lex_score, model_score)
        return combined, category, self.name

    def _onnx_score(self, text: str) -> Optional[float]:
        if self._session is None or not (text or "").strip():
            return None
        # Plug-in contract: input "text" is not standardized across exports.
        # Until Phase 4 exports a known graph, skip rather than crash.
        try:
            inputs = self._session.get_inputs()
            if not inputs:
                return None
            # Unknown graph → do not guess tensors; lexicon remains source of truth.
            return None
        except Exception:
            return None
