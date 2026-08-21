"""Stage-1 text screen: lexicon always + optional pretrained HF model.

Demo strategy: use a public DistilBERT toxicity model now; fine-tune later (Step 6).
If transformers/torch/model are missing, lexicon alone keeps the cascade alive.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from .lexicon import score_text as lexicon_score

# Small CPU-friendly toxic classifier. Swap via ANALYST_TEXT_MODEL.
DEFAULT_HF_MODEL = "martin-ha/toxic-comment-model"


class TextFast:
    def __init__(
        self,
        pretrained_score_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self._override = pretrained_score_fn
        self._pipe = None
        self._model_id = os.environ.get("ANALYST_TEXT_MODEL", DEFAULT_HF_MODEL).strip()
        self.name = "lexicon"
        if self._override is not None:
            self.name = "lexicon+injected"
            return
        self._try_load_pretrained()

    def _try_load_pretrained(self) -> None:
        try:
            from transformers import pipeline
        except Exception:
            return

        try:
            self._pipe = pipeline(
                "text-classification",
                model=self._model_id,
                tokenizer=self._model_id,
                device=-1,  # CPU only
                truncation=True,
                max_length=128,
            )
            self.name = f"lexicon+hf:{self._model_id.split('/')[-1]}"
        except Exception:
            self._pipe = None
            self.name = "lexicon"

    def score(self, text: str) -> Tuple[float, str, List[str]]:
        lex_score, category, hits = lexicon_score(text)
        model_score = self._pretrained_score(text)

        if model_score is None:
            return lex_score, category, hits

        combined = max(lex_score, model_score)
        # Prefer lexicon category when it fired (more specific); else map toxic → bullying
        if category == "none" and model_score >= 0.35:
            category = "bullying"
        return combined, category, hits

    def _pretrained_score(self, text: str) -> Optional[float]:
        if not (text or "").strip():
            return None
        if self._override is not None:
            try:
                return float(self._override(text))
            except Exception:
                return None
        if self._pipe is None:
            return None
        try:
            out = self._pipe(text[:512])[0]
            label = str(out.get("label", "")).lower().replace(" ", "_")
            conf = float(out.get("score", 0.0))

            # martin-ha/toxic-comment-model → "toxic" | "non-toxic"
            if label in ("toxic", "hate", "offensive", "abusive", "label_1", "1"):
                return conf
            if label in ("non-toxic", "non_toxic", "neutral", "benign", "label_0", "0"):
                return round(1.0 - conf, 4)

            # Unknown scheme: do not invent a score; lexicon remains source of truth
            return None
        except Exception:
            return None
