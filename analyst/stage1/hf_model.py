"""Shared HuggingFace text-classification wrapper (CPU only).

One wrapper, used by both the cascade and `evaluation/benchmark.py`, so the
numbers in the report describe the code that actually ships. A benchmark with
its own private scoring path measures a program nobody runs.

Label handling
--------------
Models disagree about label vocabularies:

    facebook/roberta-hate-speech-dynabench-r4-target   nothate | hate   (softmax)
    unitary/toxic-bert    toxic, severe_toxic, obscene, threat, insult,
                          identity_hate                                (sigmoid, multi-label)
    martin-ha/toxic-comment-model                      toxic | non-toxic

So labels are matched by *name*, never by index, and an unrecognised vocabulary
returns None rather than a guess — a fabricated 0.5 from an unknown head is
worse than an honest abstention, because it silently poisons the cascade.

Everything degrades to `available == False` when transformers/torch are absent;
the lexicon keeps the cascade alive on its own.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# Harmful label names across the models we support.
HATE_LABELS = {
    "hate", "hateful", "hate_speech", "hatespeech", "toxic", "severe_toxic",
    "obscene", "threat", "insult", "identity_hate", "identity_attack",
    "offensive", "abusive", "harassment", "sexual_explicit", "label_1", "1",
}
SAFE_LABELS = {
    "nothate", "not_hate", "non_toxic", "nontoxic", "neutral", "benign",
    "normal", "clean", "ok", "label_0", "0",
}

# Multi-label heads carry the category directly — far better evidence than
# guessing a category from a binary score. Maps to analyst.schemas.Category.
LABEL_TO_CATEGORY = {
    "identity_hate": "hate_identity",
    "identity_attack": "hate_identity",
    "hate": "hate_identity",
    "hate_speech": "hate_identity",
    "threat": "threat",
    "insult": "bullying",
    "harassment": "bullying",
    "sexual_explicit": "sexual_harassment",
    "obscene": "profanity",
    "severe_toxic": "bullying",
    "toxic": "bullying",
}

# A category claim needs more confidence than a hate/not-hate call: below this
# the label is noise and we leave the lexicon's category in place.
CATEGORY_MIN_CONFIDENCE = 0.50


def normalize_label(label) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("-", "_")


def hate_score_from_labels(rows: List[dict]) -> Optional[float]:
    """Collapse a label distribution into one hate probability.

    Multi-label heads emit independent sigmoids, so the max over harmful labels
    is the right reduction. Single-label heads emit a softmax, where the
    safe-label complement is equivalent. None when nothing is recognised.
    """
    hate = [float(r["score"]) for r in rows if normalize_label(r.get("label")) in HATE_LABELS]
    if hate:
        return round(max(hate), 4)
    safe = [float(r["score"]) for r in rows if normalize_label(r.get("label")) in SAFE_LABELS]
    if safe:
        return round(1.0 - max(safe), 4)
    return None


def category_from_labels(rows: List[dict]) -> Optional[str]:
    """Strongest harmful label above the confidence floor, as our category."""
    best_name, best_score = None, 0.0
    for row in rows:
        name = normalize_label(row.get("label"))
        if name not in LABEL_TO_CATEGORY:
            continue
        score = float(row.get("score") or 0.0)
        if score > best_score:
            best_name, best_score = name, score
    if best_name is None or best_score < CATEGORY_MIN_CONFIDENCE:
        return None
    # A bare "toxic"/"severe_toxic" win means the head found harm but no
    # specific kind; prefer a more specific sibling if one is close behind.
    if best_name in ("toxic", "severe_toxic"):
        for row in rows:
            name = normalize_label(row.get("label"))
            if name in LABEL_TO_CATEGORY and name not in ("toxic", "severe_toxic"):
                if float(row.get("score") or 0.0) >= CATEGORY_MIN_CONFIDENCE:
                    return LABEL_TO_CATEGORY[name]
    return LABEL_TO_CATEGORY[best_name]


class HfTextClassifier:
    """Lazy CPU text-classification head. Never raises into the cascade."""

    def __init__(
        self,
        model_id: str,
        env_var: Optional[str] = None,
        max_length: int = 128,
        lazy: bool = True,
    ) -> None:
        self.model_id = (os.environ.get(env_var, "").strip() or model_id) if env_var else model_id
        self.max_length = max_length
        self._pipe = None
        self._load_attempted = False
        self.last_error = ""
        self._supported = self._transformers_present()
        self.name = f"hf:{self.model_id.split('/')[-1]}" if self._supported else "unavailable"
        if not lazy:
            self._ensure_loaded()

    @staticmethod
    def _transformers_present() -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._supported and (self._pipe is not None or not self._load_attempted)

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        if self._load_attempted or not self._supported:
            return False
        self._load_attempted = True
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                device=-1,  # CPU only — the whole project is a CPU claim
                truncation=True,
                max_length=self.max_length,
                top_k=None,  # full distribution, so multi-label heads work
            )
            self.name = f"hf:{self.model_id.split('/')[-1]}"
            return True
        except Exception as exc:
            self._pipe = None
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            self.name = "unavailable"
            return False

    def label_scores(self, text: str) -> List[dict]:
        if not (text or "").strip() or not self._ensure_loaded():
            return []
        try:
            out = self._pipe(text[:512])
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return []
        if not out:
            return []
        return out[0] if isinstance(out[0], list) else out

    def score(self, text: str) -> Optional[float]:
        """Hate probability in [0,1], or None when unavailable/unrecognised."""
        rows = self.label_scores(text)
        return hate_score_from_labels(rows) if rows else None

    def score_and_category(self, text: str) -> tuple[Optional[float], Optional[str]]:
        rows = self.label_scores(text)
        if not rows:
            return None, None
        return hate_score_from_labels(rows), category_from_labels(rows)

    def read(self, text: str) -> tuple[Optional[float], Optional[str], Dict[str, float]]:
        """Score, category and the raw labels behind them, from one forward pass.

        The panel shows the labels as evidence ("why did it say that"), so they
        must come from the same inference that produced the score — asking the
        model twice could show a reader labels that never drove the decision.
        """
        rows = self.label_scores(text)
        if not rows:
            return None, None, {}
        ranked = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)
        labels = {
            normalize_label(r.get("label")): round(float(r.get("score") or 0.0), 4)
            for r in ranked[:3]
        }
        return hate_score_from_labels(rows), category_from_labels(rows), labels

    def top_labels(self, text: str, limit: int = 3) -> Dict[str, float]:
        """Evidence for the panel — what the model actually said."""
        rows = self.label_scores(text)
        ranked = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)
        return {
            normalize_label(r.get("label")): round(float(r.get("score") or 0.0), 4)
            for r in ranked[:limit]
        }
