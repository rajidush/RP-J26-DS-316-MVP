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


# Windowing for label_scores(). A 128-token head cannot see a whole desktop, and
# OCR chrome (tab titles, a long file:// URL) reliably fills the first hundreds
# of characters before any page content. 300 chars is deliberately below the
# ~512 that 128 tokens buys on clean prose, because OCR noise and URLs tokenise
# far worse than prose. The cap bounds worst-case latency on a very busy screen.
_WINDOW_CHARS = int(os.environ.get("ANALYST_TEXT_WINDOW", "300"))
_WINDOW_OVERLAP = int(os.environ.get("ANALYST_TEXT_WINDOW_OVERLAP", "60"))
_MAX_WINDOWS = int(os.environ.get("ANALYST_TEXT_MAX_WINDOWS", "16"))


class HfTextClassifier:
    """Lazy CPU text-classification head. Never raises into the cascade.

    Why the input is windowed, not truncated (measured 1 Sep 2026)
    -------------------------------------------------------------
    `label_scores` used to score `text[:512]`. On a screen blob that is the
    wrong 512 characters: OCR emits browser chrome first — tab titles, then a
    `file://` URL that can be 150 chars on its own — and the content the child
    is actually looking at lands after it. Holding the text identical and only
    moving the slur later in the blob:

        chrome before slur    slur at char    score    verdict (theta 0.55)
                       0               36    0.9959    flags
                     370              406    0.5531    flags, barely
                     740              776    0.0800    MISSES
                    1110             1146    0.0800    MISSES

    0.0800 is the lexicon floor: past the cut the model contributes nothing at
    all. This was not hypothetical — a research PDF of captioned hateful memes
    scored 0.08 with `Language model 0.00 / nothate 1.00` while OCR had in fact
    read "fuckthis somali piece ofshit" into the blob at char 317.

    Note the middle row: even *inside* the window, surrounding chrome dilutes
    the score (0.9985 for the slur alone vs 0.6173 for the same slur embedded
    in a desktop blob). Windowing fixes both — each window is mostly one kind
    of content, and the most harmful one decides.

    Stage 2 already re-read what Stage 1 truncated, but it only runs on
    escalation, so a Stage-1 miss meant it never got the chance.
    """

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

    def _windows(self, text: str) -> List[str]:
        """Overlapping character windows covering the whole blob.

        A screen blob is not a document: the harmful line can sit anywhere in
        it, usually *after* the browser chrome. Scoring only the head is what
        the old `text[:512]` did, and it silently lost real hate — see the
        measurement in the class docstring. The overlap exists so a phrase
        straddling a boundary is still whole in one window.
        """
        text = (text or "").strip()
        if len(text) <= _WINDOW_CHARS:
            return [text]
        step = _WINDOW_CHARS - _WINDOW_OVERLAP
        out: List[str] = []
        for start in range(0, len(text), step):
            chunk = text[start:start + _WINDOW_CHARS]
            if chunk.strip():
                out.append(chunk)
            if len(out) >= _MAX_WINDOWS:
                break
        return out

    def label_scores(self, text: str) -> List[dict]:
        """Labels for the most harmful window, not for the first 512 chars.

        Windows are scored in one batched call, so the extra coverage costs a
        larger batch rather than N round trips.
        """
        if not (text or "").strip() or not self._ensure_loaded():
            return []
        windows = self._windows(text)
        try:
            out = self._pipe(windows)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return []
        if not out:
            return []
        # A list input yields one entry per window; each entry is the full
        # label distribution because the pipeline is built with top_k=None.
        per_window: List[List[dict]] = []
        for entry in out:
            if isinstance(entry, list):
                per_window.append(entry)
            elif isinstance(entry, dict):
                per_window.append([entry])
        if not per_window:
            return []
        if len(per_window) == 1:
            return per_window[0]
        return max(per_window, key=lambda rows: hate_score_from_labels(rows) or 0.0)

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
