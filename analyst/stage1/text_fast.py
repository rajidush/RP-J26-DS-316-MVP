"""Stage-1 text screen: auditable lexicon + a hate-speech model + framing guard.

Layering, and why each layer is here (all numbers from
`python -m analyst.evaluation.benchmark`, persona 8-10, dev set):

    lexicon + patterns   100% recall, 0 missed — owns explicit abuse, threats,
                         bullying, grooming and obfuscation. Instant (~1 ms),
                         and every hit names the rule that fired.

    hate model           the only layer that reads *implicit* identity hate
                         ("they breed like vermin"), which the lexicon scored
                         0% on. Measured alone: 100% precision, zero false
                         positives across gaming / reporting / figurative.

    framing guard        cancels the lexicon's one weakness — a child quoting
                         or reporting abuse ("he keeps calling me a retard,
                         what do i do") scored 0.88 and would have triggered
                         an intervention against the victim.

The two detectors are combined with max() because they are complementary, not
redundant: each covers cases the other scores at zero. The framing guard is
applied *last*, to the combined score, so neither layer can route around it.

Model choice was measured, not assumed. `martin-ha/toxic-comment-model` (the
previous default) scored 58.2% accuracy / 18.2% recall and is a generic
toxicity head; the Dynabench R4 model is trained adversarially for hate and
scored 70.1% alone with no false positives. Override with ANALYST_TEXT_MODEL.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from . import framing as framing_mod
from . import lexicon as lexicon_mod
from .hf_model import HfTextClassifier
from .lexicon import score_text as lexicon_score

# Two heads, because measurement showed they fail on *different* cases and
# neither covers the space alone (held-out set, theta=0.35):
#
#   roberta-hate-dynabench   implicit identity hate 40%, bullying  0%
#   toxic-bert (Jigsaw)      implicit identity hate 20%, bullying 60%
#
# They are combined with max() and run concurrently, so the ensemble costs one
# model's latency rather than two. Override the pair with ANALYST_TEXT_MODEL
# (primary) and ANALYST_TEXT_MODEL_2 (secondary); set ANALYST_TEXT_MODEL_2=""
# to run a single head.
DEFAULT_HF_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
DEFAULT_HF_MODEL_2 = "unitary/toxic-bert"

# The model may claim a category, but a literal slur is more specific evidence,
# so the lexicon's category wins whenever it produced one.
_MODEL_CATEGORY_MIN = 0.55

# At or above the lexicon's HIGH band the model cannot alter the result, so it
# is not run. Mirrors lexicon.HIGH_BASE.
_LEXICON_DECIDES_ALONE = 0.88

# Ceiling on the statistical heads inside gaming context. This must sit below
# the stage-1 gate (decide.STAGE1_THETA = 0.35), not merely below the persona
# thresholds: a cap of 0.50 still escalated every gaming frame into Stage 2,
# burning the cascade's whole cost saving on banter. Guarded by a test.
GAMING_MODEL_CAP = 0.30


@dataclass
class ScoreDetail:
    """Everything the panel and the store need to explain one text score."""

    score: float
    category: str
    hits: List[str] = field(default_factory=list)
    lexicon_score: float = 0.0
    model_score: Optional[float] = None
    model_category: Optional[str] = None
    framing_reason: str = ""
    discounted_from: Optional[float] = None

    @property
    def discounted(self) -> bool:
        return self.discounted_from is not None

    def as_tuple(self) -> Tuple[float, str, List[str]]:
        return self.score, self.category, self.hits


class TextFast:
    def __init__(
        self,
        pretrained_score_fn: Optional[Callable[[str], float]] = None,
        model_id: Optional[str] = None,
        use_framing: bool = True,
    ) -> None:
        self._override = pretrained_score_fn
        self._use_framing = use_framing
        self._models: List[HfTextClassifier] = []

        if self._override is not None:
            self.name = "lexicon+injected"
            return

        primary = HfTextClassifier(model_id or DEFAULT_HF_MODEL, env_var="ANALYST_TEXT_MODEL")
        self._models.append(primary)
        if model_id is None:
            secondary_id = os.environ.get("ANALYST_TEXT_MODEL_2", DEFAULT_HF_MODEL_2).strip()
            if secondary_id:
                self._models.append(HfTextClassifier(secondary_id))
        self.name = self._describe()

    def _describe(self) -> str:
        live = [m.name for m in self._models if m.available]
        return "lexicon+" + "+".join(live) if live else "lexicon"

    @property
    def _model(self) -> Optional[HfTextClassifier]:
        """Primary head — kept for callers that want a single classifier."""
        return self._models[0] if self._models else None

    # -- public API -----------------------------------------------------------

    def score(self, text: str) -> Tuple[float, str, List[str]]:
        """Backwards-compatible 3-tuple. Prefer score_detailed() for evidence."""
        return self.score_detailed(text).as_tuple()

    def score_detailed(self, text: str) -> ScoreDetail:
        lex_score, category, hits = lexicon_score(text)

        # Cascade short-circuit: once the lexicon has fired in its top band the
        # model cannot change the outcome (scores combine with max, and the
        # literal hit already carries the more specific category), so skip the
        # ~180 ms model call entirely. Explicit abuse is therefore the *cheapest*
        # path through Stage 1, not the most expensive.
        if lex_score >= _LEXICON_DECIDES_ALONE:
            model_score, model_category = None, None
        else:
            model_score, model_category = self._model_reading(text)

        # Competitive trash talk is the models' worst blind spot: toxic-bert
        # scored "you're trash at this game lol" 0.93 and "im gonna kill you in
        # this match" 0.88. Cap the statistical contribution inside gaming
        # context and let the lexicon rule, so a real slur in a game lobby
        # still fires while ordinary banter does not.
        gaming = lexicon_mod.in_gaming_context(text)
        if gaming and model_score is not None and model_score > GAMING_MODEL_CAP:
            model_score = GAMING_MODEL_CAP
            model_category = None

        combined = lex_score
        if model_score is not None:
            combined = max(lex_score, model_score)
            # Lexicon hits are the more specific evidence; only take the
            # model's category when the lexicon offered none of its own.
            if not hits and model_score >= _MODEL_CATEGORY_MIN:
                # A binary head says "harmful" without saying what kind. Claiming
                # hate_identity there would assert a protected-class attack we
                # have no evidence for, so fall back to the least specific label.
                category = model_category or "bullying"

        detail = ScoreDetail(
            score=round(combined, 4),
            category=category,
            hits=list(hits),
            lexicon_score=round(lex_score, 4),
            model_score=model_score,
            model_category=model_category,
        )

        if self._use_framing:
            capped, mark = framing_mod.apply(detail.score, text, category=detail.category)
            if capped < detail.score:
                detail.discounted_from = detail.score
                detail.score = round(capped, 4)
                detail.framing_reason = mark.reason
                # A discounted run is no longer a hate claim; keep the hits for
                # the audit trail but stop asserting a category.
                if detail.score < 0.55:
                    detail.category = "none"
            elif mark.reporting:
                detail.framing_reason = mark.reason

        return detail

    def model_labels(self, text: str) -> dict:
        """Raw model labels — panel evidence for 'why did it say that'."""
        if self._model is None:
            return {}
        return self._model.top_labels(text)

    # -- internals ------------------------------------------------------------

    def _model_reading(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Highest hate score across the ensemble, with that head's category.

        max() rather than an average: the heads are complementary, so a case
        one of them understands must not be diluted by the other's ignorance.
        The cost is that either head's false positive survives — which is why
        both were checked for false positives before being combined (roberta 1,
        toxic-bert 0 on the held-out set).
        """
        if not (text or "").strip():
            return None, None
        if self._override is not None:
            try:
                return float(self._override(text)), None
            except Exception:
                return None, None
        if not self._models:
            return None, None

        readings = self._read_all(text)
        if not readings:
            return None, None
        best_score, best_category = max(readings, key=lambda pair: pair[0])
        if self.name == "lexicon":
            self.name = self._describe()  # heads load lazily on first use
        return best_score, best_category

    def _read_all(self, text: str) -> List[Tuple[float, Optional[str]]]:
        if len(self._models) == 1:
            return self._read_one(self._models[0], text)
        # Concurrent: the heads are independent and each is dominated by its
        # own forward pass, so the ensemble costs ~one model's wall time.
        with ThreadPoolExecutor(max_workers=len(self._models)) as pool:
            futures = [pool.submit(self._read_one, m, text) for m in self._models]
            out: List[Tuple[float, Optional[str]]] = []
            for future in futures:
                out.extend(future.result())
        return out

    @staticmethod
    def _read_one(model: HfTextClassifier, text: str) -> List[Tuple[float, Optional[str]]]:
        try:
            score, category = model.score_and_category(text)
        except Exception:
            return []
        return [(float(score), category)] if score is not None else []
