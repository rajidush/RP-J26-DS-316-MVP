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

The lexicon and the model layer are combined with max() because they are
complementary, not redundant: each covers cases the other scores at zero. The
two *model heads* are combined by corroboration instead — see the constants
below, where plain max() was measured to dominate the false-positive budget.
The framing guard is applied *last*, to the combined score, so no layer can
route around it.

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
# They run concurrently, so the ensemble costs one model's latency rather than
# two, and are combined by corroboration (see below). Override the pair with
# ANALYST_TEXT_MODEL (primary) and ANALYST_TEXT_MODEL_2 (secondary); set
# ANALYST_TEXT_MODEL_2="" to run a single head.
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

# --- Ensemble corroboration -------------------------------------------------
# Combining the heads with plain max() means *either* head's false positive
# survives. On published corpora that dominated the error budget: 156 of 165
# sampled Berkeley false positives and 65 of 67 Davidson ones came from a model
# rather than the lexicon.
#
# Measured on train splits (recall / false-positive rate):
#
#                    berkeley        davidson       heldout recall
#   max()            94% / 43%       96% / 18%           40%
#   corroboration    91% / 36%       95% /  9%           30%
#   mean             81% / 22%       89% /  6%           10%
#
# Corroboration halves Davidson's false positives for one point of recall.
# That trade is right for this product specifically: the cascade re-checks the
# screen every 2.5 s, so a missed detection gets another chance on the next
# tick, while a false alert interrupts the child immediately and teaches them
# the tool is noise. Precision has no second chance; recall does.
SOLO_TRUST = 0.90          # one head this confident is trusted alone
CORROBORATION_FLOOR = 0.50  # ...otherwise the other head must also agree
SOLO_DAMP = 0.70            # uncorroborated and merely confident: damp it


@dataclass
class ScoreDetail:
    """Everything the panel and the store need to explain one text score."""

    score: float
    category: str
    hits: List[str] = field(default_factory=list)
    lexicon_score: float = 0.0
    model_score: Optional[float] = None
    model_category: Optional[str] = None
    model_labels: dict = field(default_factory=dict)
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
            model_score, model_category, model_labels = None, None, {}
        else:
            model_score, model_category, model_labels = self._model_reading(text)

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
            model_labels=model_labels,
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

    # A model_labels() method used to live here and ran the model a second
    # time. Labels now ride along on ScoreDetail from the same forward pass
    # that produced the score, so the panel cannot be shown labels that did
    # not actually drive the decision.

    # -- internals ------------------------------------------------------------

    def _model_reading(self, text: str) -> Tuple[Optional[float], Optional[str], dict]:
        """Ensemble reading: highest head score, damped when uncorroborated.

        An average would dilute a case only one head understands, so the base
        is still the maximum. But a lone head that is merely confident is not
        trustworthy out-of-domain, so it is damped unless the other head also
        sees harm — see SOLO_TRUST / CORROBORATION_FLOOR / SOLO_DAMP.
        """
        if not (text or "").strip():
            return None, None, {}
        if self._override is not None:
            try:
                return float(self._override(text)), None, {}
            except Exception:
                return None, None, {}
        if not self._models:
            return None, None, {}

        readings = self._read_all(text)
        if not readings:
            return None, None, {}
        if self.name == "lexicon":
            self.name = self._describe()  # heads load lazily on first use

        best_score, best_category, best_labels = max(readings, key=lambda r: r[0])
        if len(readings) < 2:
            return best_score, best_category, best_labels

        lowest = min(r[0] for r in readings)
        if best_score >= SOLO_TRUST or lowest >= CORROBORATION_FLOOR:
            return best_score, best_category, best_labels
        # One head alone, only moderately confident: damp rather than trust.
        return round(best_score * SOLO_DAMP, 4), best_category, best_labels

    def _read_all(self, text: str) -> List[Tuple[float, Optional[str], dict]]:
        if len(self._models) == 1:
            return self._read_one(self._models[0], text)
        # Concurrent: the heads are independent and each is dominated by its
        # own forward pass, so the ensemble costs ~one model's wall time.
        with ThreadPoolExecutor(max_workers=len(self._models)) as pool:
            futures = [pool.submit(self._read_one, m, text) for m in self._models]
            out: List[Tuple[float, Optional[str], dict]] = []
            for future in futures:
                out.extend(future.result())
        return out

    @staticmethod
    def _read_one(model: HfTextClassifier, text: str) -> List[Tuple[float, Optional[str], dict]]:
        try:
            score, category, labels = model.read(text)
        except Exception:
            return []
        return [(float(score), category, labels)] if score is not None else []
