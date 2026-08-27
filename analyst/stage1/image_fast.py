"""Stage-1 image screen on CLIP embedding (zero-shot until probe is trained).

Eng plan: logistic probe on CLIP emb after Dataset C.
Demo: cosine zero-shot vs harmful vs safe prompt banks.
Missing CLIP → score 0.0, name=deferred (text path still works).

Calibration status — why zero-shot does not move the score
----------------------------------------------------------
Measured on the demo assets, this branch returns 0.324-0.393 for everything:

    01_clean_gaming   0.349      cosine margin +0.0069
    02_hate_threat    0.393                    +0.0268
    03_hate_identity  0.330                    -0.0015   (below the safe bank)
    04_benign_chat    0.324                    -0.0044
    05_vision_only    0.359                    +0.0115

A 0.07 spread with the hate asset ranking below a benign one is noise. Fed into
fusion it cleared a confirmed "you should kys" for ages 14-15 (see
stage2/fusion.py). So the branch reports `calibrated = False` and fusion
excludes it from the score while still showing it to the panel as evidence.

`calibrated` flips to True when a probe trained on labelled data is injected
(`score_fn`), which is the Milestone A2 / Step 6 deliverable. Nothing else in
the cascade needs to change when that lands.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from PIL import Image

from ..extract.embed import ImageEmbedder

HARMFUL_PROMPTS = [
    "a hateful meme attacking a person or group",
    "an image promoting violence or self-harm",
    "racist or discriminatory hate speech imagery",
    "a threatening or abusive internet meme",
]

SAFE_PROMPTS = [
    "a clean video game screenshot",
    "a friendly chat conversation window",
    "a peaceful landscape photograph",
    "a normal school or homework screen",
]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# Trained by analyst/evaluation/train_image_probe.py. Absent by default — the
# weights are gitignored, so a fresh clone runs uncalibrated until someone
# trains one, which is the correct default.
PROBE_PATH = Path(__file__).resolve().parents[1] / "models" / "image_probe.json"


def _load_probe() -> Optional[dict]:
    """Load the trained probe, but only if it earned its place.

    `meets_bar` is written by the training script and reflects held-out AUC.
    A probe that missed the bar is kept on disk for the record and deliberately
    not loaded: letting an under-performing vision score into fusion is exactly
    what cleared confirmed threats for ages 14-15 before.
    """
    try:
        if not PROBE_PATH.is_file():
            return None
        probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
        if not probe.get("meets_bar"):
            return None
        if not probe.get("coef"):
            return None
        return probe
    except Exception:
        return None


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class ImageFast:
    def __init__(
        self,
        score_fn: Optional[Callable[..., float]] = None,
        embedder: Optional[ImageEmbedder] = None,
    ) -> None:
        self._override = score_fn
        self.embedder = embedder
        self._text_bank: Optional[tuple[List[List[float]], List[List[float]]]] = None
        # Only a probe trained on labelled data may move the fused score.
        # Zero-shot prompt banks are diagnostic evidence only — see docstring.
        self.calibrated = score_fn is not None
        self._probe = None if score_fn is not None else _load_probe()
        if self._probe is not None:
            self.calibrated = True
            self.name = f"clip-probe(auc {self._probe['dev_auc']})"
            return
        if self._override is not None:
            self.name = "injected"
        elif self.embedder is not None and (
            self.embedder.name.startswith("clip") or self.embedder.name == "clip:pending"
        ):
            self.name = "clip-zeroshot-uncalibrated"
        else:
            self.name = "deferred"

    def bind_embedder(self, embedder: ImageEmbedder) -> None:
        self.embedder = embedder
        if self._override is not None or self._probe is not None:
            return  # a trained probe outranks whatever the embedder is called
        if embedder.name.startswith("clip") or embedder.name == "clip:pending":
            self.name = "clip-zeroshot-uncalibrated"
        else:
            self.name = "deferred"

    def score(
        self,
        embedding: Optional[List[float]] = None,
        image: Optional[Image.Image] = None,
    ) -> float:
        if self._override is not None:
            try:
                return float(self._override(embedding=embedding, image=image))
            except TypeError:
                try:
                    return float(self._override(embedding or []))
                except Exception:
                    return 0.0
            except Exception:
                return 0.0

        emb = embedding
        if (not emb) and image is not None and self.embedder is not None:
            emb = self.embedder.embed(image)
        if not emb or self.embedder is None:
            return 0.0
        if self._probe is not None:
            return self._probe_score(emb)
        if not (
            self.embedder.name.startswith("clip") or self.embedder.name == "clip:pending"
        ):
            return 0.0

        return self._zero_shot(emb)

    def _probe_score(self, image_emb: List[float]) -> float:
        """Trained logistic probe over the CLIP embedding."""
        probe = self._probe
        if len(image_emb) != probe.get("dim"):
            # Embedder changed under the probe; refuse rather than score noise.
            self.calibrated = False
            self.name = "clip-probe-dim-mismatch"
            self._probe = None
            return 0.0
        z = _dot(image_emb, probe["coef"]) + probe["intercept"]
        return round(_sigmoid(z), 4)

    def _zero_shot(self, image_emb: List[float]) -> float:
        assert self.embedder is not None
        if self._text_bank is None:
            harm = self.embedder.embed_texts(HARMFUL_PROMPTS)
            safe = self.embedder.embed_texts(SAFE_PROMPTS)
            if not harm or not safe:
                self.name = "deferred"
                return 0.0
            self._text_bank = (harm, safe)
            self.name = "clip-zeroshot-uncalibrated"

        harm, safe = self._text_bank
        max_h = max(_dot(image_emb, t) for t in harm)
        max_s = max(_dot(image_emb, t) for t in safe)
        # Map advantage into [0, 1]; temperature softens CLIP cosine range
        raw = (max_h - max_s + 0.15) / 0.45
        return round(min(1.0, max(0.0, raw)), 4)
