"""Stage-1 image screen on CLIP embedding (zero-shot until probe is trained).

Eng plan: logistic probe on CLIP emb after Dataset C.
Demo: cosine zero-shot vs harmful vs safe prompt banks.
Missing CLIP → score 0.0, name=deferred (text path still works).
"""

from __future__ import annotations

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


class ImageFast:
    def __init__(
        self,
        score_fn: Optional[Callable[..., float]] = None,
        embedder: Optional[ImageEmbedder] = None,
    ) -> None:
        self._override = score_fn
        self.embedder = embedder
        self._text_bank: Optional[tuple[List[List[float]], List[List[float]]]] = None
        if self._override is not None:
            self.name = "injected"
        elif self.embedder is not None and (
            self.embedder.name.startswith("clip") or self.embedder.name == "clip:pending"
        ):
            self.name = "clip-zeroshot"
        else:
            self.name = "deferred"

    def bind_embedder(self, embedder: ImageEmbedder) -> None:
        self.embedder = embedder
        if self._override is not None:
            return
        if embedder.name.startswith("clip") or embedder.name == "clip:pending":
            self.name = "clip-zeroshot"
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
        if not (
            self.embedder.name.startswith("clip") or self.embedder.name == "clip:pending"
        ):
            return 0.0

        return self._zero_shot(emb)

    def _zero_shot(self, image_emb: List[float]) -> float:
        assert self.embedder is not None
        if self._text_bank is None:
            harm = self.embedder.embed_texts(HARMFUL_PROMPTS)
            safe = self.embedder.embed_texts(SAFE_PROMPTS)
            if not harm or not safe:
                self.name = "deferred"
                return 0.0
            self._text_bank = (harm, safe)
            self.name = "clip-zeroshot"

        harm, safe = self._text_bank
        max_h = max(_dot(image_emb, t) for t in harm)
        max_s = max(_dot(image_emb, t) for t in safe)
        # Map advantage into [0, 1]; temperature softens CLIP cosine range
        raw = (max_h - max_s + 0.15) / 0.45
        return round(min(1.0, max(0.0, raw)), 4)
