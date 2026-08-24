"""Image embedding — CLIP ViT-B/32 when available; else empty (vision deferred).

Demo strategy: pretrained OpenAI CLIP via transformers+torch (CPU).
Fine-tune / MobileCLIP ONNX comes later (Step 6 / eng plan A3).
Missing deps → return [] so text/OCR/ASR cascade still works.
Model loads lazily on first embed() so text-only CLI stays fast.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ImageEmbedder:
    def __init__(
        self,
        embed_fn: Optional[Callable[[Image.Image], List[float]]] = None,
    ) -> None:
        self._override = embed_fn
        self._model = None
        self._processor = None
        self._torch = None
        self._load_attempted = False
        self.dim = 0
        if self._override is not None:
            self.name = "injected"
        else:
            # Probe availability without downloading weights yet.
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401

                self.name = "clip:pending"
            except Exception:
                self.name = "deferred"

    def _ensure_loaded(self) -> bool:
        if self._override is not None:
            return False
        if self._model is not None:
            return True
        if self._load_attempted or self.name == "deferred":
            return False
        self._load_attempted = True
        model_id = os.environ.get("ANALYST_CLIP_MODEL", DEFAULT_CLIP_MODEL).strip()
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(model_id)
            self._model = CLIPModel.from_pretrained(model_id)
            self._model.eval()
            self.dim = int(self._model.config.projection_dim)
            self.name = f"clip:{model_id.split('/')[-1]}"
            self._torch = torch
            return True
        except Exception:
            self._model = None
            self._processor = None
            self.name = "deferred"
            return False

    def embed(self, image: Optional[Image.Image]) -> List[float]:
        if image is None:
            return []
        if self._override is not None:
            try:
                return list(self._override(image))
            except Exception:
                return []
        if not self._ensure_loaded():
            return []
        try:
            torch = self._torch
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                vision = self._model.vision_model(pixel_values=inputs["pixel_values"])
                pooled = vision.pooler_output
                feats = self._model.visual_projection(pooled)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].detach().cpu().tolist()
        except Exception:
            self.name = "clip_failed"
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Text tower — used by ImageFast zero-shot. Empty if CLIP missing."""
        if not texts:
            return []
        if not self._ensure_loaded():
            return []
        try:
            torch = self._torch
            inputs = self._processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                text_out = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                pooled = text_out.pooler_output
                feats = self._model.text_projection(pooled)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.detach().cpu().tolist()
        except Exception:
            return []
