"""Hate-Speech Detection Module (The Analyst) — Component 2.

Cascade: capture-to-RAM → Stage 1 (cheap) → Stage 2 (fusion) → JSON → delete media.
Optional OCR / ASR / ONNX backends are plugged in when present; lexicon always works.
"""

from .pipeline import AnalystPipeline, get_pipeline

__all__ = ["AnalystPipeline", "get_pipeline"]
