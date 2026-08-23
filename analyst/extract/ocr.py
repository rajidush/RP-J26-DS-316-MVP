"""OCR — RapidOCR first, Tesseract fallback. Never raises into the pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

_MIN_WIDTH = 640


def _preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < _MIN_WIDTH:
        scale = _MIN_WIDTH / w
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.5)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.3)
    return np.array(rgb)


class OcrEngine:
    def __init__(self) -> None:
        self.name = "none"
        self._rapid = None
        self._rapid_api = ""
        self._tesseract = None
        # Why OCR is unavailable. A broken native install (e.g. the pyclipper
        # DLL failing under a long Windows path) otherwise looks exactly like
        # "package not installed", which is very hard to diagnose remotely.
        self.last_error = ""
        self._init_rapid()
        if self._rapid is None:
            self._init_tesseract()

    def _init_rapid(self) -> None:
        """rapidocr-onnxruntime caps at Python <3.13; `rapidocr` 3.x is its
        successor and returns a result object instead of a tuple. Try the
        original first so existing environments keep their exact behaviour."""
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapid = RapidOCR()
            self._rapid_api = "tuple"
            self.name = "rapidocr"
            return
        except ImportError:
            self._rapid = None
        except Exception as exc:
            self._rapid = None
            self.last_error = f"rapidocr-onnxruntime: {type(exc).__name__}: {exc}"

        try:
            from rapidocr import RapidOCR

            self._rapid = RapidOCR()
            self._rapid_api = "object"
            self.name = "rapidocr3"
            self.last_error = ""
        except Exception as exc:
            self._rapid = None
            self.last_error = f"rapidocr: {type(exc).__name__}: {exc}"

    def _init_tesseract(self) -> None:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._tesseract = pytesseract
            self.name = "tesseract"
            self.last_error = ""
        except Exception:
            self._tesseract = None

    def extract(self, image: Optional[Image.Image]) -> str:
        if image is None:
            return ""
        if self._rapid is not None:
            try:
                raw = self._rapid(_preprocess(image))
                if self._rapid_api == "object":
                    lines = list(getattr(raw, "txts", None) or [])
                else:
                    result = raw[0] if isinstance(raw, tuple) else raw
                    lines = [
                        row[1] for row in (result or []) if len(row) > 1 and row[1]
                    ]
                return " ".join(t for t in lines if t).strip()
            except Exception:
                pass
        if self._tesseract is not None:
            try:
                return (self._tesseract.image_to_string(image.convert("RGB")) or "").strip()
            except Exception:
                return ""
        return ""
