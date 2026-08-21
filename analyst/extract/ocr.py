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
        self._tesseract = None
        self._init_rapid()
        if self._rapid is None:
            self._init_tesseract()

    def _init_rapid(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapid = RapidOCR()
            self.name = "rapidocr"
        except Exception:
            self._rapid = None

    def _init_tesseract(self) -> None:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._tesseract = pytesseract
            self.name = "tesseract"
        except Exception:
            self._tesseract = None

    def extract(self, image: Optional[Image.Image]) -> str:
        if image is None:
            return ""
        if self._rapid is not None:
            try:
                result, _ = self._rapid(_preprocess(image))
                if not result:
                    return ""
                lines = [row[1] for row in result if len(row) > 1 and row[1]]
                return " ".join(lines).strip()
            except Exception:
                pass
        if self._tesseract is not None:
            try:
                return (self._tesseract.image_to_string(image.convert("RGB")) or "").strip()
            except Exception:
                return ""
        return ""
