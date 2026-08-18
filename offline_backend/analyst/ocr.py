"""OCR plug. RapidOCR → Tesseract → empty string. Never raises to the pipeline."""

from __future__ import annotations

from typing import Optional

from PIL import Image


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
        rgb = image.convert("RGB")
        if self._rapid is not None:
            try:
                result, _ = self._rapid(rgb)
                if not result:
                    return ""
                lines = [row[1] for row in result if len(row) > 1 and row[1]]
                return " ".join(lines).strip()
            except Exception:
                pass
        if self._tesseract is not None:
            try:
                return (self._tesseract.image_to_string(rgb) or "").strip()
            except Exception:
                return ""
        return ""
