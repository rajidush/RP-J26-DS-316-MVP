"""OCR — RapidOCR first, Tesseract fallback. Never raises into the pipeline.

Performance note (measured, 1000x220 chat screenshot, 12-core CPU)
-----------------------------------------------------------------
RapidOCR 3.x defaults are tuned for document scans, not screen capture, and
cost 18.8 s per frame here — far past the 2.5 s capture tick, so the worker
could never keep up:

    default (PP-OCRv6 small, limit_type=min/736, all cores)   18760 ms
    limit_type=max/960                                         4991 ms
    + intra_op_num_threads=4                                   3166 ms
    + PP-OCRv5 mobile det+rec, no angle classifier             1313 ms

All four produced identical text. The dominant cost was the resize rule:
`limit_type: min` scales the *shorter* side up to 736, so a 1000x220 chat bar
became 3345x736 — a 2.5 MP detection input for one line of text. Capping the
*longer* side instead is the correct rule for screenshots.

The angle classifier is off because screen text is upright; re-enable it with
ANALYST_OCR_CLS=1 if rotated content matters. Every knob is env-overridable so
a slower, more thorough profile is one variable away.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

_MIN_WIDTH = 640

# See the module docstring for the measurements behind these defaults.
_DET_SIDE_LEN = int(os.environ.get("ANALYST_OCR_SIDE_LEN", "960"))
_OCR_THREADS = int(os.environ.get("ANALYST_OCR_THREADS", "4"))
_OCR_USE_CLS = os.environ.get("ANALYST_OCR_CLS", "0").strip() in ("1", "true", "yes")


def _rapid3_params() -> dict:
    """Screen-capture profile for RapidOCR 3.x (see module docstring)."""
    from rapidocr import ModelType, OCRVersion

    return {
        # Cap the LONGER side. The default caps the shorter side, which upscales
        # wide-and-short screenshots into megapixel detection inputs.
        "Det.limit_type": "max",
        "Det.limit_side_len": _DET_SIDE_LEN,
        # PP-OCRv5 mobile: same text on our assets, ~2.4x faster than v6 small.
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
        # Small models contend on 12 threads; 4 measured fastest here.
        "EngineConfig.onnxruntime.intra_op_num_threads": _OCR_THREADS,
        "Global.use_cls": _OCR_USE_CLS,
    }


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

            self._rapid = RapidOCR(params=_rapid3_params())
            self._rapid_api = "object"
            self.name = "rapidocr3"
            self.last_error = ""
        except Exception as exc:
            # A bad params key must not cost us OCR entirely — fall back to the
            # library defaults (slow, but working) and say so.
            try:
                from rapidocr import RapidOCR

                self._rapid = RapidOCR()
                self._rapid_api = "object"
                self.name = "rapidocr3-untuned"
                self.last_error = f"tuned config rejected ({type(exc).__name__}); using defaults"
                return
            except Exception as exc2:
                self._rapid = None
                self.last_error = f"rapidocr: {type(exc2).__name__}: {exc2}"

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
