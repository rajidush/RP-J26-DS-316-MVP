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

This cap is NOT where OCR accuracy is won (measured 1 Sep 2026)
--------------------------------------------------------------
Screen text was being read badly, and this cap looked like the obvious cause:
`capture/screen` hands over a 1280 px frame, so a 960 cap resizes it again.
It is not the cause. Holding the capture encoder at LANCZOS/q92 and moving
only this knob, on a 1920x1080 dashboard fixture with 30 known strings:

    detection cap      strings read (of 30)      ms
    960                        27               2432
    1280                       27               2275
    1600                       27               2354

Identical. The entire loss was upstream in the capture encoder — see the
table in `capture/screen.py`, where BILINEAR/q75 -> LANCZOS/q92 moves the
same fixture from 20/30 to 27/30. Raising this cap *alone*, with the old
encoder, also changed nothing (20/30). Text that was aliased and
JPEG-ringed before detection cannot be recovered by detecting it larger.

What this knob must NOT do is bind below the capture width. It is the ceiling
on what detection sees, so it has to track `ANALYST_CAPTURE_WIDTH` (now 1600):
capture hands over a 1600 px frame, and a 960 or 1280 cap would shrink it
again and throw away exactly the small-text recall that widening capture just
bought. That is why this default moves with it rather than independently.

So: raise capture width to read smaller text, and keep this at or above it.
Raising this alone buys nothing — fix the encoder and the width instead.

PP-OCRv5 SERVER det/rec weights were tried for the residual errors and did
not finish a single frame inside 10 minutes on this CPU, so mobile stays.
Three of the five remaining errors are `AI`->`Al`: in most sans UI fonts
capital-I and lowercase-l are the same glyph, so that is not recoverable
from pixels at all, by any engine.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

_MIN_WIDTH = 640

# See the module docstring for the measurements behind these defaults.
_DET_SIDE_LEN = int(os.environ.get("ANALYST_OCR_SIDE_LEN", "1600"))
_OCR_THREADS = int(os.environ.get("ANALYST_OCR_THREADS", "4"))
_OCR_USE_CLS = os.environ.get("ANALYST_OCR_CLS", "0").strip() in ("1", "true", "yes")
# Sharpen/contrast enhancement, off by default — see _preprocess().
_OCR_ENHANCE = os.environ.get("ANALYST_OCR_ENHANCE", "0").strip() in ("1", "true", "yes")


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
    """Upscale very small frames. No sharpening — it was measurably harmful.

    This used to apply Sharpness(1.5) and Contrast(1.3). On a grid of 150 px
    meme thumbnails that cost every phrase OCR would otherwise have read:

        raw           2/4 target phrases recovered
        enhanced      0/4

    and on a full-size meme it changed nothing (2/2 either way). Unsharp
    masking amplifies JPEG ringing around small glyphs, so the detector loses
    strokes it could otherwise resolve — the enhancement only ever cost recall.

    Set ANALYST_OCR_ENHANCE=1 to restore the old behaviour for comparison.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < _MIN_WIDTH:
        scale = _MIN_WIDTH / w
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if _OCR_ENHANCE:
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

    def read(self, image: Optional[Image.Image]) -> Tuple[str, list]:
        """One OCR pass -> joined text plus per-line boxes.

        Both come from the same inference deliberately. Running OCR twice to
        collect boxes would cost a second full pass, which is ~7 s on a real
        1280x720 desktop — the single most expensive step in the cascade.

        Boxes are normalised to 0..1 of the frame so they survive the 320 px
        blurred thumbnail the panel draws on, and any change to capture
        resolution. `_preprocess` scales uniformly, so normalised coordinates
        are identical before and after it.
        """
        if image is None:
            return "", []
        if self._rapid is not None:
            try:
                arr = _preprocess(image)
                raw = self._rapid(arr)
                height, width = arr.shape[0], arr.shape[1]
                if self._rapid_api == "object":
                    lines = list(getattr(raw, "txts", None) or [])
                    polys = getattr(raw, "boxes", None)
                    confs = list(getattr(raw, "scores", None) or [])
                else:
                    result = raw[0] if isinstance(raw, tuple) else raw
                    rows = [r for r in (result or []) if len(r) > 1 and r[1]]
                    lines = [r[1] for r in rows]
                    polys = [r[0] for r in rows]
                    confs = [r[2] if len(r) > 2 else 0.0 for r in rows]
                text = " ".join(t for t in lines if t).strip()
                return text, _regions(polys, lines, confs, width, height)
            except Exception:
                pass
        if self._tesseract is not None:
            try:
                # Tesseract path has no boxes here; text still works.
                return (self._tesseract.image_to_string(image.convert("RGB")) or "").strip(), []
            except Exception:
                return "", []
        return "", []

    def extract(self, image: Optional[Image.Image]) -> str:
        """Text only. Prefer read() when the caller also wants positions."""
        return self.read(image)[0]


def _regions(polys, lines, confs, width: int, height: int) -> list:
    if polys is None or width <= 0 or height <= 0:
        return []
    out = []
    for i, poly in enumerate(polys):
        try:
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
        except Exception:
            continue
        out.append({
            "kind": "text",
            "box": [
                round(max(0.0, min(1.0, min(xs) / width)), 4),
                round(max(0.0, min(1.0, min(ys) / height)), 4),
                round(max(0.0, min(1.0, max(xs) / width)), 4),
                round(max(0.0, min(1.0, max(ys) / height)), 4),
            ],
            "text": (lines[i] if i < len(lines) else "")[:120],
            "conf": round(float(confs[i]), 3) if i < len(confs) else 0.0,
        })
    return out
