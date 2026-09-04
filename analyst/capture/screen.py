"""Temporary C2 screen capture via mss (until C1 owns capture).

Encoding note (measured, 1920x1080 dense dashboard, 30 ground-truth strings)
---------------------------------------------------------------------------
This JPEG is not just a preview — `worker` hands these exact bytes to
`pipeline.analyze`, so whatever the encoder throws away is thrown away
*before* OCR ever sees the frame. The original BILINEAR/quality-75 pair was
costing a third of the readable text on a normal desktop:

    downscale filter + quality    strings read (of 30)
    BILINEAR, q75  (was)                  20
    BILINEAR, q92                         24
    LANCZOS,  q75                         23
    LANCZOS,  q92  (now)                  27

This, not the OCR detection cap, is where screen-text accuracy is won; see
`extract/ocr.py`, where moving that cap across 960/1280/1600 scores 27/30
every time. Once a glyph is aliased and JPEG-ringed here, no downstream
setting recovers it.

The failures were not dropped lines — detection found all 30 regions either
way — but character damage inside them: `credits`->`eredits`, `Oct`->`Oet`,
`billing_edit`->`billingedit`, `If your`->`Ifyour`, `Org Admin:`->`Org Admirc`,
and one line truncated halfway. That damage matters more here than it would
in a document scanner: Stage-1 lexicon matching is word-boundary based, so a
swallowed space or underscore hides a term the child was actually shown.

BILINEAR is the wrong filter for *downscaling* (it point-samples a 1.5x
reduction and aliases small glyphs); LANCZOS area-averages correctly. q75
adds ringing around 12-13 px strokes. Together they cost ~200 ms per frame
at 1280 px, which buys back seven garbled lines.

Held out from the tuning above, three fixtures with different fonts, themes
and densities (light-theme chat, dark monospace code, serif document), scored
with the same harness: 26/30 -> 28/30. The change never regressed a fixture.

Why the width matters more than PIL fixtures suggest
----------------------------------------------------
All of the fixtures above are PIL renders, which lack the antialiasing a real
browser applies. Measured against a *real* headless-Chrome render of the same
1920x1080 page (10 known lines from 28 px down to 11 px, including muted
low-contrast greys), the capture width — not the encoder, not the detection
cap — is what decides whether small UI text survives:

    capture width     lines read exactly     OCR ms
    1280                    7/10              1227
    1600                   10/10              1842
    1920                   10/10              2257

At 1280 the three lost lines were all 11-13 px low-contrast greys: a 1920 px
screen scaled to 1280 puts 11 px type at ~7 px, below what the mobile
recogniser resolves. The PIL fixtures scored 1600 as worth only +1 and hid
this completely — treat synthetic renders as a lower bound and re-check
anything important against a real browser capture.

1600 is the default because it is the smallest width that read the real
render perfectly. Cost: the slowest fixture (dense dashboard) takes ~2.3 s in
OCR, and the full warm cascade ~2.5 s, which is level with the default 2.5 s
capture tick. On busy screens raise `interval_s` to 3-4 s, or drop this back
to 1280 to trade small-text recall for headroom. Warm CLIP and the Stage-1
heads add only ~180 ms; the 40 s figures seen on a first frame are one-time
model loads, not per-frame cost.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from typing import Optional, Tuple

from PIL import Image

# See the module docstring for the measurements behind these defaults.
_JPEG_QUALITY = int(os.environ.get("ANALYST_CAPTURE_QUALITY", "92"))

# Frame width handed to the cascade — the resolution OCR actually sees, and
# the knob that decides whether small UI text survives at all (see docstring).
# Keep ANALYST_OCR_SIDE_LEN >= this, or the detection cap discards the gain.
# Drop to 1280 for ~600 ms more headroom against the capture tick, at the cost
# of 11-13 px low-contrast text.
_CAPTURE_WIDTH = int(os.environ.get("ANALYST_CAPTURE_WIDTH", "1600"))

_MSS_OK = False
try:
    import mss  # noqa: F401

    _MSS_OK = True
except Exception:
    mss = None  # type: ignore

_THREAD = threading.local()
_GRAPHICS_ERR = ("bitblt", "graphics function", "screen grab failed")


def _is_graphics_glitch(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in _GRAPHICS_ERR)


def _reset_mss() -> None:
    sct = getattr(_THREAD, "sct", None)
    if sct is not None:
        try:
            sct.close()
        except Exception:
            pass
    _THREAD.sct = None


def _mss_session():
    import mss

    sct = getattr(_THREAD, "sct", None)
    if sct is None:
        sct = mss.MSS()
        _THREAD.sct = sct
    return sct


def _grab_mss(monitor: int):
    import mss

    sct = _mss_session()
    idx = max(0, min(monitor, len(sct.monitors) - 1))
    return sct.grab(sct.monitors[idx])


def _grab_imagegrab() -> Optional[Image.Image]:
    if sys.platform != "win32":
        return None
    try:
        from PIL import ImageGrab

        return ImageGrab.grab(all_screens=True)
    except Exception:
        return None


def _encode_jpeg(img: Image.Image, max_width: int) -> Tuple[bytes, int, int]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w > max_width:
        nh = max(1, int(h * (max_width / w)))
        # LANCZOS, not BILINEAR: this is a downscale, and bilinear aliases
        # small UI glyphs badly enough to corrupt OCR (see module docstring).
        img = img.resize((max_width, nh), Image.Resampling.LANCZOS)
        w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buf.getvalue(), w, h


class ScreenCapture:
    def __init__(self, max_width: Optional[int] = None, monitor: int = 1) -> None:
        self.max_width = _CAPTURE_WIDTH if max_width is None else max_width
        self.monitor = monitor
        self.name = "mss" if _MSS_OK else "none"
        self._last_error = ""

    @property
    def available(self) -> bool:
        return _MSS_OK or sys.platform == "win32"

    @property
    def last_error(self) -> str:
        return self._last_error

    def grab_jpeg(self) -> Tuple[Optional[bytes], int, int]:
        """Return (jpeg_bytes, w, h) or (None, 0, 0) on failure."""
        if not self.available:
            self._last_error = "screen capture unavailable (install mss)"
            return None, 0, 0

        monitors = [self.monitor]
        if self.monitor != 0:
            monitors.append(0)

        last_msg = ""
        for attempt in range(3):
            for mon in monitors:
                if _MSS_OK:
                    try:
                        shot = _grab_mss(mon)
                        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                        data, w, h = _encode_jpeg(img, self.max_width)
                        self._last_error = ""
                        return data, w, h
                    except Exception as exc:
                        last_msg = str(exc)
                        if _is_graphics_glitch(last_msg):
                            _reset_mss()
                        continue

            img = _grab_imagegrab()
            if img is not None:
                data, w, h = _encode_jpeg(img, self.max_width)
                self.name = "imagegrab" if not _MSS_OK else self.name
                self._last_error = ""
                return data, w, h

            if attempt < 2 and _is_graphics_glitch(last_msg):
                _reset_mss()

        if last_msg and _is_graphics_glitch(last_msg):
            self._last_error = "screen capture glitch (Windows BitBlt) — retrying"
        else:
            self._last_error = last_msg or "screen capture failed"
        return None, 0, 0

    def close(self) -> None:
        _reset_mss()
