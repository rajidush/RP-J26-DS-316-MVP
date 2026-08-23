"""Temporary C2 screen capture via mss (until C1 owns capture)."""

from __future__ import annotations

import io
import sys
import threading
from typing import Optional, Tuple

from PIL import Image

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
        img = img.resize((max_width, nh), Image.Resampling.BILINEAR)
        w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    return buf.getvalue(), w, h


class ScreenCapture:
    def __init__(self, max_width: int = 1280, monitor: int = 1) -> None:
        self.max_width = max_width
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
