"""Screen grab for demo when no image is uploaded. Never writes a PNG to disk."""

from __future__ import annotations

import io
from typing import Optional, Tuple


def grab_screen_jpeg() -> Tuple[Optional[bytes], str]:
    try:
        from PIL import ImageGrab
    except Exception:
        return None, "none"

    try:
        image = ImageGrab.grab()
        if image is None:
            return None, "pillow_grab"
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=70)
        return buf.getvalue(), "pillow_grab"
    except Exception:
        return None, "pillow_grab_failed"
