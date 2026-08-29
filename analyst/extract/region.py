"""Find the picture a child is actually looking at, inside a full screen grab.

Why this exists
---------------
Measured on a 450M vision-language model (lfm2.5-vl-450m), with a bullying
poster embedded in a browser window:

    full desktop  -> "a red square with the text GUARANTEE"   (hallucinated)
                  -> transcribed an invented website nav menu (fabricated)
    cropped poster -> "yellow text that says nobody likes you,
                       go back to your country"                (correct)
                   -> "NO BODY LIKES YOU / GO BACK TO YOUR COUNTRY" (correct)

A small VLM given a whole desktop does not say "I am not sure" -- it invents
content confidently. Feeding that into a hate cascade is worse than running no
vision branch at all, because the fabrication is indistinguishable from a real
reading. So we crop to the dominant picture region first, and if we cannot find
one we decline to ask.

Approach: browser and OS chrome is pale, flat and low-saturation; photos, memes
and posters are saturated and busy. Score a coarse grid on saturation + local
contrast, keep the hot cells, take the largest connected blob, and return its
bounding box. Pure numpy/PIL -- no extra dependency.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Analysis resolution: small is fine and keeps this well under a millisecond.
_ANALYSIS_W = 320
_GRID = 20                  # cells across
_MIN_AREA_FRAC = 0.04       # ignore blobs smaller than 4% of the frame
_MAX_AREA_FRAC = 0.92       # a blob covering ~everything means "no distinct picture"
_PAD_FRAC = 0.02            # breathing room around the crop
# A page of black text on white is high-contrast but has no colour. Pictures,
# memes and posters do. Without this floor a plain document reads as a picture
# and we waste a VLM call on something OCR already handled.
_MIN_BLOB_SATURATION = 0.12
# Gallery detection: how many blobs of comparable size mean "grid of pictures"
# rather than "one picture". See the guard in find_image_region().
_PEER_SIZE_FRAC = 0.45      # a peer is at least 45% the size of the largest
_GALLERY_MIN_PEERS = 4      # this many comparable blobs = a gallery

Box = Tuple[int, int, int, int]  # left, top, right, bottom in ORIGINAL pixels


def _cell_scores(rgb: np.ndarray, rows: int, cols: int):
    """Per-cell (interest, saturation) grids. Interest drives selection;
    saturation is kept separately to reject text-only regions."""
    arr = rgb.astype(np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    gray = arr.mean(axis=2)
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    contrast = np.clip((gx + gy) * 4.0, 0.0, 1.0)

    interest = 0.65 * saturation + 0.35 * contrast

    h, w = interest.shape
    out = np.zeros((rows, cols), dtype=np.float32)
    sat = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0, y1 = r * h // rows, max((r + 1) * h // rows, r * h // rows + 1)
        for c in range(cols):
            x0, x1 = c * w // cols, max((c + 1) * w // cols, c * w // cols + 1)
            out[r, c] = float(interest[y0:y1, x0:x1].mean())
            sat[r, c] = float(saturation[y0:y1, x0:x1].mean())
    return out, sat


def _blobs(hot: np.ndarray) -> List[List[Tuple[int, int]]]:
    """All 4-connected components of True cells, largest first."""
    rows, cols = hot.shape
    seen = np.zeros_like(hot, dtype=bool)
    found: List[List[Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not hot[r, c] or seen[r, c]:
                continue
            comp: List[Tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and hot[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            found.append(comp)
    found.sort(key=len, reverse=True)
    return found


def _largest_blob(hot: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Largest 4-connected component of True cells."""
    found = _blobs(hot)
    return found[0] if found else None


def find_image_region(image: Optional[Image.Image]) -> Optional[Box]:
    """Bounding box of the dominant picture, or None if the frame is all chrome."""
    if image is None:
        return None
    try:
        rgb_full = image.convert("RGB")
        ow, oh = rgb_full.size
        if ow < 64 or oh < 64:
            return None

        scale = min(1.0, _ANALYSIS_W / ow)
        small = rgb_full.resize(
            (max(1, int(ow * scale)), max(1, int(oh * scale))), Image.BILINEAR
        )
        arr = np.asarray(small)

        rows = max(4, int(_GRID * arr.shape[0] / max(arr.shape[1], 1)))
        cells, sat_cells = _cell_scores(arr, rows, _GRID)

        # "Hot" = clearly above this screen's own baseline. Using the frame's
        # own statistics keeps it adaptive to light and dark themes alike.
        thresh = float(cells.mean() + 0.6 * cells.std())
        hot = cells > max(thresh, 0.10)
        if not hot.any():
            return None

        found = _blobs(hot)
        if not found:
            return None
        blob = found[0]

        # Gallery guard. A search-results page or an image grid produces many
        # comparable blobs rather than one dominant picture. Cropping to the
        # biggest of them hands the vision model a wall of thumbnails, and a
        # 450M model given something it cannot resolve does not hedge — it
        # invents. Measured on a 12-thumbnail grid it reported
        # "MASSACHUSETTS ELECTION 2018", none of which was on screen.
        #
        # A fabricated reading is worse than no reading, because nothing
        # downstream can tell the two apart. So when the frame looks like a
        # gallery we decline, and OCR alone covers it.
        peers = sum(1 for b in found if len(b) >= len(blob) * _PEER_SIZE_FRAC)
        if peers >= _GALLERY_MIN_PEERS:
            return None

        frac = len(blob) / float(cells.size)
        if frac < _MIN_AREA_FRAC or frac > _MAX_AREA_FRAC:
            return None

        blob_sat = float(np.mean([sat_cells[r, c] for r, c in blob]))
        if blob_sat < _MIN_BLOB_SATURATION:
            return None  # text/UI, not a picture -- OCR already covers it

        rs = [r for r, _ in blob]
        cs = [c for _, c in blob]
        top = min(rs) / rows
        bottom = (max(rs) + 1) / rows
        left = min(cs) / _GRID
        right = (max(cs) + 1) / _GRID

        pad = _PAD_FRAC
        left = max(0.0, left - pad)
        top = max(0.0, top - pad)
        right = min(1.0, right + pad)
        bottom = min(1.0, bottom + pad)

        box = (int(left * ow), int(top * oh), int(right * ow), int(bottom * oh))
        if box[2] - box[0] < 48 or box[3] - box[1] < 48:
            return None
        return box
    except Exception:
        return None


def crop_to_region(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], Optional[Box]]:
    """(cropped image, box). Falls back to (None, None) when nothing stands out."""
    box = find_image_region(image)
    if box is None or image is None:
        return None, None
    try:
        return image.convert("RGB").crop(box), box
    except Exception:
        return None, None
