"""Always-on Stage-1/Stage-2 text scorer. No weights, no network, no GPU.

Fine-tuned MiniLM later plugs in beside this; scores are combined with max()
so a missing transformer cannot block a demo.
"""

from __future__ import annotations

import re
from typing import Tuple

# High-severity hate / threat / identity attack. Matched on normalized text.
HIGH_PHRASES = (
    "kill yourself",
    "kys",
    "go die",
    "you should die",
    "hang yourself",
    "rape",
    "gas the",
    "nigger",
    "nigga",
    "faggot",
    "retard",
    "retarded",
    "tranny",
    "shemale",
)

# Cyberbullying / harassment that is still in-scope for this component.
MID_PHRASES = (
    "nobody likes you",
    "kill yourself already",
    "your kind",
    "go back to",
    "hate you",
    "hate all",
    "subhuman",
    "monkey",
    "slave",
    "ill find you",
    "i'll find you",
    "i will find you",
)

# Gaming trash talk — must NOT alone push the score over the 0.85 gate.
GAMING_BENIGN = (
    "ez",
    "easy",
    "noob",
    "gg",
    "gg ez",
    "trash",
    "bot",
    "lmao",
    "ratio",
)

_LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


def normalize(text: str) -> str:
    lowered = (text or "").lower().translate(_LEET)
    collapsed = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _contains(haystack: str, needle: str) -> bool:
    if " " in needle:
        return needle in haystack
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def score_text(text: str) -> Tuple[float, str]:
    """Return (score in [0, 1], coarse category)."""
    blob = normalize(text)
    if not blob:
        return 0.05, "none"

    high_hits = [p for p in HIGH_PHRASES if _contains(blob, p)]
    mid_hits = [p for p in MID_PHRASES if _contains(blob, p)]

    if high_hits:
        score = min(0.98, 0.88 + 0.03 * (len(high_hits) - 1))
        joined = " ".join(high_hits)
        if any(p in joined for p in ("kys", "kill yourself", "go die", "hang yourself")):
            return score, "threat"
        if any(
            p in joined
            for p in ("nigger", "nigga", "faggot", "tranny", "shemale", "retard", "retarded")
        ):
            return score, "identity_attack"
        return score, "hate_speech"

    if mid_hits:
        score = min(0.86, 0.72 + 0.05 * len(mid_hits))
        return score, "cyberbullying"

    # Gaming slang alone stays well under the intercept gate.
    if any(_contains(blob, p) for p in GAMING_BENIGN):
        return 0.18, "none"

    return 0.08, "none"
