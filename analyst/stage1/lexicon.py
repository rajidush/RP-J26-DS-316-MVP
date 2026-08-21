"""Stage-1 lexicon — explainable, instant, no weights."""

from __future__ import annotations

import re
from typing import List, Tuple

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

MID_PHRASES = (
    "nobody likes you",
    "your kind",
    "go back to",
    "hate you",
    "hate all",
    "subhuman",
    "ill find you",
    "i'll find you",
    "i will find you",
)

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
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)


def normalize(text: str) -> str:
    lowered = (text or "").lower().translate(_LEET)
    collapsed = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _contains(haystack: str, needle: str) -> bool:
    if " " in needle:
        return needle in haystack
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def score_text(text: str) -> Tuple[float, str, List[str]]:
    """Return (score, category, lexicon_hits). Categories match eng-plan names."""
    blob = normalize(text)
    if not blob:
        return 0.05, "none", []

    high = [p for p in HIGH_PHRASES if _contains(blob, p)]
    mid = [p for p in MID_PHRASES if _contains(blob, p)]

    if high:
        score = min(0.98, 0.88 + 0.03 * (len(high) - 1))
        joined = " ".join(high)
        if any(p in joined for p in ("kys", "kill yourself", "go die", "hang yourself")):
            return score, "threat", high
        if any(
            p in joined
            for p in ("nigger", "nigga", "faggot", "tranny", "shemale", "retard", "retarded")
        ):
            return score, "hate_identity", high
        return score, "bullying", high

    if mid:
        return min(0.86, 0.72 + 0.05 * len(mid)), "bullying", mid

    if any(_contains(blob, p) for p in GAMING_BENIGN):
        return 0.18, "none", []

    return 0.08, "none", []
