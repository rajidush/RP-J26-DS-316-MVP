"""Stage-1 lexicon — explainable, instant, no weights.

Score bands are chosen so the persona thresholds in decide.py actually
discriminate: the MID band straddles theta2 (0.55 / 0.65 / 0.75) so the same
phrase can escalate for an 8-year-old and clear for a 15-year-old. The HIGH
band (slurs, self-harm directives) sits above every threshold by design.
"""

from __future__ import annotations

import re
from typing import List, Tuple

HIGH_PHRASES = (
    "kill yourself",
    "kill your self",
    "kill urself",
    "killurself",
    "kys",
    "go die",
    "you should die",
    "hang yourself",
    "rape",
    "raped",
    "raping",
    "rapist",
    "gas the jews",
    "gas the lot of you",
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
    "your kind of people",
    "go back to your country",
    "go back to where you came from",
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

THREAT_MARKERS = (
    "kys",
    "kill yourself",
    "kill your self",
    "kill urself",
    "killurself",
    "go die",
    "you should die",
    "hang yourself",
)

SEXUAL_MARKERS = (
    "rape",
    "raped",
    "raping",
    "rapist",
)

IDENTITY_MARKERS = (
    "nigger",
    "nigga",
    "faggot",
    "tranny",
    "shemale",
    "retard",
    "retarded",
    "gas the jews",
    "gas the lot of you",
)

# Score bands (see module docstring).
HIGH_BASE, HIGH_STEP, HIGH_CAP = 0.88, 0.03, 0.98
MID_BASE, MID_STEP, MID_CAP = 0.52, 0.06, 0.80
GAMING_SCORE = 0.18
NEUTRAL_SCORE = 0.08
EMPTY_SCORE = 0.05

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

# "kiiill" -> "kiill" (keep doubles: "kill", "gg"); 3+ runs are obfuscation.
_RUNS = re.compile(r"(.)\1{2,}")
# "k y s" -> "kys": 3+ single letters separated by whitespace.
_SPACED = re.compile(r"\b(?:[a-z]\s+){2,}[a-z]\b")


def normalize(text: str) -> str:
    lowered = (text or "").lower().translate(_LEET)
    collapsed = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def collapse_runs(blob: str, repeat: int) -> str:
    return _RUNS.sub(lambda m: m.group(1) * repeat, blob)


def despace(blob: str) -> str:
    """Second view of the text with letter-spaced obfuscation joined up."""
    return _SPACED.sub(lambda m: re.sub(r"\s+", "", m.group(0)), blob)


def _pattern(needle: str) -> re.Pattern:
    # Word-boundary anchored, tolerant of runs of whitespace between words.
    parts = [re.escape(part) for part in needle.split()]
    return re.compile(rf"\b{r'\s+'.join(parts)}\b")


_COMPILED = {p: _pattern(p) for p in HIGH_PHRASES + MID_PHRASES + GAMING_BENIGN}


def _contains(haystack: str, needle: str) -> bool:
    pattern = _COMPILED.get(needle) or _pattern(needle)
    return pattern.search(haystack) is not None


def _hits(views: Tuple[str, ...], phrases: Tuple[str, ...]) -> List[str]:
    found = []
    for phrase in phrases:
        if any(_contains(view, phrase) for view in views):
            found.append(phrase)
    return found


def score_text(text: str) -> Tuple[float, str, List[str]]:
    """Return (score, category, lexicon_hits). Categories match eng-plan names."""
    blob = normalize(text)
    if not blob:
        return EMPTY_SCORE, "none", []

    doubled = collapse_runs(blob, 2)
    singled = collapse_runs(blob, 1)
    views = (blob, doubled, singled, despace(blob), despace(singled))
    high = _hits(views, HIGH_PHRASES)
    mid = _hits(views, MID_PHRASES)

    if high:
        score = round(min(HIGH_CAP, HIGH_BASE + HIGH_STEP * (len(high) - 1)), 4)
        if any(marker in high for marker in THREAT_MARKERS):
            return score, "threat", high
        if any(marker in high for marker in IDENTITY_MARKERS):
            return score, "hate_identity", high
        if any(marker in high for marker in SEXUAL_MARKERS):
            return score, "sexual_harassment", high
        return score, "bullying", high

    if mid:
        return round(min(MID_CAP, MID_BASE + MID_STEP * len(mid)), 4), "bullying", mid

    if _hits(views, GAMING_BENIGN):
        return GAMING_SCORE, "none", []

    return NEUTRAL_SCORE, "none", []
