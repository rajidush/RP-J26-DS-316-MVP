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
    # "hate all" alone matched "i hate all vegetables honestly" — the phrase
    # only carries hate when the object is a person or group, so anchor it.
    "hate all of you",
    "hate all of them",
    "hate you people",
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


# --- Pattern layer -----------------------------------------------------------
# Literal phrases cover slurs well but miss whole categories that are phrased
# freely: threats, personal degradation, grooming/sexual coercion and
# dehumanisation. Measured on the development set, the literal list scored 20%
# on threats, 20% on bullying and 0% on sexual harassment.
#
# These are regex *families* rather than a bag of words, kept high-precision on
# purpose: this layer must stay auditable, so a hit names the family that fired
# and a reviewer can read the rule. Everything statistical belongs in the model.

# Competitive trash talk reuses violent phrasing ("gonna kill you in this
# match"). Threat patterns are suppressed when the surrounding text is clearly
# about a game — the single largest false-positive source for this age group.
_GAMING_CONTEXT = re.compile(
    r"\b(?:match|game|gaming|round|lobby|respawn|spawn|noob|gg|ez|team|teammate|"
    r"boss|raid|level|server|killstreak|headshot|squad|clutch|scoreboard|rank|"
    r"loot|xp|queue|map|player|players|minecraft|fortnite|roblox|valorant|"
    r"overwatch|cod|warzone|apex|smash)\b"
)


def in_gaming_context(text: str) -> bool:
    """True when the surrounding words are clearly about playing a game.

    Exported because the statistical heads need it too: a generic toxicity
    model reads "you're trash at this game" and "im gonna kill you in this
    match" as abuse, which is the single largest false-positive source for this
    age group. The lexicon's explicit layer is unaffected, so a real slur in a
    game lobby still fires.
    """
    return bool(_GAMING_CONTEXT.search(normalize(text)))


class Family:
    __slots__ = ("name", "category", "band", "patterns", "gaming_sensitive")

    def __init__(self, name, category, band, patterns, gaming_sensitive=False):
        self.name = name
        self.category = category
        self.band = band  # "high" | "mid"
        self.patterns = tuple(re.compile(p) for p in patterns)
        self.gaming_sensitive = gaming_sensitive

    def matches(self, views: Tuple[str, ...]) -> bool:
        return any(p.search(v) for p in self.patterns for v in views)


PATTERN_FAMILIES: Tuple[Family, ...] = (
    Family(
        "threat:violence_intent", "threat", "high",
        (
            r"\b(?:i|we)\s*(?:'m|m|am|re|are)?\s*(?:gonna|going to|gna|will)\s+"
            r"(?:hurt|beat|jump|batter|smash|end|destroy)\s+(?:you|u|him|her|them)\b",
            r"\bme\s+and\s+my\s+(?:friends|mates|boys)\s+(?:are\s+)?(?:gonna|going to)\s+"
            r"(?:jump|get|hurt|beat)\s+(?:you|u)\b",
            r"\bwait\s+(?:till|until)\s+(?:i|we)\s+(?:see|catch|find)\s+you\b",
        ),
        gaming_sensitive=True,
    ),
    Family(
        "threat:intimidation", "threat", "mid",
        (
            r"\bwatch\s+your\s+back\b",
            r"\bi\s+know\s+where\s+you\s+live\b",
            r"\byou'?re\s+(?:gonna|going to)\s+(?:regret|pay for)\s+(?:it|this|that)\b",
            r"\bmeet\s+me\s+(?:after\s+school|behind\s+\w+)\b",
        ),
        gaming_sensitive=True,
    ),
    Family(
        "bullying:degradation", "bullying", "mid",
        (
            r"\byou'?re\s+(?:so\s+|such\s+a\s+|a\s+)?(?:worthless|pathetic|useless|"
            r"disgusting|waste\s+of\s+space|waste\s+of\s+air)\b",
            r"\byou'?re\s+so\s+ugly\b",
            r"\bno\s+wonder\s+(?:you|nobody)\b.{0,20}\b(?:no\s+friends|likes?\s+you)\b",
            r"\beveryone\s+(?:hates|laughs\s+at|makes\s+fun\s+of)\s+you\b",
            r"\beveryone\s+in\s+\w+\s+laughs\s+at\s+you\b",
        ),
    ),
    Family(
        "bullying:exclusion", "bullying", "mid",
        (
            r"\bnobody\s+(?:likes|wants|needs|cares\s+about)\s+you\b",
            r"\bno\s+one\s+(?:likes|wants|cares\s+about)\s+you\b",
            r"\b(?:everyone|we|we'?d\s+all|everybody)\s+(?:would\s+)?be\s+"
            r"(?:better|happier)\s*(?:off\s*)?(?:without\s+you|if\s+you\s+(?:were\s+)?(?:gone|left|died))\b",
            r"\bwhy\s+(?:do|would)\s+you\s+even\s+(?:bother|try|exist|show\s+up|come)\b",
            r"\bgo\s+away\s+nobody\s+wants\s+you\b",
        ),
    ),
    Family(
        "sexual:coercion", "sexual_harassment", "high",
        (
            r"\bsend\s+(?:me\s+)?(?:nudes|nude|pics|pictures|photos)\b",
            r"\btake\s+(?:your|ur|yr)\s+clothes\s+off\b",
            r"\btake\s+(?:it|them)\s+off\s+for\s+(?:me|the\s+camera)\b",
            r"\bdon'?t\s+tell\s+(?:your|ur)\s+(?:parents|mum|mom|dad|family)\b",
        ),
    ),
    Family(
        "identity:dehumanisation", "hate_identity", "high",
        (
            r"\b(?:they|them|those\s+people|their\s+kind|your\s+kind|your\s+sort|"
            r"their\s+sort)\b.{0,40}\b(?:animals|vermin|rats|roaches|subhuman|"
            r"filth|scum|parasites?)\b",
            r"\b(?:breed|multiply|spread)\s+like\s+(?:vermin|rats|animals|flies|roaches)\b",
            r"\bshould\s+be\s+(?:caged|exterminated|wiped\s+out|put\s+down)\b",
        ),
    ),
    Family(
        "identity:exclusion", "hate_identity", "mid",
        (
            r"\b(?:people\s+like\s+(?:you|them)|your\s+kind|their\s+kind|your\s+sort|"
            r"their\s+lot)\s+(?:should\s*n'?t|should\s+not)\s+be\s+(?:allowed|here|let\s+in)\b",
            r"\bshould\s*n'?t\s+be\s+allowed\s+in\s+(?:this|our|the)\s+country\b",
            r"\b(?:their|your)\s+kind\s+(?:always|never)\s+\w+",
            r"\bwe'?d\s+(?:all\s+)?be\s+better\s+off\s+without\s+(?:your|their)\s+"
            r"(?:sort|kind|lot|people)\b",
            r"\btake\s+our\s+jobs\b",
        ),
    ),
)


def _family_hits(views: Tuple[str, ...], gaming: bool) -> List[Tuple[str, str, str]]:
    """(family_name, category, band) for every family that fires."""
    out: List[Tuple[str, str, str]] = []
    for family in PATTERN_FAMILIES:
        if gaming and family.gaming_sensitive:
            continue
        if family.matches(views):
            out.append((family.name, family.category, family.band))
    return out


def _category_for_literals(high: List[str]) -> str:
    if any(marker in high for marker in THREAT_MARKERS):
        return "threat"
    if any(marker in high for marker in IDENTITY_MARKERS):
        return "hate_identity"
    if any(marker in high for marker in SEXUAL_MARKERS):
        return "sexual_harassment"
    return "bullying"


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

    gaming = bool(_GAMING_CONTEXT.search(blob))
    families = _family_hits(views, gaming)
    high_families = [f for f in families if f[2] == "high"]
    mid_families = [f for f in families if f[2] == "mid"]

    # Literal slurs outrank patterns for category, since they are the most
    # specific evidence available; patterns fill the categories they cannot see.
    if high or high_families:
        hits = high + [f[0] for f in high_families]
        score = round(min(HIGH_CAP, HIGH_BASE + HIGH_STEP * (len(hits) - 1)), 4)
        category = _category_for_literals(high) if high else high_families[0][1]
        return score, category, hits

    if mid or mid_families:
        hits = mid + [f[0] for f in mid_families]
        score = round(min(MID_CAP, MID_BASE + MID_STEP * len(hits)), 4)
        category = mid_families[0][1] if mid_families and not mid else "bullying"
        return score, category, hits

    if _hits(views, GAMING_BENIGN):
        return GAMING_SCORE, "none", []

    return NEUTRAL_SCORE, "none", []
