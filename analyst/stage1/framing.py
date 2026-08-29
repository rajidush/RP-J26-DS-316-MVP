"""Is this text *being* hate, or *about* hate?

Why this exists
---------------
The scorers match harmful language wherever it appears, which flags a child who
is quoting the abuse rather than committing it:

    "someone in the group chat told me to kill myself and i'm scared"  -> 0.88
    "miss he keeps calling me a retard what do i do"                   -> 0.88
    "mum someone wrote go back to your country on my locker"           -> 0.58

Every one of those is a child asking an adult for help. Interrupting them with
an intervention is the worst possible response: it teaches that reporting abuse
triggers the same machinery as committing it, which is how a safety tool trains
children to stop reporting.

Design
------
This is a **discount, not a veto**. Three signal families are counted:

  attribution   the harm is credited to someone else ("he called me", "they said")
  condemnation  the text argues against the harm ("you should never say that")
  meta          the text discusses harm as a topic ("we learned about hate speech")

With at least one signal the score is capped just below the lowest persona
threshold (0.55) rather than zeroed, so the run still appears in telemetry as a
near-miss and a parent can review it. The cap is deliberately not a full
suppression: this must not become a reliable evasion channel.

Limits, stated plainly
----------------------
Pattern matching cannot read intent. "i'm not saying kys but you should quit"
carries a condemnation shape and will be discounted. The trade is accepted
because on a child's screen, help-seeking is far more common than that
construction — but it is a known hole, not an oversight, and a learned
framing classifier is the Milestone A3 replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# Ceiling applied when framing signals fire. Sits just under the most
# protective persona threshold (0.55) so a discounted run cannot alert for any
# age, while still recording a non-trivial score for review.
FRAMING_CAP = 0.50

# Harm attributed to a third party — the child is the target or a witness.
_ATTRIBUTION = (
    r"\b(?:someone|somebody|some\s?one|this\s+(?:kid|boy|girl|guy)|a\s+(?:kid|boy|girl|classmate)"
    r"|he|she|they|them|people|kids|classmates?)\s+"
    r"(?:just\s+|keeps?\s+|kept\s+|always\s+)?"
    r"(?:said|says|told|telling|wrote|writes|called|calling|posted|sent|sends|typed|commented)\b",
    r"\bcalled\s+me\b",
    r"\bwrote\s+.{0,30}\bon\s+my\b",
    r"\bin\s+the\s+(?:group\s+)?chat\b.{0,40}\b(?:said|told|wrote)\b",
    r"\bgot\s+(?:called|told)\b",
)

# The text argues against the harm rather than delivering it.
_CONDEMNATION = (
    # "don't say that" is condemnation; "don't tell your parents" is grooming.
    # The lookahead keeps the second one out — without it, the single most
    # dangerous phrase in the set was being discounted as if it were advice.
    r"\b(?:never|don'?t|do\s+not|shouldn'?t|should\s+not|stop)\s+"
    r"(?:say|saying|tell|telling|call|calling|type|post)\b"
    r"(?!\s+(?:your|ur|yr)\s+(?:parents|mum|mom|dad|family|teacher|anyone|anybody))",
    r"\bthat'?s\s+(?:awful|horrible|terrible|wrong|mean|not\s+ok(?:ay)?|unacceptable|so\s+rude)\b",
    r"\b(?:is|that'?s)\s+(?:bullying|racist|hate\s+speech|harassment)\b",
    r"\breport(?:ed|ing)?\s+(?:it|them|him|her|this)\b",
)

# Harm as a subject of study or discussion.
_META = (
    r"\b(?:article|lesson|assembly|class|homework|essay|project|documentary|news)\b"
    r".{0,30}\b(?:about|on|discusses?|covers?)\b",
    r"\bwe\s+(?:learned|learnt|talked|were\s+talking)\s+about\b",
    r"\bis\s+it\s+(?:bullying|racist|hate|harassment)\b",
    r"\bwhat\s+(?:do|should)\s+i\s+do\b",
    r"\bi'?m\s+(?:scared|worried|upset|frightened)\b",
)

_COMPILED: Tuple[Tuple[str, Tuple[re.Pattern, ...]], ...] = (
    ("attribution", tuple(re.compile(p, re.IGNORECASE) for p in _ATTRIBUTION)),
    ("condemnation", tuple(re.compile(p, re.IGNORECASE) for p in _CONDEMNATION)),
    ("meta", tuple(re.compile(p, re.IGNORECASE) for p in _META)),
)


@dataclass
class Framing:
    reporting: bool = False
    signals: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.reporting:
            return ""
        return "quoted_or_reported:" + "+".join(sorted(set(self.signals)))


def detect(text: str) -> Framing:
    """Which framing families appear in this text."""
    blob = (text or "").strip()
    if not blob:
        return Framing()
    signals: List[str] = []
    for family, patterns in _COMPILED:
        if any(p.search(blob) for p in patterns):
            signals.append(family)
    return Framing(reporting=bool(signals), signals=signals)


# Grooming is built to look like confiding — secrecy, trust, "don't tell".
# Those are the same surface features that mark a child reporting abuse, so the
# discount can never be allowed to apply here. If the two are indistinguishable
# by shape, the safe error is to alert.
NEVER_DISCOUNTED = ("sexual_harassment",)


def apply(score: float, text: str, category: str = "") -> Tuple[float, Framing]:
    """Cap a harmful score when the text is reporting rather than committing.

    Returns the (possibly reduced) score plus the framing that explains it, so
    the decision stays auditable — a discount must never be invisible.
    """
    framing = detect(text)
    if not framing.reporting:
        return score, framing
    if category in NEVER_DISCOUNTED:
        return score, Framing(reporting=False, signals=[])
    return (min(score, FRAMING_CAP), framing) if score > FRAMING_CAP else (score, framing)
