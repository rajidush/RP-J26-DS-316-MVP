"""Labelled development set for the Stage-1/Stage-2 text path.

What this is
------------
A hand-built **development** set, not a benchmark. Its job is to catch the
failure modes that matter for a child-safety tool and to make regressions
visible. Milestone A3/A4 replaces the headline numbers with HateXplain /
Jigsaw / CONDA; do not quote these figures as external validation.

Why the buckets exist
---------------------
A single accuracy number hides the interesting part. Grouping cases by *why*
they are hard shows where the cascade actually breaks:

  explicit          slurs and direct abuse — the easy band, lexicon should win
  implicit          hate with no keyword (dehumanisation, exclusion) — needs a model
  threat            intent to harm, including conditional/deferred
  bullying          personal degradation without protected-class targeting
  sexual            harassment / coercion aimed at a minor
  obfuscated        leetspeak, spacing, repetition — evasion of the lexicon
  gaming            competitive trash talk that must NOT trip (the main FP source)
  reporting         quoting/condemning hate — a child telling an adult MUST NOT trip
  figurative        violent idiom with no target ("this homework is killing me")
  benign            ordinary chat

`reporting` and `figurative` are weighted heavily in review because a false
positive there teaches a child that asking for help triggers surveillance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextCase:
    text: str
    hate: bool
    category: str  # expected category when hate, else "none"
    bucket: str


# --- Harmful -----------------------------------------------------------------

_EXPLICIT: List[TextCase] = [
    TextCase("you should kys", True, "threat", "explicit"),
    TextCase("kill yourself already loser", True, "threat", "explicit"),
    TextCase("go die in a hole", True, "threat", "explicit"),
    TextCase("you're a retard and everyone knows it", True, "hate_identity", "explicit"),
    TextCase("shut up faggot", True, "hate_identity", "explicit"),
    TextCase("nobody likes you go back to your country", True, "bullying", "explicit"),
    TextCase("hang yourself you waste of space", True, "threat", "explicit"),
]

_IMPLICIT: List[TextCase] = [
    TextCase("all of them are animals and should be caged", True, "hate_identity", "implicit"),
    TextCase("people like you shouldn't be allowed in this country", True, "hate_identity", "implicit"),
    TextCase("their kind always ruins everything they touch", True, "hate_identity", "implicit"),
    TextCase("we'd all be better off without your sort around here", True, "hate_identity", "implicit"),
    TextCase("they breed like vermin and take our jobs", True, "hate_identity", "implicit"),
    TextCase("everyone would be happier if you were gone", True, "bullying", "implicit"),
    TextCase("nobody wants you here go away", True, "bullying", "implicit"),
    TextCase("why do you even bother showing up", True, "bullying", "implicit"),
]

_THREAT: List[TextCase] = [
    TextCase("i'm going to hurt you after school", True, "threat", "threat"),
    TextCase("i'll find you and make you regret it", True, "threat", "threat"),
    TextCase("watch your back tomorrow", True, "threat", "threat"),
    TextCase("me and my friends are gonna jump you", True, "threat", "threat"),
    TextCase("i know where you live and i'm coming", True, "threat", "threat"),
]

_BULLYING: List[TextCase] = [
    TextCase("you're worthless and everyone knows it", True, "bullying", "bullying"),
    TextCase("stop being such a loser nobody likes you", True, "bullying", "bullying"),
    TextCase("everyone in class laughs at you behind your back", True, "bullying", "bullying"),
    TextCase("you're so ugly it hurts to look at you", True, "bullying", "bullying"),
    TextCase("no wonder you have no friends", True, "bullying", "bullying"),
]

_SEXUAL: List[TextCase] = [
    TextCase("send me pics or i'll tell everyone", True, "sexual_harassment", "sexual"),
    TextCase("take your clothes off for the camera", True, "sexual_harassment", "sexual"),
    TextCase("don't tell your parents this is our secret", True, "sexual_harassment", "sexual"),
]

_OBFUSCATED: List[TextCase] = [
    TextCase("you should k y s", True, "threat", "obfuscated"),
    TextCase("kiiiill yourself", True, "threat", "obfuscated"),
    TextCase("g0 d1e already", True, "threat", "obfuscated"),
    TextCase("ky5 loser", True, "threat", "obfuscated"),
    TextCase("n o b o d y likes you", True, "bullying", "obfuscated"),
]

# --- Benign — the cases that must NOT trip -----------------------------------

_GAMING: List[TextCase] = [
    TextCase("gg ez noob", False, "none", "gaming"),
    TextCase("you're trash at this game lol", False, "none", "gaming"),
    TextCase("im gonna kill you in this match", False, "none", "gaming"),
    TextCase("that boss destroyed me lmao", False, "none", "gaming"),
    TextCase("my team got rekt but it was fun", False, "none", "gaming"),
    TextCase("ez clap get good", False, "none", "gaming"),
    TextCase("i'm gonna snipe you from the tower", False, "none", "gaming"),
    TextCase("dude you're actually a bot at this", False, "none", "gaming"),
    TextCase("rip my whole squad wiped", False, "none", "gaming"),
    TextCase("that was a sick headshot", False, "none", "gaming"),
]

_REPORTING: List[TextCase] = [
    TextCase("you should never tell someone to kys thats awful", False, "none", "reporting"),
    TextCase("someone in the group chat told me to kill myself and i'm scared", False, "none", "reporting"),
    TextCase("miss he keeps calling me a retard what do i do", False, "none", "reporting"),
    TextCase("the article discusses racism in schools", False, "none", "reporting"),
    TextCase("we learned about hate speech in class today", False, "none", "reporting"),
    TextCase("is it bullying if someone says nobody likes you", False, "none", "reporting"),
    TextCase("mum someone wrote go back to your country on my locker", False, "none", "reporting"),
]

_FIGURATIVE: List[TextCase] = [
    TextCase("this homework is killing me", False, "none", "figurative"),
    TextCase("i hate mondays so much", False, "none", "figurative"),
    TextCase("i hate all vegetables honestly", False, "none", "figurative"),
    TextCase("my feet are killing me after that walk", False, "none", "figurative"),
    TextCase("i'd die for a slice of pizza right now", False, "none", "figurative"),
    TextCase("that joke murdered me lmao", False, "none", "figurative"),
    TextCase("i'm dying of laughter", False, "none", "figurative"),
]

_BENIGN: List[TextCase] = [
    TextCase("can you help me with my science project", False, "none", "benign"),
    TextCase("great game everyone well played", False, "none", "benign"),
    TextCase("what time is football practice tomorrow", False, "none", "benign"),
    TextCase("i finished the maths worksheet already", False, "none", "benign"),
    TextCase("do you want to come over after school", False, "none", "benign"),
    TextCase("the new update added a whole new map", False, "none", "benign"),
    TextCase("happy birthday hope you have a great day", False, "none", "benign"),
    TextCase("my mum said i can play until eight", False, "none", "benign"),
    TextCase("this song is actually so good", False, "none", "benign"),
    TextCase("i got a b plus on the history test", False, "none", "benign"),
]


def all_cases() -> List[TextCase]:
    return [
        *_EXPLICIT,
        *_IMPLICIT,
        *_THREAT,
        *_BULLYING,
        *_SEXUAL,
        *_OBFUSCATED,
        *_GAMING,
        *_REPORTING,
        *_FIGURATIVE,
        *_BENIGN,
    ]


def buckets() -> List[str]:
    seen: List[str] = []
    for case in all_cases():
        if case.bucket not in seen:
            seen.append(case.bucket)
    return seen


def summary() -> dict:
    cases = all_cases()
    return {
        "total": len(cases),
        "hate": sum(1 for c in cases if c.hate),
        "safe": sum(1 for c in cases if not c.hate),
        "buckets": {b: sum(1 for c in cases if c.bucket == b) for b in buckets()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=2))
