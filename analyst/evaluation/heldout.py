"""Held-out generalisation set — written AFTER the detectors, never tuned on.

Why it exists
-------------
The development set in `dataset.py` was used to design the lexicon pattern
families, so the cascade reaching 100% on it proves only that the patterns
match the cases they were written from. That number is not evidence of
generalisation and must never be quoted on its own.

This set expresses the same underlying harms in **different words**: synonyms
the patterns do not contain ("waste of oxygen" not "waste of space", "your
type" not "your kind", "disease" not "vermin"), and benign cases built from the
same vocabulary the detectors key on.

Rules for this file
-------------------
1. Cases are written before running them, from the *concept*, not from failures.
2. When the cascade fails here, the fix goes in the detector only if the miss
   is a genuine class of harm — never by adding the literal string to a pattern.
3. If a case is ever used to tune a rule, it moves to `dataset.py` and a
   replacement is written here.

`unalive yourself` is included deliberately: platform-evasion slang that the
lexicon does not know. It is expected to fail, and it documents a real gap
rather than hiding one.
"""

from __future__ import annotations

from typing import List

from analyst.evaluation.dataset import TextCase

_HARMFUL: List[TextCase] = [
    # bullying — degradation and exclusion, reworded
    TextCase("youre nothing but a waste of oxygen", True, "bullying", "bullying"),
    TextCase("nobody in this school can stand you", True, "bullying", "bullying"),
    TextCase("everyone would celebrate if you left this school", True, "bullying", "bullying"),
    TextCase("your whole family should be ashamed you exist", True, "bullying", "bullying"),
    TextCase("do everyone a favour and disappear", True, "bullying", "bullying"),
    # threats — no "gonna hurt you" phrasing
    TextCase("im waiting outside for you after class", True, "threat", "threat"),
    TextCase("youll regret ever talking to me", True, "threat", "threat"),
    TextCase("ill make sure you never come back here", True, "threat", "threat"),
    TextCase("bring your friends youll need them", True, "threat", "threat"),
    # identity — different dehumanisation vocabulary
    TextCase("those people are a disease on this country", True, "hate_identity", "implicit"),
    TextCase("we dont want your type living next to us", True, "hate_identity", "implicit"),
    TextCase("the world would be cleaner without them", True, "hate_identity", "implicit"),
    TextCase("get out of our neighbourhood you dont belong here", True, "hate_identity", "implicit"),
    TextCase("they should all be shipped back where they came from", True, "hate_identity", "implicit"),
    # sexual / grooming — reworded
    TextCase("send a photo of yourself without a shirt", True, "sexual_harassment", "sexual"),
    TextCase("keep this between us dont mention it to mum", True, "sexual_harassment", "sexual"),
    # obfuscation the normaliser should still crack
    TextCase("d1e already", True, "threat", "obfuscated"),
    TextCase("k*y*s", True, "threat", "obfuscated"),
    TextCase("g o  d i e", True, "threat", "obfuscated"),
    # known gap, documented not hidden: platform-evasion slang
    TextCase("unalive yourself", True, "threat", "obfuscated"),
]

_BENIGN: List[TextCase] = [
    # gaming — built from the same violent vocabulary the detectors key on
    TextCase("that play was insane you destroyed them", False, "none", "gaming"),
    TextCase("our team wiped the enemy base", False, "none", "gaming"),
    TextCase("this level is impossible im gonna lose it", False, "none", "gaming"),
    TextCase("youre killing it in this game", False, "none", "gaming"),
    TextCase("i sniped him from across the map", False, "none", "gaming"),
    TextCase("we got absolutely destroyed that round", False, "none", "gaming"),
    # figurative violence
    TextCase("im dead that was so funny", False, "none", "figurative"),
    TextCase("i got destroyed in that exam", False, "none", "figurative"),
    TextCase("my brother keeps annoying me i hate it", False, "none", "figurative"),
    TextCase("this queue is murdering my patience", False, "none", "figurative"),
    # reporting / help-seeking, reworded
    TextCase("she said something really mean should i tell a teacher", False, "none", "reporting"),
    TextCase("he called me a name today and i felt bad", False, "none", "reporting"),
    TextCase("theyre posting horrible things about me what do i do", False, "none", "reporting"),
    # educational
    TextCase("were reading a book about the holocaust for history", False, "none", "reporting"),
    TextCase("we did a project on discrimination last term", False, "none", "reporting"),
    # ordinary chat
    TextCase("can i borrow your notes for biology", False, "none", "benign"),
    TextCase("the referee made a terrible call", False, "none", "benign"),
    TextCase("that outfit is actually fire", False, "none", "benign"),
    TextCase("are you coming to the match on saturday", False, "none", "benign"),
    TextCase("i finally beat my highscore", False, "none", "benign"),
]


def all_cases() -> List[TextCase]:
    return [*_HARMFUL, *_BENIGN]


def summary() -> dict:
    cases = all_cases()
    return {
        "total": len(cases),
        "hate": sum(1 for c in cases if c.hate),
        "safe": sum(1 for c in cases if not c.hate),
    }
