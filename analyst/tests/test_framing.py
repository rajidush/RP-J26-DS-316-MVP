"""Framing guard — a child reporting abuse must not be treated as committing it."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.stage1 import framing
from analyst.stage1.lexicon import score_text


class DetectionTests(unittest.TestCase):
    def test_third_party_attribution_is_reporting(self):
        for text in (
            "someone in the group chat told me to kill myself and i'm scared",
            "miss he keeps calling me a retard what do i do",
            "mum someone wrote go back to your country on my locker",
            "this kid said i should go die",
        ):
            with self.subTest(text=text):
                self.assertTrue(framing.detect(text).reporting, text)

    def test_condemnation_is_reporting(self):
        for text in (
            "you should never tell someone to kys thats awful",
            "don't call people that it's not ok",
            "is it bullying if someone says nobody likes you",
        ):
            with self.subTest(text=text):
                self.assertTrue(framing.detect(text).reporting, text)

    def test_educational_context_is_reporting(self):
        for text in (
            "the article discusses racism in schools",
            "we learned about hate speech in class today",
        ):
            with self.subTest(text=text):
                self.assertTrue(framing.detect(text).reporting, text)

    def test_direct_abuse_is_not_reporting(self):
        """The guard must not fire on first-person abuse — that would be an
        evasion channel, not a safety feature."""
        for text in (
            "you should kys",
            "nobody likes you go back to your country",
            "i'm going to hurt you after school",
            "all of them are animals and should be caged",
            "shut up faggot",
        ):
            with self.subTest(text=text):
                self.assertFalse(framing.detect(text).reporting, text)

    def test_ordinary_chat_is_not_reporting(self):
        for text in ("gg ez noob", "can you help me with my science project"):
            with self.subTest(text=text):
                self.assertFalse(framing.detect(text).reporting, text)


class ApplyTests(unittest.TestCase):
    def test_reported_hate_is_capped_below_every_persona_threshold(self):
        text = "miss he keeps calling me a retard what do i do"
        raw, _cat, _hits = score_text(text)
        self.assertGreater(raw, 0.80)  # the lexicon does fire on the slur
        capped, mark = framing.apply(raw, text)
        self.assertLessEqual(capped, framing.FRAMING_CAP)
        self.assertLess(capped, 0.55)  # below the most protective persona
        self.assertTrue(mark.reporting)

    def test_discount_is_recorded_for_audit(self):
        _score, mark = framing.apply(0.9, "he called me a retard what do i do")
        self.assertTrue(mark.reason.startswith("quoted_or_reported:"))
        self.assertIn("attribution", mark.signals)

    def test_direct_abuse_score_is_untouched(self):
        raw, _c, _h = score_text("you should kys")
        capped, mark = framing.apply(raw, "you should kys")
        self.assertEqual(capped, raw)
        self.assertFalse(mark.reporting)

    def test_already_low_score_is_not_raised(self):
        capped, _ = framing.apply(0.08, "we learned about hate speech in class today")
        self.assertEqual(capped, 0.08)


class GamingGuardTests(unittest.TestCase):
    """Competitive banter must stop at stage 1, not merely avoid alerting."""

    def test_gaming_cap_sits_below_the_stage1_gate(self):
        from analyst.decide import STAGE1_THETA
        from analyst.stage1.text_fast import GAMING_MODEL_CAP

        # A cap above theta1 lets every gaming frame escalate into Stage 2,
        # which silently removes the cascade's cost saving.
        self.assertLess(GAMING_MODEL_CAP, STAGE1_THETA)

    def test_gaming_context_is_detected(self):
        from analyst.stage1.lexicon import in_gaming_context

        for text in ("gg ez noob", "im gonna kill you in this match", "rip my whole squad wiped"):
            with self.subTest(text=text):
                self.assertTrue(in_gaming_context(text), text)

    def test_non_gaming_abuse_is_not_gaming_context(self):
        from analyst.stage1.lexicon import in_gaming_context

        for text in ("im going to hurt you after school", "you should kys"):
            with self.subTest(text=text):
                self.assertFalse(in_gaming_context(text), text)


class LexiconRegressionTests(unittest.TestCase):
    def test_hate_all_no_longer_matches_a_food_opinion(self):
        score, category, _hits = score_text("i hate all vegetables honestly")
        self.assertLess(score, 0.55)
        self.assertEqual(category, "none")

    def test_hate_directed_at_people_still_matches(self):
        score, _cat, hits = score_text("i hate all of you")
        self.assertGreaterEqual(score, 0.55)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
