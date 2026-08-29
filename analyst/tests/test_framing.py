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


class EnsembleCorroborationTests(unittest.TestCase):
    """Two heads combined with max() let either one's false positive through.

    On published corpora that was the dominant error source, so a lone,
    merely-confident head is damped unless the other agrees.
    """

    def _scores(self, a, b):
        """Drive TextFast with two stubbed heads and read the model reading."""
        from analyst.stage1.text_fast import TextFast

        tf = TextFast.__new__(TextFast)          # bypass model loading
        tf._override = None
        tf._use_framing = False
        tf._models = [object(), object()]        # len() == 2 is all that matters
        tf.name = "stub"
        tf._read_all = lambda _text: [(a, "bullying", {"toxic": a}), (b, None, {})]
        return tf._model_reading("some text")[0]

    def test_both_heads_agreeing_is_trusted(self):
        from analyst.stage1.text_fast import CORROBORATION_FLOOR

        self.assertEqual(self._scores(0.80, CORROBORATION_FLOOR + 0.01), 0.80)

    def test_one_very_confident_head_is_trusted_alone(self):
        from analyst.stage1.text_fast import SOLO_TRUST

        self.assertEqual(self._scores(SOLO_TRUST + 0.05, 0.01), SOLO_TRUST + 0.05)

    def test_lone_moderately_confident_head_is_damped(self):
        from analyst.stage1.text_fast import SOLO_DAMP

        got = self._scores(0.80, 0.05)
        self.assertAlmostEqual(got, round(0.80 * SOLO_DAMP, 4), places=4)
        self.assertLess(got, 0.80)

    def test_damping_can_drop_a_lone_head_below_the_persona_threshold(self):
        """The point of the rule: an uncorroborated 0.72 must not alert."""
        from analyst.decide import PERSONA_THETA2

        self.assertLess(self._scores(0.72, 0.10), PERSONA_THETA2["8-10"])

    def test_a_single_head_setup_is_never_damped(self):
        from analyst.stage1.text_fast import TextFast

        tf = TextFast.__new__(TextFast)
        tf._override = None
        tf._use_framing = False
        tf._models = [object()]
        tf.name = "stub"
        tf._read_all = lambda _t: [(0.72, "bullying", {})]
        self.assertEqual(tf._model_reading("x")[0], 0.72)

    def test_winning_head_labels_reach_the_score_detail(self):
        """The labels shown as evidence must come from the head that won.

        This field sat unpopulated for a while because nothing asserted on it.
        """
        from analyst.stage1.text_fast import TextFast

        tf = TextFast.__new__(TextFast)
        tf._override = None
        tf._use_framing = False
        tf._models = [object(), object()]
        tf.name = "stub"
        tf._read_all = lambda _t: [
            (0.95, "threat", {"threat": 0.95}),
            (0.10, None, {"toxic": 0.10}),
        ]
        detail = tf.score_detailed("no lexicon hit here at all")
        self.assertEqual(detail.model_labels, {"threat": 0.95})

    def test_constants_stay_ordered(self):
        from analyst.stage1.text_fast import (
            CORROBORATION_FLOOR, SOLO_DAMP, SOLO_TRUST,
        )

        self.assertLess(CORROBORATION_FLOOR, SOLO_TRUST)
        self.assertLess(SOLO_DAMP, 1.0)


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
