"""Fusion safety invariants.

These are the rules that stop an uninformative branch from changing a safety
decision. They are properties, not examples, so they are tested over a grid.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.decide import PERSONA_THETA2
from analyst.stage2.fusion import Signal, TextFull, fuse, fuse_signals

GRID = [i / 20 for i in range(21)]  # 0.00 .. 1.00


class FloorInvariantTests(unittest.TestCase):
    """Rule 2: fusion may raise confidence, never lower it."""

    def test_fused_never_below_strongest_calibrated_modality(self):
        for t in GRID:
            for v in GRID:
                with self.subTest(text=t, vision=v):
                    fused = fuse(t, v, vision_calibrated=True)
                    self.assertGreaterEqual(
                        fused + 1e-9,
                        max(t, v),
                        f"fuse({t}, {v}) = {fused} fell below max({t}, {v})",
                    )

    def test_uncalibrated_vision_cannot_change_the_score(self):
        for t in GRID:
            for v in GRID:
                with self.subTest(text=t, vision=v):
                    self.assertAlmostEqual(fuse(t, v, vision_calibrated=False), round(t, 4), places=4)


class UncalibratedVisionRegressionTests(unittest.TestCase):
    """The bug this design exists to prevent.

    CLIP zero-shot returned 0.324-0.393 for every demo asset. Averaged in, it
    pulled a confirmed 0.88 text detection down to 0.685 — cleared for ages
    14-15 — while 0.330 left it at 0.880. Safety behaviour must not depend on
    which side of a guard the noise lands.
    """

    NOISE = (0.324, 0.330, 0.349, 0.359, 0.393)

    def test_confident_text_survives_every_observed_noise_value(self):
        for v in self.NOISE:
            with self.subTest(vision=v):
                self.assertEqual(fuse(0.88, v, vision_calibrated=False), 0.88)

    def test_confident_text_flags_at_every_persona_age(self):
        for v in self.NOISE:
            fused = fuse(0.88, v, vision_calibrated=False)
            for persona, theta in PERSONA_THETA2.items():
                with self.subTest(vision=v, persona=persona):
                    self.assertGreaterEqual(
                        fused, theta,
                        f"vision noise {v} cleared a 0.88 detection for {persona}",
                    )

    def test_uncalibrated_branch_is_reported_not_silently_dropped(self):
        result = fuse_signals([
            Signal("text", 0.88, True, "ocr"),
            Signal("vision", 0.393, False, "clip-zeroshot-uncalibrated"),
        ])
        self.assertEqual(result.fused, 0.88)
        self.assertIn("vision", result.ignored)
        self.assertNotIn("vision", result.contributing)


class AgreementTests(unittest.TestCase):
    def test_two_mid_signals_corroborate(self):
        """The hateful-meme case: neither modality is damning alone."""
        alone = 0.5
        fused = fuse(alone, alone, vision_calibrated=True)
        self.assertGreater(fused, alone)

    def test_agreement_needs_both_modalities_in_band(self):
        result = fuse_signals([
            Signal("text", 0.5, True, "ocr"),
            Signal("vision", 0.95, True, "probe"),
        ])
        self.assertFalse(result.agreement)
        self.assertGreaterEqual(result.fused, 0.95)  # floor still holds

    def test_single_modality_is_not_agreement(self):
        result = fuse_signals([Signal("text", 0.5, True, "ocr")])
        self.assertEqual(result.mode, "text_only")
        self.assertEqual(result.fused, 0.5)

    def test_no_signals_is_idle(self):
        result = fuse_signals([Signal("vision", 0.4, False, "uncal")])
        self.assertEqual(result.mode, "idle")
        self.assertEqual(result.fused, 0.0)


class TextFullTests(unittest.TestCase):
    """Stage 2 re-reads text Stage 1 truncated."""

    def test_short_text_is_passed_through_untouched(self):
        full = TextFull(scorer=lambda _c: 0.99)
        self.assertEqual(full.score("short text", 0.2), 0.2)

    def test_harm_after_the_stage1_window_is_found(self):
        # Stage 1 sees the first window only; the harm sits far past it.
        filler = "we played the new map and it was fun. " * 20
        blob = filler + "you should kys"
        full = TextFull(scorer=lambda chunk: 0.9 if "kys" in chunk else 0.1)
        self.assertGreaterEqual(full.score(blob, 0.1), 0.9)

    def test_unbound_stage2_is_a_passthrough(self):
        self.assertEqual(TextFull().score("x" * 900, 0.42), 0.42)

    def test_a_failing_scorer_cannot_take_down_stage2(self):
        def boom(_chunk):
            raise RuntimeError("model died")

        full = TextFull(scorer=boom)
        self.assertEqual(full.score("y" * 900, 0.33), 0.33)


if __name__ == "__main__":
    unittest.main()
