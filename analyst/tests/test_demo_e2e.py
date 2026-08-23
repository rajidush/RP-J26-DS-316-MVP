"""Step 5 — fusion math + demo runner smoke (text-only, no heavy assets)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.demo_e2e import Case, _run_case, _table
from analyst.pipeline import AnalystPipeline
from analyst.stage2.fusion import TEXT_WEIGHT, VISION_WEIGHT, fuse, fusion_detail


class FusionMathTests(unittest.TestCase):
    def test_weighted_average(self):
        # Vision inactive must not dilute text
        self.assertAlmostEqual(fuse(1.0, 0.0), 1.0, places=4)
        self.assertAlmostEqual(fuse(0.0, 1.0), 1.0, places=4)
        both = fuse(0.8, 0.8)
        self.assertGreater(both, 0.0)

    def test_meme_bump_when_both_mid(self):
        bumped = fuse(0.5, 0.5)
        base = TEXT_WEIGHT * 0.5 + VISION_WEIGHT * 0.5
        self.assertGreater(bumped, base)
        self.assertLessEqual(bumped, 1.0)
        detail = fusion_detail(0.5, 0.5)
        self.assertTrue(detail["meme_bump"])
        self.assertEqual(detail["fused"], bumped)

    def test_no_bump_when_one_low(self):
        # Strong text + mild vision keeps text floor
        self.assertAlmostEqual(fuse(0.9, 0.1), 0.9, places=4)
        self.assertFalse(fusion_detail(0.9, 0.1)["meme_bump"])

    def test_strong_text_not_diluted(self):
        self.assertGreaterEqual(fuse(0.88, 0.2), 0.88)


class DemoRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = AnalystPipeline()

    def test_text_cases_match_expectation(self):
        clean = _run_case(
            self.pipe,
            Case("text_clean", "not-hate", overlay="gg ez noob"),
            age=12,
        )
        hate = _run_case(
            self.pipe,
            Case("text_hate", "hate", overlay="you should kys"),
            age=10,
        )
        self.assertEqual(clean.match, "ok")
        self.assertEqual(hate.match, "ok")
        self.assertIsNotNone(hate.envelope)
        self.assertEqual(hate.envelope["topic"], "hate.detected")
        self.assertIn("child_safe_summary", hate.envelope["payload"])
        self.assertGreater(hate.total_ms, 0.0)
        table = _table([clean, hate])
        self.assertIn("text_hate", table)

    def test_missing_asset_skips(self):
        row = _run_case(
            self.pipe,
            Case(
                "missing_wav",
                "hate",
                audio=Path("does_not_exist.wav"),
                skip_if_missing=True,
            ),
            age=10,
        )
        self.assertEqual(row.match, "skip")
        self.assertIn("missing", row.skipped)


if __name__ == "__main__":
    unittest.main()
