import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.buffer import TransientMediaBuffer
from analyst.pipeline import AnalystPipeline
from analyst.stage1.lexicon import score_text
from analyst.stage2.fusion import fuse


class LexiconTests(unittest.TestCase):
    def test_threat(self):
        score, cat, hits = score_text("dude just kys already")
        self.assertGreater(score, 0.85)
        self.assertEqual(cat, "threat")
        self.assertTrue(hits)

    def test_gaming_benign(self):
        score, cat, _ = score_text("gg ez noob")
        self.assertLess(score, 0.40)
        self.assertEqual(cat, "none")


class BufferTests(unittest.TestCase):
    def test_wipe_on_error(self):
        buf = TransientMediaBuffer()
        try:
            with buf.hold("t1", frame=b"secret"):
                raise RuntimeError("x")
        except RuntimeError:
            pass
        self.assertEqual(buf.occupied(), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipe = AnalystPipeline()

    def test_hate_emits_contract(self):
        r = self.pipe.analyze(child_age=10, overlay_text="you should kys")
        self.assertEqual(r.decision, "hate")
        self.assertIsNotNone(r.payload)
        self.assertEqual(r.payload.category, "threat")
        self.assertTrue(r.payload.child_safe_summary)
        self.assertEqual(r.envelope.topic, "hate.detected")
        self.assertTrue(r.media_deleted)

    def test_clean_stops_stage1(self):
        r = self.pipe.analyze(child_age=12, overlay_text="gg ez that was fun")
        self.assertEqual(r.decision, "not-hate")
        self.assertIn("stopped_at_stage1", r.notes)

    def test_fusion_meme_bump(self):
        self.assertGreater(fuse(0.5, 0.5), 0.6 * 0.5 + 0.4 * 0.5)


if __name__ == "__main__":
    unittest.main()
