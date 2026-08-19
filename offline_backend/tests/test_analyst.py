import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyst.asr import AsrEngine
from analyst.buffer import TransientMediaBuffer
from analyst.lexicon import score_text
from analyst.pipeline import AnalystPipeline, fuse


class LexiconTests(unittest.TestCase):
    def test_threat_phrase_crosses_gate(self):
        score, category = score_text("dude just kys already")
        self.assertGreater(score, 0.85)
        self.assertEqual(category, "threat")

    def test_gaming_slang_stays_below_gate(self):
        score, category = score_text("gg ez noob")
        self.assertLess(score, 0.40)
        self.assertEqual(category, "none")

    def test_clean_text_is_low(self):
        score, category = score_text("want to play minecraft later?")
        self.assertLess(score, 0.20)
        self.assertEqual(category, "none")


class BufferTests(unittest.TestCase):
    def test_hold_deletes_even_on_error(self):
        buf = TransientMediaBuffer(max_slots=2)
        try:
            with buf.hold("t1", frame=b"secret-frame", audio=b"wav"):
                self.assertEqual(buf.occupied(), 1)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(buf.occupied(), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipe = AnalystPipeline()

    def test_overlay_hate_without_models(self):
        result = self.pipe.analyze(
            child_age=10,
            overlay_text="you should kys",
        )
        self.assertTrue(result.media_deleted)
        self.assertEqual(result.decision, "hate")
        self.assertGreater(result.risk_score, 0.85)
        self.assertEqual(self.pipe.buffer.occupied(), 0)

    def test_clean_overlay_stops_at_stage1(self):
        result = self.pipe.analyze(
            child_age=12,
            overlay_text="that was a fun game, gg",
        )
        self.assertEqual(result.decision, "not-hate")
        self.assertFalse(result.escalated)
        self.assertIn("stopped_at_stage1", result.notes)

    def test_no_input_does_not_crash(self):
        result = self.pipe.analyze(child_age=8)
        self.assertEqual(result.decision, "not-hate")
        self.assertIn("no_media_or_text", result.notes)

    def test_audio_only_hate_via_asr(self):
        self.pipe.asr = AsrEngine(transcribe_fn=lambda _b: "you should kys")
        result = self.pipe.analyze(child_age=10, audio_bytes=b"RIFF....fake")
        self.assertEqual(result.decision, "hate")
        self.assertTrue(result.source["asr"])
        self.assertEqual(result.transcript, "you should kys")
        self.assertTrue(result.media_deleted)

    def test_audio_clean_gaming_via_asr(self):
        self.pipe.asr = AsrEngine(transcribe_fn=lambda _b: "gg ez noob")
        result = self.pipe.analyze(child_age=10, audio_bytes=b"RIFF....fake")
        self.assertEqual(result.decision, "not-hate")
        self.assertFalse(result.escalated)

    def test_fusion_meme_bump(self):
        bumped = fuse(0.50, 0.50)
        plain = 0.60 * 0.50 + 0.40 * 0.50
        self.assertGreater(bumped, plain)


if __name__ == "__main__":
    unittest.main()
