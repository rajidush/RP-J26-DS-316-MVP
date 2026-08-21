"""Step 3 — ASR must feed the same Stage-1 cascade as OCR/overlay text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.extract.asr import AsrEngine, _WHISPER_OK
from analyst.pipeline import AnalystPipeline

ASSETS = Path(__file__).resolve().parents[1] / "demo_assets"


@unittest.skipUnless(ASSETS.is_dir(), "demo_assets missing")
class AsrDemoTests(unittest.TestCase):
    def test_asr_backend_reports_whisper_or_none(self):
        eng = AsrEngine()
        self.assertIn(eng.name.split("-")[0], ("faster", "none", "injected", "whisper"))

    def test_injected_asr_drives_hate_decision(self):
        pipe = AnalystPipeline()
        pipe.asr = AsrEngine(transcribe_fn=lambda _b: "you should kys")
        result = pipe.analyze(child_age=10, audio_bytes=b"RIFF....fake")
        self.assertEqual(result.decision, "hate")
        self.assertEqual(result.transcript, "you should kys")
        self.assertTrue(result.media_deleted)

    def test_injected_asr_clean_gaming(self):
        pipe = AnalystPipeline()
        pipe.asr = AsrEngine(transcribe_fn=lambda _b: "gg ez noob")
        result = pipe.analyze(child_age=12, audio_bytes=b"RIFF....fake")
        self.assertEqual(result.decision, "not-hate")

    @unittest.skipUnless(_WHISPER_OK, "faster-whisper not installed")
    def test_whisper_on_hate_wav_if_present(self):
        wav = ASSETS / "02_hate_threat.wav"
        if not wav.is_file():
            self.skipTest("Run: python -m analyst.demo_assets.generate_audio")
        pipe = AnalystPipeline()
        result = pipe.analyze(child_age=10, audio_bytes=wav.read_bytes())
        self.assertTrue(result.transcript, f"empty transcript; backends={result.backends}")
        # Whisper wording varies; lexicon/threat path should still fire on kill yourself
        lowered = result.transcript.lower()
        self.assertTrue(
            any(t in lowered for t in ("kill", "yourself", "kys", "nobody", "likes")),
            f"unexpected transcript: {result.transcript!r}",
        )
        self.assertEqual(result.decision, "hate")
        self.assertTrue(result.media_deleted)


if __name__ == "__main__":
    unittest.main()
