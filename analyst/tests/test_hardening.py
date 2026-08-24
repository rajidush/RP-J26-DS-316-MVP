"""Modality weights + OCR redaction hardening."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.decide import modality_weights
from analyst.pipeline import AnalystPipeline
from analyst.store.persist import redact_snippet


class ModalityTests(unittest.TestCase):
    def test_ocr_plus_asr_splits_audio(self):
        m = modality_weights(0.9, 0.1, "hello ocr", "hello asr")
        self.assertGreater(m.audio, 0.0)
        self.assertGreater(m.text, 0.0)
        self.assertAlmostEqual(m.text + m.image + m.audio, 1.0, places=2)

    def test_asr_only(self):
        m = modality_weights(0.8, 0.0, "", "kill yourself")
        self.assertEqual(m.text, 0.0)
        self.assertGreater(m.audio, 0.9)

    def test_pipeline_exposes_modalities_and_explanation(self):
        pipe = AnalystPipeline()
        r = pipe.analyze(child_age=10, overlay_text="you should kys")
        self.assertEqual(r.decision, "hate")
        self.assertTrue(r.modalities)
        self.assertTrue(r.explanation)
        self.assertEqual(r.protection_state, "threat")
        self.assertIsNotNone(r.stage2)
        self.assertGreaterEqual(r.stage2["fused"], 0.85)


class RedactTests(unittest.TestCase):
    def test_password_redacted(self):
        s = redact_snippet("[sudo] password for dilnuka: hunter2 and password: secret123")
        self.assertNotIn("hunter2", s)
        self.assertNotIn("secret123", s)
        self.assertIn("[redacted]", s)


if __name__ == "__main__":
    unittest.main()
