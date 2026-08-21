"""Step 2 — OCR must recover key phrases from demo_assets screenshots."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image

from analyst.extract.ocr import OcrEngine
from analyst.pipeline import AnalystPipeline

ASSETS = Path(__file__).resolve().parents[1] / "demo_assets"
EXPECTED = ASSETS / "expected.json"


@unittest.skipUnless(ASSETS.is_dir(), "demo_assets missing — run python -m analyst.demo_assets.generate")
class OcrDemoAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OcrEngine()
        cls.expect = {}
        if EXPECTED.is_file():
            cls.expect = json.loads(EXPECTED.read_text(encoding="utf-8"))

    def test_ocr_backend_available(self):
        self.assertNotEqual(
            self.engine.name,
            "none",
            "Install rapidocr-onnxruntime for Step 2 OCR demos",
        )

    def test_ocr_recovers_key_tokens(self):
        if self.engine.name == "none":
            self.skipTest("OCR not installed")
        pngs = sorted(ASSETS.glob("*.png"))
        self.assertTrue(pngs, "No PNGs in demo_assets")
        for path in pngs:
            text = self.engine.extract(Image.open(path)).lower()
            expected = self.expect.get(path.name, path.stem.replace("_", " "))
            # Require at least 2 distinctive tokens from the expected line
            if path.name not in self.expect:
                continue  # vision-only / extra assets are not OCR fixtures
            tokens = [t for t in expected.split() if len(t) > 2][:3]
            hits = sum(1 for t in tokens if t in text)
            self.assertGreaterEqual(
                hits,
                2,
                f"OCR weak on {path.name}: got {text!r}, expected tokens {tokens}",
            )

    def test_pipeline_hate_from_image_ocr(self):
        if self.engine.name == "none":
            self.skipTest("OCR not installed")
        hate = ASSETS / "02_hate_threat.png"
        if not hate.is_file():
            self.skipTest("02_hate_threat.png missing")
        pipe = AnalystPipeline()
        result = pipe.analyze(child_age=10, image_bytes=hate.read_bytes())
        self.assertTrue(result.ocr_text, "OCR returned empty on hate screenshot")
        self.assertEqual(result.decision, "hate")
        self.assertTrue(result.media_deleted)

    def test_pipeline_clean_from_image_ocr(self):
        if self.engine.name == "none":
            self.skipTest("OCR not installed")
        clean = ASSETS / "01_clean_gaming.png"
        if not clean.is_file():
            self.skipTest("01_clean_gaming.png missing")
        pipe = AnalystPipeline()
        result = pipe.analyze(child_age=12, image_bytes=clean.read_bytes())
        self.assertTrue(result.ocr_text)
        self.assertEqual(result.decision, "not-hate")


if __name__ == "__main__":
    unittest.main()
