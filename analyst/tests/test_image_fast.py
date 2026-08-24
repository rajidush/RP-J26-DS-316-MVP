"""Step 4 — vision branch can escalate without OCR/ASR text."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.extract.embed import ImageEmbedder
from analyst.pipeline import AnalystPipeline
from analyst.stage1.image_fast import ImageFast


def _png_bytes(color=(200, 40, 40), size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ImageFastInjectedTests(unittest.TestCase):
    def test_injected_score_high(self):
        fast = ImageFast(score_fn=lambda **_: 0.88)
        self.assertEqual(fast.name, "injected")
        self.assertGreaterEqual(fast.score(embedding=[0.1]), 0.88)

    def test_deferred_without_clip(self):
        emb = ImageEmbedder(embed_fn=None)
        # Fresh embedder with no torch → deferred unless CLIP already loaded
        if emb.name == "deferred":
            fast = ImageFast(embedder=emb)
            self.assertEqual(fast.name, "deferred")
            self.assertEqual(fast.score(embedding=[]), 0.0)


class VisionOnlyPipelineTests(unittest.TestCase):
    def test_vision_only_hate_via_injection(self):
        pipe = AnalystPipeline()
        pipe.ocr.extract = lambda _img: ""  # type: ignore[method-assign]
        pipe.asr.transcribe = lambda _a: ""  # type: ignore[method-assign]
        pipe.embed = ImageEmbedder(embed_fn=lambda _img: [0.1] * 8)
        pipe.image_fast = ImageFast(
            score_fn=lambda **_: 0.92,
            embedder=pipe.embed,
        )
        result = pipe.analyze(child_age=10, image_bytes=_png_bytes())
        self.assertEqual(result.decision, "hate")
        self.assertIn("vision_only_escalation", result.notes)
        self.assertGreaterEqual(result.stage1["vision_score"], 0.9)
        self.assertTrue(result.media_deleted)
        self.assertEqual(result.payload.category, "hate_identity")

    def test_vision_low_stays_not_hate(self):
        pipe = AnalystPipeline()
        pipe.ocr.extract = lambda _img: ""  # type: ignore[method-assign]
        pipe.asr.transcribe = lambda _a: ""  # type: ignore[method-assign]
        pipe.image_fast = ImageFast(score_fn=lambda **_: 0.05)
        result = pipe.analyze(child_age=10, image_bytes=_png_bytes(color=(240, 240, 240)))
        self.assertEqual(result.decision, "not-hate")
        self.assertLess(result.stage1["vision_score"], 0.35)


class ClipOptionalTests(unittest.TestCase):
    def test_clip_loads_or_defers(self):
        emb = ImageEmbedder()
        prefix = emb.name.split(":")[0]
        self.assertIn(prefix, ("clip", "deferred", "injected", "clip_failed"))
        if prefix not in ("clip",):
            self.skipTest("torch+transformers CLIP not installed")
        img = Image.new("RGB", (224, 224), color=(30, 30, 30))
        vec = emb.embed(img)
        self.assertGreater(len(vec), 0)
        self.assertTrue(emb.name.startswith("clip:"))
        fast = ImageFast(embedder=emb)
        score = fast.score(embedding=vec, image=img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
