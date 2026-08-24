"""Tests for SQLite store + persist + injected capture worker."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.capture.worker import CaptureWorker
from analyst.pipeline import AnalystPipeline
from analyst.schemas import AnalystRunResult
from analyst.store.db import AnalystStore
from analyst.store.persist import make_blurred_thumb, persist_result


def _png_bytes(color=(40, 40, 40), size=(200, 120)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class StorePersistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "t.db"
        self.store = AnalystStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_list_roundtrip(self):
        fake = AnalystRunResult(
            decision="hate",
            ocr_text="you should kys",
            transcript="",
            stage1={"text_score": 0.88, "vision_score": 0.0},
            stage2={"fused": 0.88},
            backends={"ocr": "none"},
            notes=[],
            latency_ms={"total_ms": 12.0},
        )
        # Minimal payload via pipeline for realism
        pipe = AnalystPipeline()
        result = pipe.analyze(child_age=10, overlay_text="you should kys")
        run_id = persist_result(
            self.store,
            result,
            child_age=10,
            app_exe="test",
            frame_bytes=_png_bytes(),
        )
        self.assertTrue(run_id)
        latest = self.store.latest_run()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["decision"], "hate")
        self.assertIn("kys", (latest["ocr_snippet"] or "") + str(latest.get("lexicon_hits")))
        self.assertTrue(latest.get("thumb_jpeg"))
        self.assertEqual(self.store.stats()["hate"], 1)
        got = self.store.get_run(run_id)
        self.assertEqual(got["id"], run_id)
        self.assertIsNotNone(fake.decision)

    def test_blurred_thumb_small(self):
        thumb = make_blurred_thumb(_png_bytes(size=(800, 600)))
        self.assertIsNotNone(thumb)
        self.assertLessEqual(len(thumb), 48_000)


class InjectedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AnalystStore(Path(self.tmp.name) / "c.db")
        self.frame = _png_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_tick_once_persists(self):
        pipe = AnalystPipeline()
        # Inject text via overlay by wrapping analyze — use frame that OCR may empty;
        # force path: grab_frame returns demo hate image bytes from generate if present,
        # else use overlay by customizing pipeline call through a thin subclass.
        worker = CaptureWorker(
            store=self.store,
            pipeline=pipe,
            grab_frame=lambda: (self.frame, 200, 120),
            grab_audio=lambda: (None, 0.0),
        )

        # Directly analyze+persist with known text to assert DB without OCR flakiness
        result = pipe.analyze(child_age=10, overlay_text="gg ez noob", image_bytes=self.frame)
        persist_result(self.store, result, child_age=10, frame_bytes=self.frame)
        self.assertEqual(self.store.latest_run()["decision"], "not-hate")

        # tick_once with injected frame still completes
        run_id = worker.tick_once()
        self.assertTrue(run_id)
        self.assertGreaterEqual(self.store.stats()["total"], 2)


if __name__ == "__main__":
    unittest.main()
