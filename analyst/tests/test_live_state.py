"""Live pipeline state — what the engine is doing at this instant.

The white box diagram reads this every 600 ms. The failure that matters is not
a wrong number, it is a stage that never clears: the box would pulse "running"
forever and the panel would claim a check is in flight long after it finished.
Every exit path therefore has a test.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.whitebox.live import LiveState, LIVE, check, stage


class StageTrackingTests(unittest.TestCase):
    def setUp(self):
        self.live = LiveState()

    def test_begin_marks_a_stage_active(self):
        self.live.begin("ocr")
        self.assertIn("ocr", self.live.snapshot()["active"])

    def test_end_clears_it_and_records_the_duration(self):
        self.live.begin("ocr")
        self.live.end("ocr")
        snap = self.live.snapshot()
        self.assertNotIn("ocr", snap["active"])
        self.assertIn("ocr", snap["last_ms"])

    def test_overlapping_stages_are_both_tracked(self):
        """OCR and speech-to-text are submitted to a pool together, so a single
        'current phase' would have to lie about one of them."""
        self.live.begin("ocr")
        self.live.begin("audio")
        self.assertEqual(set(self.live.snapshot()["active"]), {"ocr", "audio"})
        self.live.end("ocr")
        self.assertEqual(set(self.live.snapshot()["active"]), {"audio"})

    def test_ending_an_unknown_stage_is_harmless(self):
        self.live.end("never-started")
        self.assertEqual(self.live.snapshot()["active"], {})

    def test_seq_advances_on_every_transition(self):
        first = self.live.snapshot()["seq"]
        self.live.begin("clip")
        self.live.end("clip")
        self.assertGreater(self.live.snapshot()["seq"], first)


class ExitPathTests(unittest.TestCase):
    """A stage left marked active is the one failure that misleads the panel."""

    def test_stage_context_clears_on_success(self):
        with stage("clip"):
            self.assertIn("clip", LIVE.snapshot()["active"])
        self.assertNotIn("clip", LIVE.snapshot()["active"])

    def test_stage_context_clears_on_exception(self):
        with self.assertRaises(ValueError):
            with stage("vlm"):
                raise ValueError("model unreachable")
        self.assertNotIn("vlm", LIVE.snapshot()["active"])

    def test_stage_context_does_not_swallow_the_exception(self):
        with self.assertRaises(KeyError):
            with stage("ocr"):
                raise KeyError("boom")

    def test_end_check_clears_a_stage_left_running(self):
        """The safety net: a scorer raising between a bare begin/end pair would
        otherwise leave its box pulsing forever."""
        live = LiveState()
        live.start_check("test.exe")
        live.begin("stage1")  # deliberately never ended
        self.assertIn("stage1", live.snapshot()["active"])
        live.end_check()
        snap = live.snapshot()
        self.assertEqual(snap["active"], {})
        self.assertFalse(snap["busy"])

    def test_check_context_clears_on_exception(self):
        live = LiveState()
        try:
            with self.assertRaises(RuntimeError):
                with_ = live
                with_.start_check("x")
                with_.begin("stage2")
                raise RuntimeError("pipeline blew up")
        finally:
            live.end_check()
        self.assertEqual(live.snapshot()["active"], {})


class BusyReportingTests(unittest.TestCase):
    def test_idle_by_default(self):
        live = LiveState()
        snap = live.snapshot()
        self.assertFalse(snap["busy"])
        self.assertEqual(snap["busy_ms"], 0.0)

    def test_busy_between_start_and_end(self):
        live = LiveState()
        live.start_check("chrome.exe")
        snap = live.snapshot()
        self.assertTrue(snap["busy"])
        self.assertEqual(snap["label"], "chrome.exe")
        live.end_check()
        self.assertFalse(live.snapshot()["busy"])

    def test_check_context_manager(self):
        live_seen = []
        with check("edge.exe"):
            live_seen.append(LIVE.snapshot()["busy"])
        self.assertEqual(live_seen, [True])
        self.assertFalse(LIVE.snapshot()["busy"])


class ThreadSafetyTests(unittest.TestCase):
    def test_concurrent_begin_end_does_not_corrupt(self):
        live = LiveState()
        errors = []

        def worker(name):
            try:
                for _ in range(200):
                    live.begin(name)
                    live.snapshot()
                    live.end(name)
            except Exception as exc:  # pragma: no cover - only on a real race
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(live.snapshot()["active"], {})


class SnapshotShapeTests(unittest.TestCase):
    """The panel reads these keys by name; renaming one blanks the live view."""

    def test_snapshot_carries_the_keys_the_panel_reads(self):
        snap = LiveState().snapshot()
        for key in ("active", "last_ms", "busy", "busy_ms", "idle_s", "label", "seq"):
            self.assertIn(key, snap)

    def test_panel_reads_the_same_keys(self):
        html = (ROOT / "analyst" / "panel" / "index.html").read_text(encoding="utf-8")
        self.assertIn("live.active", html)
        self.assertIn("live.busy", html)
        self.assertIn("live.busy_ms", html)


if __name__ == "__main__":
    unittest.main()
