"""Whitebox trace builder tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.schemas import AnalystRunResult
from analyst.whitebox.trace import (
    TraceBuffer,
    build_trace_failed,
    build_trace_from_result,
    build_trace_skipped,
    trace_from_run_row,
)


class TraceBuilderTests(unittest.TestCase):
    def test_analyzed_trace_has_steps(self):
        result = AnalystRunResult(
            decision="not-hate",
            ocr_text="hello world",
            transcript="",
            stage1={"text_score": 0.08, "vision_score": 0.0},
            stage2={"fused": 0.08},
            backends={"ocr": "rapid", "clip": "clip"},
            notes=["stopped_at_stage1"],
            latency_ms={"ocr_ms": 100, "clip_ms": 50, "stage1_ms": 10, "total_ms": 200},
            risk_score=0.08,
            explanation="Stage-1 clear.",
        )
        tr = build_trace_from_result(
            ts="2026-01-01T12:00:00",
            tick=1,
            run_id="abc",
            app={"exe": "chrome.exe", "title": "Chat", "title_hash": "x"},
            frame_w=1280,
            frame_h=720,
            had_frame=True,
            had_audio=False,
            screen_ok=True,
            screen_error="",
            capture_ms=40,
            result=result,
        )
        self.assertEqual(tr["outcome"], "analyzed")
        self.assertEqual(len(tr["steps"]), 7)
        self.assertEqual(tr["steps"][0]["id"], "capture")
        self.assertEqual(tr["steps"][-1]["id"], "decide")
        self.assertIn("Clear", tr["steps"][-1]["detail"])

    def test_skipped_trace(self):
        tr = build_trace_skipped(ts="t", tick=2, app={"exe": "Desktop", "title": ""})
        self.assertEqual(tr["outcome"], "skipped")
        self.assertEqual(tr["steps"][0]["detail"], "same frame as last tick")

    def test_failed_trace(self):
        tr = build_trace_failed(ts="t", tick=3, app={"exe": "x"}, error="BitBlt fail")
        self.assertEqual(tr["outcome"], "capture_failed")
        self.assertEqual(tr["steps"][0]["status"], "fail")

    def test_trace_buffer_recent_apps(self):
        buf = TraceBuffer(maxlen=5)
        buf.push(build_trace_skipped(ts="1", tick=1, app={"exe": "a.exe", "title": "A"}))
        buf.push(
            build_trace_from_result(
                ts="2",
                tick=2,
                run_id="r",
                app={"exe": "b.exe", "title": "B"},
                frame_w=100,
                frame_h=80,
                had_frame=True,
                had_audio=False,
                screen_ok=True,
                screen_error="",
                capture_ms=1,
                result=AnalystRunResult(decision="not-hate", risk_score=0.0),
            )
        )
        apps = buf.recent_apps()
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0]["exe"], "b.exe")

    def test_trace_from_run_row(self):
        row = {
            "id": "run1",
            "ts": "2026-01-01T12:00:00",
            "decision": "not-hate",
            "score": 0.1,
            "app_exe": "firefox.exe",
            "ocr_snippet": "hi",
            "stage1": {"text_score": 0.1, "vision_score": 0.0},
            "stage2": {"fused": 0.1},
            "latency_ms": {"total_ms": 50},
            "notes": ["explanation:test"],
            "backends": {},
        }
        tr = trace_from_run_row(row)
        self.assertEqual(tr["run_id"], "run1")
        self.assertGreaterEqual(len(tr["steps"]), 5)


if __name__ == "__main__":
    unittest.main()
