"""The list endpoint must stay light.

/api/runs is polled every 3 seconds. It used to serialise every run complete —
preview image, rebuilt trace and all per-region detections — which measured
3.16 MB per poll, 63 MB a minute, to draw a list of headlines worth 0.07 MB.
The heavy fields are only ever rendered for the single run the panel has
selected, and it fetches those from /api/runs/{id}.

Nothing about that is enforced by types, so it is enforced here.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "analyst" / "panel" / "index.html"

# Fields big enough to matter, and drawn only for the selected run.
HEAVY = ("thumb_data_url", "trace")


def _row():
    """A stored row shaped like the real thing, with the bulky parts present."""
    return {
        "id": "abc123",
        "ts": "2026-09-03T12:00:00+05:30",
        "decision": "hate",
        "category": "hate_identity",
        "score": 0.88,
        "app_exe": "chrome.exe",
        "child_safe_summary": "Someone used hurtful language.",
        "stage1": {"text_score": 0.88},
        "stage2": {"fused": 0.88},
        "latency_ms": {"total_ms": 1234.5},
        "notes": [],
        "lexicon_hits": ["kill yourself"],
        "thumb_jpeg": b"\xff\xd8\xff" + b"x" * 40000,
        "evidence": {
            "escalated": True,
            "lexicon_score": 0.88,
            "detections": [
                {"kind": "text", "text": "a line of text " * 4, "box": [0, 0, 1, 1],
                 "score": 0.08, "hits": []}
                for _ in range(120)
            ],
        },
    }


class ListPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from analyst.serve import _public_run
        cls.public = staticmethod(_public_run)

    def test_list_drops_the_heavy_fields(self):
        light = self.public(_row(), full=False)
        for key in HEAVY:
            self.assertNotIn(key, light, f"{key} is back in the list payload")

    def test_list_drops_per_region_detections(self):
        light = self.public(_row(), full=False)
        self.assertNotIn("detections", light.get("evidence") or {})

    def test_list_keeps_what_the_list_draws(self):
        """renderFeed needs a headline plus evidence.escalated; renderHome adds
        the child-safe summary. Dropping any of these blanks the list."""
        light = self.public(_row(), full=False)
        for key in ("id", "ts", "decision", "category", "score", "app_exe",
                    "child_safe_summary"):
            self.assertIn(key, light, f"the list needs {key}")
        self.assertIn("escalated", light.get("evidence") or {})

    def test_detail_still_carries_everything(self):
        full = self.public(_row())
        self.assertIn("thumb_data_url", full)
        self.assertTrue(full["thumb_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("trace", full)
        self.assertIn("detections", full["evidence"])

    def test_list_is_dramatically_smaller(self):
        row = _row()
        light = len(json.dumps(self.public(row, full=False)))
        full = len(json.dumps(self.public(row)))
        self.assertLess(
            light, full * 0.25,
            f"list row is {light} bytes against {full} full — the split is not working",
        )

    def test_list_does_not_rebuild_the_trace(self):
        """Rebuilding a trace per row, 40 rows per poll, for output that is
        then discarded."""
        self.assertIsNone(self.public(_row(), full=False).get("trace"))


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class PanelDetailFetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PANEL.read_text(encoding="utf-8")

    def test_panel_fetches_detail_for_the_selection(self):
        self.assertIn("loadDetail", self.html)
        self.assertIn("/api/runs/${encodeURIComponent(id)}", self.html)

    def test_detail_cache_is_bounded(self):
        self.assertIn("DETAIL_CACHE_MAX", self.html,
                      "an unbounded cache of runs with thumbnails grows all session")

    def test_missing_preview_is_not_reported_as_text_only(self):
        """Before the detail lands there is no thumbnail, which is not the same
        as a check that had no screen image."""
        self.assertIn("run._full", self.html)


if __name__ == "__main__":
    unittest.main()
