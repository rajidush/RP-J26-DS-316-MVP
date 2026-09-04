"""Which detections survive into the stored run.

The panel's "where the engine looked" overlay is drawn entirely from the
detections persisted with a run, and only a bounded number can be kept. The
selection used to be `detections[:40]` — the first forty in reading order.

Reading order starts at the top of the screen, which on any browser is the tab
strip, the URL bar and the bookmarks bar. So the stored evidence was forty
fragments of browser chrome, the whole page body was discarded, and the overlay
drew boxes only across the tabs. Working OCR looked blind.

Detection itself was never affected — the classifier scores the full OCR text.
This is the evidence trail, and it was keeping the least interesting regions on
screen.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.store.persist import _MAX_DETECTIONS, _keep_detections


def _screenful():
    """Browser chrome at the top, page content below — in reading order."""
    chrome = [{"kind": "text", "text": f"tab{i}", "score": 0.08,
               "box": [0, 0.01, 0.1, 0.03]} for i in range(45)]
    body = [{"kind": "text", "text": f"a real sentence of page content number {i}",
             "score": 0.08, "box": [0.1, 0.4, 0.9, 0.42]} for i in range(45)]
    return chrome + body


class SelectionTests(unittest.TestCase):
    def test_page_content_is_not_discarded_for_chrome(self):
        kept = _keep_detections(_screenful())
        body = [d for d in kept if "page content" in d["text"]]
        self.assertTrue(
            body,
            "stored evidence kept only top-of-screen chrome — the overlay would "
            "show boxes on the tab strip and nothing on the page",
        )

    def test_matched_regions_always_survive_the_cap(self):
        """A region that fired a rule is the whole reason the run exists."""
        dets = [{"kind": "text", "text": f"quiet line {i}", "score": 0.08,
                 "box": [0, 0.01, 0.1, 0.03]} for i in range(_MAX_DETECTIONS + 60)]
        dets.append({"kind": "text", "text": "the one that fired", "score": 0.91,
                     "box": [0.2, 0.5, 0.6, 0.52]})
        kept = _keep_detections(dets)
        self.assertLessEqual(len(kept), _MAX_DETECTIONS)
        self.assertEqual(kept[0]["text"], "the one that fired")

    def test_picture_region_is_always_kept(self):
        """There is at most one, and it is what the vision model was shown."""
        dets = [{"kind": "text", "text": f"line {i}", "score": 0.5,
                 "box": [0, 0.01, 0.1, 0.03]} for i in range(_MAX_DETECTIONS + 20)]
        dets.append({"kind": "picture", "text": "a caption", "score": 0.0,
                     "box": [0.1, 0.1, 0.5, 0.5]})
        kept = _keep_detections(dets)
        self.assertLessEqual(len(kept), _MAX_DETECTIONS)
        self.assertTrue(any(d["kind"] == "picture" for d in kept))

    def test_fragments_rank_below_real_text(self):
        """"x" and "G" off a tab strip are noise, not content."""
        dets = [{"kind": "text", "text": "x", "score": 0.08, "box": [0, 0, 0.1, 0.1]},
                {"kind": "text", "text": "G", "score": 0.08, "box": [0, 0, 0.1, 0.1]},
                {"kind": "text", "text": "Hate speech is a form of violence",
                 "score": 0.08, "box": [0, 0.5, 0.9, 0.52]}]
        self.assertEqual(_keep_detections(dets)[0]["text"],
                         "Hate speech is a form of violence")

    def test_cap_is_respected(self):
        dets = [{"kind": "text", "text": f"line {i}", "score": 0.08,
                 "box": [0, 0, 0.1, 0.1]} for i in range(_MAX_DETECTIONS * 3)]
        self.assertLessEqual(len(_keep_detections(dets)), _MAX_DETECTIONS)

    def test_cap_covers_a_busy_desktop(self):
        """A 1920x1080 screen yields ~100 regions; a cap below that would throw
        away real content on every ordinary check."""
        self.assertGreaterEqual(_MAX_DETECTIONS, 100)

    def test_empty_input_is_safe(self):
        self.assertEqual(_keep_detections(None), [])
        self.assertEqual(_keep_detections([]), [])

    def test_missing_fields_do_not_raise(self):
        """Detections come from OCR output; a malformed one must not take the
        persist step down with it."""
        dets = [{"kind": "text"},
                {"kind": "text", "text": None, "score": None},
                {"text": "no kind", "score": "not a number"}]
        kept = _keep_detections(dets)
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main()
