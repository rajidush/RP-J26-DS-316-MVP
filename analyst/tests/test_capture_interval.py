"""Tick spacing must describe a rate the engine can actually deliver.

The panel used to offer "2.5 seconds". Measured over 92 real checks, 87% of
them ran longer than that, so every tick collided with the one before it, got
skipped, and left the panel frozen — the setting promised a rate that was never
physically available. These tests keep the three layers (UI options, API
bounds, worker floor) agreed on what is achievable.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.capture.worker import _DEFAULT_INTERVAL_S, _MIN_INTERVAL_S

PANEL = ROOT / "analyst" / "panel" / "index.html"

# p90 of a full check, measured on a 1920x1080 desktop. The floor has to sit
# above this or overlap is the normal case rather than the exception.
MEASURED_P90_S = 9.3


class WorkerIntervalTests(unittest.TestCase):
    def test_floor_exceeds_a_typical_check(self):
        self.assertGreater(
            _MIN_INTERVAL_S,
            MEASURED_P90_S,
            "the minimum interval is shorter than 9 checks in 10 — those ticks "
            "can only overlap and be skipped",
        )

    def test_default_leaves_the_panel_responsive(self):
        """A check blocks the API for its duration, so the default must spend
        most of the cycle idle rather than mid-check."""
        busy_fraction = MEASURED_P90_S / _DEFAULT_INTERVAL_S
        self.assertLess(busy_fraction, 0.5, "default interval leaves the UI blocked over half the time")

    def test_default_is_at_or_above_the_floor(self):
        self.assertGreaterEqual(_DEFAULT_INTERVAL_S, _MIN_INTERVAL_S)


class ApiBoundTests(unittest.TestCase):
    def _model(self):
        from analyst.serve import StartBody

        return StartBody

    def test_default_matches_the_worker(self):
        self.assertEqual(self._model()().interval_s, _DEFAULT_INTERVAL_S)

    def test_impossible_interval_is_rejected(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._model()(interval_s=2.5)

    def test_offered_intervals_are_accepted(self):
        model = self._model()
        for value in _panel_options():
            model(interval_s=value)  # must not raise


def _panel_options():
    html = PANEL.read_text(encoding="utf-8")
    block = re.search(r'<select id="interval">(.*?)</select>', html, re.S)
    assert block, "interval select not found in the panel"
    return [float(v) for v in re.findall(r'value="([\d.]+)"', block.group(1))]


def _panel_default():
    html = PANEL.read_text(encoding="utf-8")
    block = re.search(r'<select id="interval">(.*?)</select>', html, re.S)
    sel = re.search(r'value="([\d.]+)"[^>]*\bselected\b', block.group(1))
    assert sel, "no interval option marked selected"
    return float(sel.group(1))


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class PanelOptionTests(unittest.TestCase):
    def test_no_option_is_below_the_floor(self):
        offered = _panel_options()
        self.assertTrue(offered)
        too_fast = [v for v in offered if v < _MIN_INTERVAL_S]
        self.assertEqual(
            too_fast,
            [],
            f"panel offers {too_fast}s, which the worker silently clamps to "
            f"{_MIN_INTERVAL_S}s — the UI would be lying about the rate",
        )

    def test_panel_default_matches_the_worker_default(self):
        self.assertEqual(_panel_default(), _DEFAULT_INTERVAL_S)

    def test_options_are_ordered_and_distinct(self):
        offered = _panel_options()
        self.assertEqual(len(offered), len(set(offered)), "duplicate interval options")
        self.assertEqual(offered, sorted(offered), "interval options are out of order")


if __name__ == "__main__":
    unittest.main()
