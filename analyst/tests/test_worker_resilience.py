"""The capture loop must outlive any single tick.

`_tick_once_unlocked` guards the analyse-and-persist half of a check, but the
half before it — grabbing the frame, reading the foreground window, building
the trace — was unguarded, and `_run` called `tick_once()` bare. An exception
escaping there killed the capture thread outright.

Nothing resets `_capturing` on that path, so the failure was invisible: the
panel went on reporting "monitoring · next check in Ns", with a countdown, while
no check ever ran again. A monitor that has silently stopped monitoring is
worse than one that reports itself broken.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.capture.worker import CaptureWorker


class _Boom(RuntimeError):
    pass


def _worker_with_tick(tick):
    """A worker whose tick body we control, with no store or pipeline."""
    w = CaptureWorker.__new__(CaptureWorker)
    w._stop = threading.Event()
    w._consecutive_failures = 0
    w.last_error = ""
    w.interval_s = 0.05
    w._last_tick_end = None
    w.tick_once = tick
    return w


class LoopSurvivalTests(unittest.TestCase):
    def test_loop_survives_a_raising_tick(self):
        # Stop after the second tick: each failure adds 2s of backoff, so
        # asking for four would spend 2+4+6s waiting before the loop exits.
        calls = []

        def tick():
            calls.append(1)
            if len(calls) >= 2:
                w._stop.set()
            raise _Boom("pipeline exploded")

        w = _worker_with_tick(tick)
        thread = threading.Thread(target=w._run, daemon=True)
        thread.start()
        thread.join(timeout=20)

        self.assertFalse(thread.is_alive(), "capture thread never exited")
        self.assertGreaterEqual(
            len(calls), 2,
            "the loop stopped after the first raising tick — the thread died",
        )

    def test_failure_backoff_grows(self):
        """Each consecutive failure should slow the retry, not spin hot."""
        waits = []
        orig_wait = threading.Event.wait

        def tick():
            if len(waits) >= 2:
                w._stop.set()
            raise _Boom("nope")

        w = _worker_with_tick(tick)
        w._stop.wait = lambda t=None: (waits.append(t), orig_wait(w._stop, 0))[1]
        w._run()
        self.assertGreaterEqual(len(waits), 2)
        self.assertGreater(waits[1], waits[0], f"backoff did not grow: {waits}")

    def test_a_crash_is_counted_not_swallowed(self):
        def tick():
            w._stop.set()
            raise _Boom("kaboom")

        w = _worker_with_tick(tick)
        w._run()
        self.assertEqual(w._consecutive_failures, 1)
        self.assertIn("kaboom", w.last_error)
        self.assertIn("tick crashed", w.last_error)

    def test_crash_message_is_bounded(self):
        """last_error is rendered in the panel; an enormous exception string
        should not be pasted into it whole."""
        def tick():
            w._stop.set()
            raise _Boom("x" * 5000)

        w = _worker_with_tick(tick)
        w._run()
        self.assertLessEqual(len(w.last_error), 200)

    def test_repeated_crashes_reach_the_degraded_threshold(self):
        """status() calls a run degraded at 3 consecutive failures, so the
        counter has to actually climb."""
        n = []

        def tick():
            n.append(1)
            if len(n) >= 3:
                w._stop.set()
            raise _Boom("again")

        w = _worker_with_tick(tick)
        w._run()
        self.assertGreaterEqual(w._consecutive_failures, 3)

    def test_a_healthy_tick_does_not_count_a_failure(self):
        def tick():
            w._stop.set()
            return "run-id"

        w = _worker_with_tick(tick)
        w._run()
        self.assertEqual(w._consecutive_failures, 0)
        self.assertEqual(w.last_error, "")

    def test_loop_records_when_the_tick_finished(self):
        """The panel's "next check in Ns" countdown reads this."""
        def tick():
            w._stop.set()

        w = _worker_with_tick(tick)
        self.assertIsNone(w._last_tick_end)
        w._run()
        self.assertIsNotNone(w._last_tick_end, "countdown has no reference point")


if __name__ == "__main__":
    unittest.main()
