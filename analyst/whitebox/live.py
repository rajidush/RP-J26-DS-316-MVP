"""What the engine is doing *right now*.

The trace in `trace.py` is a post-mortem: it exists only once a check has
finished. That answers "what happened" but not "what is happening", and the
panel's white box needs the second one — a parent watching the diagram should
see the current check move through it, not a replay of the previous one.

This is deliberately a process-global singleton rather than something threaded
through every call. The pipeline is already deep (worker → pipeline → extract →
scorers), and passing a reporter down that chain purely so a UI can watch would
put presentation concerns in every signature. The cost of a global is that two
concurrent analyses would blur together; the worker runs one check at a time by
design, and the manual /api/analyze path is a single request, so in practice
there is one pipeline in flight.

Stage names are the same ids the trace uses (`capture`, `ocr`, `audio`, `clip`,
`vlm`, `stage1`, `stage2`), so the panel can key one diagram off both.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional


class LiveState:
    """Thread-safe record of which stages are executing at this instant.

    Stages overlap — OCR and speech-to-text are submitted to a pool together —
    so this tracks a *set* of active stages, not a single current phase.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, float] = {}
        self._last_ms: Dict[str, float] = {}
        self._seq = 0
        self._busy_since: Optional[float] = None
        self._idle_since: float = time.time()
        self._label = ""
        self._marks: Dict[str, object] = {}

    # -- writers ---------------------------------------------------------------

    def begin(self, stage: str) -> None:
        with self._lock:
            self._active[stage] = time.perf_counter()
            self._seq += 1

    def end(self, stage: str) -> None:
        with self._lock:
            started = self._active.pop(stage, None)
            if started is not None:
                self._last_ms[stage] = round((time.perf_counter() - started) * 1000, 1)
            self._seq += 1

    def mark(self, key: str, value: object) -> None:
        """Record a fact about the current check, for the panel to show.

        Used for things a duration cannot express — notably whether Stage 1
        short-circuited past the language models because the lexicon already
        decided. Without this the diagram claims "lexicon + 2 models" on a
        check where the models never ran.
        """
        with self._lock:
            self._marks[key] = value
            self._seq += 1

    def start_check(self, label: str = "") -> None:
        """A whole check begins — used for the 'checking now' banner.

        Durations are cleared here, which makes `last_ms` mean "finished so far
        *in this check*". That is what lets the panel fill the diagram in
        progressively: a stage that has run stays marked done for the rest of
        the check instead of blinking once and vanishing between two polls.
        Some sub-steps take a few milliseconds, so a poll will simply never
        catch them mid-flight — but it will always see that they completed.
        """
        with self._lock:
            self._busy_since = time.perf_counter()
            self._label = label or ""
            self._last_ms.clear()
            self._marks.clear()
            self._seq += 1

    def end_check(self) -> None:
        with self._lock:
            self._busy_since = None
            self._active.clear()
            self._idle_since = time.time()
            self._seq += 1

    # -- reader ----------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            now = time.perf_counter()
            active = {
                stage: round((now - started) * 1000, 1)
                for stage, started in self._active.items()
            }
            return {
                # stage id -> how long it has been running, in ms
                "active": active,
                # stage id -> how long it took, for stages finished in the
                # current check (cleared by start_check)
                "last_ms": dict(self._last_ms),
                # convenience for the panel: the stages already finished
                "done": sorted(self._last_ms),
                "busy": self._busy_since is not None,
                "busy_ms": round((now - self._busy_since) * 1000, 1) if self._busy_since else 0.0,
                "idle_s": round(time.time() - self._idle_since, 1),
                "label": self._label,
                # per-check facts that are not durations
                "marks": dict(self._marks),
                # bumps on every transition, so a client can tell "nothing has
                # changed" from "changed and changed back between two polls"
                "seq": self._seq,
            }


LIVE = LiveState()


@contextmanager
def stage(name: str):
    """Mark a stage as executing for the duration of the block.

    Never swallows or alters exceptions — a stage that raises must still be
    recorded as finished, or the diagram would show it running forever.
    """
    LIVE.begin(name)
    try:
        yield
    finally:
        LIVE.end(name)


@contextmanager
def check(label: str = ""):
    LIVE.start_check(label)
    try:
        yield
    finally:
        LIVE.end_check()
