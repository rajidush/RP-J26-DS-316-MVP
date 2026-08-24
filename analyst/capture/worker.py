"""Background capture loop: screen (+ optional loopback) → pipeline → SQLite.



Hardened:

- no overlapping ticks

- skip near-duplicate frames when silent

- failure backoff

- rich protection status for the panel

- whitebox trace (process + pipeline steps)

"""



from __future__ import annotations



import hashlib

import threading

import time

from datetime import datetime, timezone

from typing import Callable, Optional, Tuple



from analyst.capture.audio import LoopbackCapture

from analyst.capture.process import get_foreground_app

from analyst.capture.screen import ScreenCapture

from analyst.pipeline import AnalystPipeline

from analyst.store.db import AnalystStore

from analyst.store.persist import persist_result

from analyst.whitebox.trace import (

    TraceBuffer,

    build_trace_failed,

    build_trace_from_result,

    build_trace_skipped,

)





def _now() -> str:

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")





class CaptureWorker:

    def __init__(

        self,

        store: Optional[AnalystStore] = None,

        pipeline: Optional[AnalystPipeline] = None,

        screen: Optional[ScreenCapture] = None,

        audio: Optional[LoopbackCapture] = None,

        *,

        grab_frame: Optional[Callable[[], Tuple[Optional[bytes], int, int]]] = None,

        grab_audio: Optional[Callable[[], Tuple[Optional[bytes], float]]] = None,

    ) -> None:

        self.store = store or AnalystStore()

        self.pipeline = pipeline or AnalystPipeline()

        self.screen = screen or ScreenCapture()

        self.audio = audio or LoopbackCapture()

        self._grab_frame = grab_frame

        self._grab_audio = grab_audio

        self._thread: Optional[threading.Thread] = None

        self._stop = threading.Event()

        self._lock = threading.Lock()

        self._tick_lock = threading.Lock()

        self.child_age = 10

        self.interval_s = 2.5

        self.app_exe = "Desktop"

        self.last_error = ""

        self.last_run_id: Optional[str] = None

        self.last_ok_ts: Optional[str] = None

        self.ticks = 0

        self.skipped = 0

        self.failures = 0

        self._capturing = False

        self._last_frame_hash: Optional[str] = None

        self._consecutive_failures = 0

        self._trace = TraceBuffer(maxlen=30)

        self._current_app: dict = {"exe": "Desktop", "title": "", "title_hash": "", "pid": 0}

        self._last_frame_w = 0

        self._last_frame_h = 0



    @property

    def capturing(self) -> bool:

        return self._capturing



    @property

    def trace_buffer(self) -> TraceBuffer:

        return self._trace



    def whitebox(self) -> dict:

        """Live blackbox → whitebox snapshot for the panel."""

        app = get_foreground_app()

        self._current_app = app

        latest = self._trace.latest()

        return {

            "capturing": self._capturing,

            "current_app": app,

            "last_trace": latest,

            "traces": self._trace.list(limit=15),

            "recent_apps": self._trace.recent_apps(limit=8),

            "modules": {

                "screen": self.screen.name,

                "audio": self.audio.name,

                "audio_running": self.audio.running,

                **(self.pipeline.backends()),

            },

            "ticks": self.ticks,

            "skipped": self.skipped,

            "failures": self.failures,

        }



    def status(self) -> dict:

        latest = self.store.latest_run()

        protection = "idle"

        if self._capturing:

            protection = "guarding"

        if latest and latest.get("decision") == "hate":

            protection = "threat"

        elif self._consecutive_failures >= 3:

            protection = "degraded"

        elif self.last_error and "audio degraded" in self.last_error:

            protection = "guarding" if self._capturing else "idle"



        return {

            "capturing": self._capturing,

            "protection": protection,

            "child_age": self.child_age,

            "interval_s": self.interval_s,

            "ticks": self.ticks,

            "skipped": self.skipped,

            "failures": self.failures,

            "last_error": self.last_error,

            "last_run_id": self.last_run_id,

            "last_ok_ts": self.last_ok_ts,

            "screen": self.screen.name,

            "audio": self.audio.name,

            "audio_running": self.audio.running,

            "backends": self.pipeline.backends(),

            "stats": self.store.stats(),

            "current_app": self._current_app,

            "product": "Guardian Analyst",

            "note": "Temporary C2 capture — C1 will own capture later.",

        }



    def start(self, *, child_age: int = 10, interval_s: float = 2.5) -> dict:

        with self._lock:

            if self._capturing:

                return self.status()

            if self._grab_frame is None and not self.screen.available:

                self.last_error = self.screen.last_error or "screen capture unavailable (install mss)"

                return self.status()

            self.child_age = int(child_age)

            self.interval_s = max(2.0, float(interval_s))

            self.last_error = ""

            self._consecutive_failures = 0

            if self._grab_audio is None:

                if not self.audio.start():

                    self.last_error = f"audio degraded: {self.audio.last_error or 'loopback unavailable'}"

            self._stop.clear()

            self._thread = threading.Thread(target=self._run, name="analyst-capture", daemon=True)

            self._capturing = True

            self._thread.start()

            return self.status()



    def stop(self) -> dict:

        with self._lock:

            self._stop.set()

            thread = self._thread

        if thread and thread.is_alive():

            thread.join(timeout=8.0)

        try:

            self.audio.stop()

        except Exception:

            pass

        with self._lock:

            self._capturing = False

            self._thread = None

        return self.status()



    def tick_once(self) -> Optional[str]:

        """One capture→analyse→persist cycle (also used by tests with injectors)."""

        if not self._tick_lock.acquire(blocking=False):

            return None

        try:

            return self._tick_once_unlocked()

        finally:

            self._tick_lock.release()



    def _tick_once_unlocked(self) -> Optional[str]:

        ts = _now()

        tick_num = self.ticks + self.skipped + self.failures + 1

        app = get_foreground_app()

        self._current_app = app

        self.app_exe = str(app.get("exe") or "Desktop")



        t_cap = time.perf_counter()

        frame, fw, fh = self._take_frame()

        capture_ms = round((time.perf_counter() - t_cap) * 1000, 1)

        audio_b, _rms = self._take_audio()

        had_frame = frame is not None

        had_audio = audio_b is not None

        screen_ok = had_frame or not self.screen.last_error

        screen_err = self.screen.last_error or ""



        if frame is None and audio_b is None:

            if self.screen.last_error:

                self.last_error = f"screen: {self.screen.last_error}"

                self._consecutive_failures += 1

                self.failures += 1

                self._trace.push(

                    build_trace_failed(

                        ts=ts,

                        tick=tick_num,

                        app=app,

                        error=self.last_error,

                        capture_ms=capture_ms,

                    )

                )

            return None



        if frame is None and audio_b is not None and self.screen.last_error:

            self.last_error = f"screen: {self.screen.last_error}"



        if had_frame:

            self._last_frame_w, self._last_frame_h = fw, fh



        frame_hash = hashlib.sha1(frame).hexdigest() if frame else None

        if frame_hash and frame_hash == self._last_frame_hash and not audio_b:

            self.skipped += 1

            self._trace.push(

                build_trace_skipped(ts=ts, tick=tick_num, app=app)

            )

            return self.last_run_id



        try:

            result = self.pipeline.analyze(

                child_age=self.child_age,

                image_bytes=frame,

                audio_bytes=audio_b,

                app_exe=self.app_exe,

                app_category="desktop",

            )

            trace = build_trace_from_result(

                ts=ts,

                tick=tick_num,

                run_id="",

                app=app,

                frame_w=fw if had_frame else self._last_frame_w,

                frame_h=fh if had_frame else self._last_frame_h,

                had_frame=had_frame,

                had_audio=had_audio,

                screen_ok=screen_ok,

                screen_error=screen_err,

                capture_ms=capture_ms,

                result=result,

            )

            run_id = persist_result(

                self.store,

                result,

                child_age=self.child_age,

                app_exe=self.app_exe,

                frame_bytes=frame,

                trace=trace,

            )

            trace["run_id"] = run_id

            self._trace.push(trace)

            self.last_run_id = run_id

            self.last_ok_ts = ts

            self.ticks += 1

            self._last_frame_hash = frame_hash

            self._consecutive_failures = 0

            if result.decision != "hate":

                if "audio degraded" in (self.last_error or ""):

                    pass

                elif frame is None and self.screen.last_error:

                    self.last_error = f"screen: {self.screen.last_error}"

                else:

                    self.last_error = ""

            return run_id

        except Exception as exc:

            self.last_error = str(exc)

            self._consecutive_failures += 1

            self.failures += 1

            self._trace.push(

                build_trace_failed(

                    ts=ts,

                    tick=tick_num,

                    app=app,

                    error=str(exc),

                    capture_ms=capture_ms,

                )

            )

            return None



    def _take_frame(self):

        if self._grab_frame is not None:

            return self._grab_frame()

        return self.screen.grab_jpeg()



    def _take_audio(self):

        if self._grab_audio is not None:

            return self._grab_audio()

        if not self.audio.running:

            return None, 0.0

        try:

            return self.audio.pull_wav_if_speech()

        except Exception as exc:

            self.last_error = f"audio degraded: {exc}"

            return None, 0.0



    def _run(self) -> None:

        while not self._stop.is_set():

            t0 = time.perf_counter()

            self.tick_once()

            elapsed = time.perf_counter() - t0

            backoff = min(15.0, 2.0 * self._consecutive_failures) if self._consecutive_failures else 0.0

            wait = max(0.2, self.interval_s - elapsed + backoff)

            self._stop.wait(wait)


