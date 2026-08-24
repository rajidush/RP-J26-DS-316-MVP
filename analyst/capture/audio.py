"""Temporary C2 speaker loopback via soundcard (WASAPI on Windows)."""

from __future__ import annotations

import io
import threading
import wave
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

_SOUND_OK = False
try:
    import soundcard as sc  # noqa: F401

    _SOUND_OK = True
except Exception:
    sc = None  # type: ignore


class LoopbackCapture:
    """Ring buffer of loopback samples; pull short WAV when energy is high enough."""

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_seconds: float = 4.0,
        rms_gate: float = 0.012,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.rms_gate = rms_gate
        self.name = "soundcard-loopback" if _SOUND_OK else "none"
        self._last_error = ""
        self._lock = threading.Lock()
        self._chunks: Deque[np.ndarray] = deque()
        self._max_samples = int(sample_rate * buffer_seconds)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()   # set once the device is actually open
        self._running = False

    @property
    def available(self) -> bool:
        return _SOUND_OK

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if not _SOUND_OK:
            self._last_error = "soundcard not installed"
            return False
        if self._running:
            return True
        self._stop.clear()
        self._ready.clear()
        self._last_error = ""
        self._thread = threading.Thread(target=self._loop, name="analyst-loopback", daemon=True)
        self._thread.start()
        # Wait for the recorder to actually open. Without this the thread can
        # die on an unsupported device while start() still reports success,
        # leaving the panel showing a healthy audio branch that captures
        # nothing.
        if not self._ready.wait(timeout=5.0):
            self._stop.set()
            if not self._last_error:
                self._last_error = "loopback device did not start within 5s"
            self._running = False
            return False
        # _ready is also set by the failure path, so confirm the loop is live.
        if not self._running:
            if not self._last_error:
                self._last_error = "loopback device failed to open"
            return False
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running = False
        with self._lock:
            self._chunks.clear()

    def _loop(self) -> None:
        try:
            import soundcard as sc

            # Speaker objects only play. To record what the speakers output we
            # need the matching loopback *microphone*.
            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            block = int(self.sample_rate * 0.25)
            with mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
                self._running = True
                self._ready.set()
                while not self._stop.is_set():
                    data = np.asarray(rec.record(numframes=block), dtype=np.float32)
                    mono = data.mean(axis=1) if data.ndim > 1 else data.reshape(-1)
                    with self._lock:
                        self._chunks.append(mono.copy())
                        self._trim()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._running = False
            self._ready.set()   # unblock start() so it can report the failure

    def _trim(self) -> None:
        total = sum(len(c) for c in self._chunks)
        while total > self._max_samples and self._chunks:
            dropped = self._chunks.popleft()
            total -= len(dropped)

    def _concat(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self._chunks))

    def pull_wav_if_speech(self, duration_s: float = 3.0) -> Tuple[Optional[bytes], float]:
        """Return (wav_bytes, rms) if recent energy exceeds gate; else (None, rms)."""
        audio = self._concat()
        if audio.size == 0:
            return None, 0.0
        need = int(self.sample_rate * duration_s)
        if audio.size > need:
            audio = audio[-need:]
        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
        if rms < self.rms_gate:
            return None, rms
        return _float_to_wav(audio, self.sample_rate), rms


def _float_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
