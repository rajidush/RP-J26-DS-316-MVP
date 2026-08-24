"""Bounded RAM slots. Frames/audio never go to disk; wiped after each run."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional


def _to_buffer(data: Optional[bytes]) -> Optional[bytearray]:
    if data is None:
        return None
    if isinstance(data, bytearray):
        return data
    return bytearray(data)


def _zero(buf: Optional[bytearray]) -> None:
    if buf is None:
        return
    for i in range(len(buf)):
        buf[i] = 0
    buf.clear()


class TransientMediaBuffer:
    def __init__(self, max_slots: int = 2) -> None:
        self.max_slots = max_slots
        self._lock = threading.Lock()
        self._slots: OrderedDict[str, dict] = OrderedDict()

    def put(
        self,
        trigger_id: str,
        frame: Optional[bytes] = None,
        audio: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            if trigger_id in self._slots:
                self._wipe(self._slots.pop(trigger_id))
            while len(self._slots) >= self.max_slots:
                _, old = self._slots.popitem(last=False)
                self._wipe(old)
            self._slots[trigger_id] = {
                "frame": _to_buffer(frame),
                "audio": _to_buffer(audio),
            }

    def get(self, trigger_id: str) -> dict:
        """Detached copies. Callers own them; delete() cannot wipe these."""
        with self._lock:
            slot = self._slots.get(trigger_id)
            if slot is None:
                return {"frame": None, "audio": None}
            return {
                "frame": bytes(slot["frame"]) if slot["frame"] is not None else None,
                "audio": bytes(slot["audio"]) if slot["audio"] is not None else None,
            }

    def borrow(self, trigger_id: str) -> dict:
        """Live buffers — the same objects delete() zeroes. Never copy or retain."""
        with self._lock:
            slot = self._slots.get(trigger_id)
            if slot is None:
                return {"frame": None, "audio": None}
            return {"frame": slot["frame"], "audio": slot["audio"]}

    def delete(self, trigger_id: str) -> bool:
        with self._lock:
            slot = self._slots.pop(trigger_id, None)
            if slot is None:
                return False
            self._wipe(slot)
            return True

    def occupied(self) -> int:
        with self._lock:
            return len(self._slots)

    def _wipe(self, slot: Optional[dict]) -> None:
        if not slot:
            return
        _zero(slot.get("frame"))
        _zero(slot.get("audio"))
        slot["frame"] = None
        slot["audio"] = None

    @contextmanager
    def hold(
        self,
        trigger_id: str,
        frame: Optional[bytes] = None,
        audio: Optional[bytes] = None,
    ) -> Iterator[dict]:
        self.put(trigger_id, frame=frame, audio=audio)
        try:
            # borrow(), not get(): the yielded buffers are the ones delete()
            # zeroes, so the wipe actually covers the bytes the caller read.
            yield self.borrow(trigger_id)
        finally:
            self.delete(trigger_id)
