"""Foreground window / process — what is on screen right now (Windows)."""

from __future__ import annotations

import hashlib
import sys
from typing import Dict

_DEFAULT = {"exe": "Desktop", "title": "", "title_hash": "", "pid": 0}


def _hash_title(title: str) -> str:
    if not title:
        return ""
    return hashlib.sha256(title.encode("utf-8", errors="replace")).hexdigest()[:16]


def get_foreground_app() -> Dict[str, object]:
    """Return {exe, title, title_hash, pid} for the active window."""
    if sys.platform != "win32":
        return dict(_DEFAULT)

    try:
        import ctypes
        from ctypes import wintypes
        from pathlib import Path

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return dict(_DEFAULT)

        length = user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(hwnd, buf, length)
        title = (buf.value or "")[:120]

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        exe = "unknown"
        h = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED
        if h:
            try:
                size = wintypes.DWORD(32768)
                exe_buf = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(h, 0, exe_buf, ctypes.byref(size)):
                    exe = Path(exe_buf.value).name or "unknown"
            finally:
                kernel32.CloseHandle(h)

        return {
            "exe": exe,
            "title": title,
            "title_hash": _hash_title(title),
            "pid": int(pid.value),
        }
    except Exception:
        return dict(_DEFAULT)
