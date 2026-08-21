"""Ingest helpers — until C1 bus exists, load frames/audio from disk for replay."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def load_image_bytes(path: Path) -> Optional[bytes]:
    if not path.is_file():
        return None
    return path.read_bytes()


def load_audio_bytes(path: Path) -> Optional[bytes]:
    if not path.is_file():
        return None
    return path.read_bytes()


def load_pair(
    image: Optional[Path] = None,
    audio: Optional[Path] = None,
) -> Tuple[Optional[bytes], Optional[bytes]]:
    return (
        load_image_bytes(image) if image else None,
        load_audio_bytes(audio) if audio else None,
    )
