"""Synthesize short demo WAV clips via Windows SAPI (no child recordings)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Spoken lines chosen so Whisper + lexicon both work.
CLIPS = [
    ("01_clean_gaming.wav", "good game easy noob that was a fun match"),
    ("02_hate_threat.wav", "you should kill yourself nobody likes you"),
    ("03_hate_identity.wav", "go back to where you came from you retard"),
    ("04_benign_chat.wav", "want to play minecraft later tonight"),
]


def _speak_to_wav(text: str, wav_path: Path) -> None:
    # Escape for PowerShell single-quoted string
    safe = text.replace("'", "''")
    path = str(wav_path).replace("'", "''")
    ps = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -2
$synth.SetOutputToWaveFile('{path}')
$synth.Speak('{safe}')
$synth.Dispose()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in CLIPS:
        dest = OUT / name
        try:
            _speak_to_wav(text, dest)
            print("wrote", name, f"({dest.stat().st_size} bytes)")
        except Exception as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
