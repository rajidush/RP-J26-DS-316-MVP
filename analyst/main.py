"""CLI: python -m analyst --text "..." | --image path | --audio path | --replay dir"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m analyst` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyst.ingest.frames import load_pair
from analyst.pipeline import AnalystPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="J26-DS-316 Analyst (C2) — on-device hate-speech cascade"
    )
    p.add_argument("--text", default="", help="Overlay / chat text (no OCR needed)")
    p.add_argument("--image", type=Path, help="Screenshot / frame path")
    p.add_argument("--audio", type=Path, help="Voice clip (wav/webm)")
    p.add_argument("--age", type=int, default=10, help="Child age (persona thresholds)")
    p.add_argument("--exe", default="Discord.exe", help="Foreground app exe")
    p.add_argument("--category", default="chat", help="App category")
    p.add_argument(
        "--replay",
        type=Path,
        help="Directory with *.png/*.jpg and optional *.wav — run each image",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON result")
    return p


def print_result(result, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    print(f"decision:  {result.decision}")
    print(f"backends:  {result.backends}")
    print(f"stage1:    {result.stage1}")
    if result.stage2:
        print(f"stage2:    {result.stage2}")
    if result.ocr_text:
        print(f"ocr:       {result.ocr_text[:120]}")
    if result.transcript:
        print(f"asr:       {result.transcript[:120]}")
    print(f"latency:   {result.latency_ms}")
    print(f"media_del: {result.media_deleted}")
    if result.payload:
        print(f"score:     {result.payload.score}")
        print(f"category:  {result.payload.category}")
        print(f"summary:   {result.payload.child_safe_summary}")
        print(f"action:    {result.payload.recommended_action}")
        print(f"topic:     hate.detected")
    if result.notes:
        print(f"notes:     {', '.join(result.notes)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipe = AnalystPipeline()

    if args.replay:
        folder = args.replay
        if not folder.is_dir():
            print(f"Replay folder not found: {folder}", file=sys.stderr)
            return 2
        images = sorted(
            list(folder.glob("*.png"))
            + list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
        )
        if not images:
            print("No images in replay folder.", file=sys.stderr)
            return 2
        for img in images:
            wav = img.with_suffix(".wav")
            audio = wav if wav.is_file() else None
            print(f"\n=== {img.name} ===")
            frame, audio_b = load_pair(img, audio)
            result = pipe.analyze(
                child_age=args.age,
                overlay_text=args.text,
                image_bytes=frame,
                audio_bytes=audio_b,
                app_exe=args.exe,
                app_category=args.category,
            )
            print_result(result, args.json)
        return 0

    frame, audio_b = load_pair(args.image, args.audio)
    result = pipe.analyze(
        child_age=args.age,
        overlay_text=args.text,
        image_bytes=frame,
        audio_bytes=audio_b,
        app_exe=args.exe,
        app_category=args.category,
    )
    print_result(result, args.json)
    return 0 if result.decision in ("hate", "not-hate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
