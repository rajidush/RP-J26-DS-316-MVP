"""One-shot screen OCR test. Run this while a text-heavy window is visible.

Usage:
    .\\venv\\Scripts\\python.exe tests\\test_ocr_screen.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyst.capture import grab_screen_jpeg
from analyst.ocr import OcrEngine
from analyst.pipeline import AnalystPipeline
from PIL import Image


def main() -> None:
    print("=== Grabbing screen ===")
    frame_bytes, how = grab_screen_jpeg()
    if not frame_bytes:
        print(f"Screen grab unavailable ({how}). Run on Windows with a display.")
        return

    print(f"Captured {len(frame_bytes):,} bytes via {how}")

    print("\n=== Running OCR ===")
    engine = OcrEngine()
    print(f"Engine: {engine.name}")
    image = Image.open(io.BytesIO(frame_bytes))
    text = engine.extract(image)
    print(f"Extracted text ({len(text)} chars):")
    print(text[:600] or "(nothing extracted)")

    print("\n=== Running full Analyst pipeline on screen ===")
    pipe = AnalystPipeline()
    result = pipe.analyze(
        child_age=10,
        image_bytes=frame_bytes,
        capture_screen=False,  # already grabbed above
    )
    print(f"Decision:   {result.decision}")
    print(f"Risk score: {result.risk_score:.2f}")
    print(f"Category:   {result.category}")
    print(f"OCR text:   {result.ocr_text[:200] or '(none)'}")
    print(f"Escalated:  {result.escalated}")
    print(f"Stage 1:    {result.stage1}")
    print(f"Stage 2:    {result.stage2}")
    print(f"Media del:  {result.media_deleted}")
    print(f"Backends:   {result.backends.model_dump()}")
    print(f"Notes:      {result.notes}")


if __name__ == "__main__":
    main()
