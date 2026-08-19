"""Test OCR on a synthetic image with clear printed text (works without a real screenshot)."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyst.ocr import OcrEngine
from analyst.pipeline import AnalystPipeline
from PIL import Image, ImageDraw, ImageFont


def make_test_image(text: str) -> bytes:
    img = Image.new("RGB", (600, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run(label: str, text: str) -> None:
    print(f"\n--- {label} ---")
    img_bytes = make_test_image(text)
    engine = OcrEngine()
    image = Image.open(io.BytesIO(img_bytes))
    extracted = engine.extract(image)
    print(f"Rendered : {repr(text)}")
    print(f"OCR out  : {repr(extracted)}")

    pipe = AnalystPipeline()
    result = pipe.analyze(child_age=10, image_bytes=img_bytes)
    print(f"Decision : {result.decision}  score={result.risk_score:.2f}  cat={result.category}")
    print(f"OCR text : {repr(result.ocr_text)}")
    print(f"Media del: {result.media_deleted}")


if __name__ == "__main__":
    run("Clean gaming text",     "gg ez wp nice game everyone")
    run("Threat phrase",         "you should kys loser nobody likes you")
    run("Identity attack",       "get out of here retard go back to where you came from")
    run("Benign meme caption",   "when you miss the bus on Monday morning lol")
