"""Create synthetic chat screenshots optimized for OCR demos (no child data)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent

SAMPLES = [
    ("01_clean_gaming.png", "gg ez noob that was a fun match", "clean"),
    ("02_hate_threat.png", "you should kys nobody likes you", "hate"),
    ("03_hate_identity.png", "go back to where you came from retard", "hate"),
    ("04_benign_chat.png", "want to play minecraft later tonight", "clean"),
]


def _font(size: int = 36):
    for name in ("arial.ttf", "C:\\Windows\\Fonts\\arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_vision_only(path: Path) -> None:
    """No OCR text — exercises the CLIP / image_fast branch (Step 4)."""
    img = Image.new("RGB", (512, 512), color=(25, 25, 30))
    draw = ImageDraw.Draw(img)
    # Aggressive visual motif without letters (OCR should stay empty / weak).
    draw.ellipse((80, 80, 430, 430), fill=(180, 20, 20), outline=(255, 60, 60), width=8)
    draw.polygon([(256, 100), (400, 380), (112, 380)], fill=(40, 0, 0))
    draw.rectangle((200, 220, 312, 340), fill=(10, 10, 10))
    img.save(path, format="PNG")


def render(path: Path, text: str) -> None:
    # Light high-contrast frame — RapidOCR is more reliable than dark Discord theme.
    img = Image.new("RGB", (1000, 220), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 980, 200), fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    draw.text((40, 40), "DemoUser:", fill=(20, 20, 20), font=_font(28))
    draw.text((40, 100), text, fill=(0, 0, 0), font=_font(36))
    img.save(path, format="PNG")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text, _kind in SAMPLES:
        render(OUT / name, text)
        print("wrote", name)
    vision = OUT / "05_vision_only.png"
    render_vision_only(vision)
    print("wrote", vision.name)
    (OUT / "README.md").write_text(
        "# Demo assets (synthetic)\n\n"
        "High-contrast chat PNGs (OCR) + `05_vision_only.png` (no text, CLIP path).\n\n"
        "```powershell\n"
        "python -m analyst.demo_assets.generate\n"
        "python -m analyst --replay analyst/demo_assets --age 10\n"
        "```\n\n"
        "No real child data.\n",
        encoding="utf-8",
    )
    expect = {name: text.lower() for name, text, _ in SAMPLES}
    import json

    (OUT / "expected.json").write_text(json.dumps(expect, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
