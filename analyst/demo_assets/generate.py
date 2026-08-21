"""Create synthetic chat screenshots for --replay demos (no child data)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent


SAMPLES = [
    ("01_clean_gaming.png", "gg ez noob that was a fun match"),
    ("02_hate_threat.png", "you should kys nobody likes you"),
    ("03_hate_identity.png", "go back to where you came from retard"),
    ("04_benign_chat.png", "want to play minecraft later tonight?"),
]


def _font(size: int = 28):
    for name in ("arial.ttf", "seguiemj.ttf", "Calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render(path: Path, text: str) -> None:
    img = Image.new("RGB", (900, 200), color=(32, 34, 37))  # discord-ish dark
    draw = ImageDraw.Draw(img)
    draw.rectangle((16, 16, 884, 184), fill=(47, 49, 54))
    draw.text((36, 40), "DemoUser", fill=(114, 137, 218), font=_font(22))
    draw.text((36, 80), text, fill=(220, 221, 222), font=_font(28))
    img.save(path, format="PNG")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES:
        render(OUT / name, text)
        print("wrote", name)
    (OUT / "README.md").write_text(
        "# Demo assets (synthetic)\n\n"
        "Generated chat-style PNGs for `python -m analyst --replay analyst/demo_assets`.\n"
        "No real child data. Regenerate: `python -m analyst.demo_assets.generate`\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
