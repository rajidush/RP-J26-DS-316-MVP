"""Image *meaning* and *image text* via a local vision-language model.

Why this exists
---------------
`image_fast` scores a CLIP embedding and is deferred until Milestone A2, which
leaves the meme case uncovered: a picture whose harm is not in any text OCR can
reach. A small local VLM (LM Studio, OpenAI-compatible) closes that gap.

Two hard lessons, both measured on lfm2.5-vl-450m:

1. **Never send the whole desktop.** Given a full screen grab the model does not
   hedge, it invents. A bullying poster embedded in a browser came back as
   "a red square with the text GUARANTEE", and asking for the visible text
   produced a fabricated website navigation menu. Cropped to the poster, the
   same model returned the slogans verbatim. So we crop with
   `extract.region.crop_to_region` first and decline to ask when no distinct
   picture is found.

2. **Never let the model decide "is this hate".** It cannot follow a strict
   label format at this size (it replies "ONLY"), and delegating the verdict
   would move the definition of hate out of the auditable lexicon. The VLM only
   *reads*; Stage 1 still judges.

The two readings serve different questions the panel must answer:
  - `caption`    -> what the picture means
  - `image_text` -> what words are written in the picture

Both are scored by the same Stage-1 scorer as OCR and ASR text.

Env:
    ANALYST_VLM_URL     e.g. http://127.0.0.1:1234/v1   (unset = disabled)
    ANALYST_VLM_MODEL   default lfm2.5-vl-450m
    ANALYST_VLM_WIDTH   default 512   (crop is small already; latency is fine)
    ANALYST_VLM_TIMEOUT default 60 (seconds) - accuracy over speed
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image

from .region import crop_to_region

# Benchmarked on the demo assets and a synthetic desktop (CPU, cropped, 512px):
#   lfm2.5-vl-450m  ~2-4s  reads overlay text verbatim   <- default
#   lfm2-vl-450m    ~0.4s  faster but paraphrases text away, losing slurs
#   google/gemma-4-e4b     returned empty in this LM Studio setup
DEFAULT_MODEL = "lfm2.5-vl-450m"
DEFAULT_WIDTH = 512
DEFAULT_TIMEOUT = 60.0

MEANING_PROMPT = "Describe what is happening in this image in one sentence."
TEXT_PROMPT = (
    "Transcribe every word of text visible in this image, verbatim. "
    "If there is no text, reply NONE."
)

# Small chat models wrap answers in scaffolding; strip it so the lexicon sees
# the words themselves rather than "Sure! Here's the transcription:".
_PREAMBLE = re.compile(
    r"^(sure[!,.]?\s*)?(here('s| is)[^:]{0,40}:)\s*", re.IGNORECASE
)
_MARKDOWN = re.compile(r"[*_`#>\-]{1,3}")


@dataclass
class VisionReading:
    caption: str = ""
    image_text: str = ""
    box: Optional[Tuple[int, int, int, int]] = None
    ms: float = 0.0
    error: str = ""
    notes: list = field(default_factory=list)

    @property
    def any_text(self) -> bool:
        return bool(self.caption or self.image_text)

    def combined(self) -> str:
        parts = [p for p in (self.caption, self.image_text) if p]
        return " ".join(parts).strip()


def _clean(text: str) -> str:
    out = _PREAMBLE.sub("", (text or "").strip())
    out = _MARKDOWN.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    if out.upper().strip(" .") == "NONE":
        return ""
    return out


def _jpeg(image: Image.Image, width: int) -> bytes:
    rgb = image.convert("RGB")
    if rgb.width > width:
        rgb = rgb.resize((width, max(1, int(rgb.height * width / rgb.width))), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


class VisionMeaning:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        width: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ANALYST_VLM_URL", "")).rstrip("/")
        self.model = model or os.environ.get("ANALYST_VLM_MODEL", DEFAULT_MODEL)
        self.width = int(width or os.environ.get("ANALYST_VLM_WIDTH", DEFAULT_WIDTH))
        self.timeout = float(timeout or os.environ.get("ANALYST_VLM_TIMEOUT", DEFAULT_TIMEOUT))
        self.last_error = ""
        self.name = f"lmstudio:{self.model}" if self.base_url else "none"

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _ask(self, jpeg: bytes, prompt: str, max_tokens: int = 160) -> str:
        b64 = base64.b64encode(jpeg).decode("ascii")
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"] or ""

    def read(self, image: Optional[Image.Image]) -> VisionReading:
        """Crop to the picture on screen, then read its meaning and its text."""
        reading = VisionReading()
        if image is None or not self.enabled:
            return reading

        crop, box = crop_to_region(image)
        if crop is None:
            # No distinct picture: asking anyway is how hallucinations get in.
            reading.notes.append("no_image_region")
            return reading
        reading.box = box

        t0 = time.perf_counter()
        try:
            jpeg = _jpeg(crop, self.width)
            reading.caption = _clean(self._ask(jpeg, MEANING_PROMPT, max_tokens=110))
            reading.image_text = _clean(self._ask(jpeg, TEXT_PROMPT, max_tokens=180))
            self.last_error = ""
        except urllib.error.URLError as exc:
            reading.error = f"vlm unreachable at {self.base_url}: {exc.reason}"
            self.last_error = reading.error
        except Exception as exc:
            reading.error = f"{type(exc).__name__}: {exc}"
            self.last_error = reading.error
        reading.ms = round((time.perf_counter() - t0) * 1000, 1)
        return reading

    def caption(self, image: Optional[Image.Image]) -> str:
        """Back-compat shorthand: just the meaning sentence."""
        return self.read(image).caption
