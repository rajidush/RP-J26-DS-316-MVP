"""Step 5 — end-to-end demo: text / OCR / ASR / vision + fusion table.

    python -m analyst.demo_e2e --age 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyst.ingest.frames import load_pair
from analyst.pipeline import AnalystPipeline
from analyst.schemas import AnalystRunResult

ASSETS = Path(__file__).resolve().parent / "demo_assets"
REPORT = Path(__file__).resolve().parent / "evaluation" / "demo_e2e_report.md"


@dataclass
class Case:
    name: str
    expect: str  # hate | not-hate | any
    overlay: str = ""
    image: Optional[Path] = None
    audio: Optional[Path] = None
    skip_if_missing: bool = False


def default_cases() -> list[Case]:
    return [
        Case("text_clean_gaming", "not-hate", overlay="gg ez noob that was a fun match"),
        Case("text_hate_threat", "hate", overlay="you should kys nobody likes you"),
        Case(
            "ocr_clean_01",
            "not-hate",
            image=ASSETS / "01_clean_gaming.png",
            skip_if_missing=True,
        ),
        Case(
            "ocr_hate_02",
            "hate",
            image=ASSETS / "02_hate_threat.png",
            skip_if_missing=True,
        ),
        Case(
            "ocr_hate_03",
            "hate",
            image=ASSETS / "03_hate_identity.png",
            skip_if_missing=True,
        ),
        Case(
            "ocr_clean_04",
            "not-hate",
            image=ASSETS / "04_benign_chat.png",
            skip_if_missing=True,
        ),
        Case(
            "vision_only_05",
            "any",
            image=ASSETS / "05_vision_only.png",
            skip_if_missing=True,
        ),
        Case(
            "asr_hate_02",
            "hate",
            audio=ASSETS / "02_hate_threat.wav",
            skip_if_missing=True,
        ),
        Case(
            "asr_clean_01",
            "not-hate",
            audio=ASSETS / "01_clean_gaming.wav",
            skip_if_missing=True,
        ),
        Case(
            "multimodal_hate_02",
            "hate",
            image=ASSETS / "02_hate_threat.png",
            audio=ASSETS / "02_hate_threat.wav",
            skip_if_missing=True,
        ),
    ]


@dataclass
class Row:
    case: str
    decision: str
    expect: str
    match: str
    text_s1: float
    vision_s1: float
    fused: Optional[float]
    total_ms: float
    ocr_ms: float
    asr_ms: float
    clip_ms: float
    notes: str
    envelope: Optional[dict[str, Any]] = None
    skipped: str = ""


def _run_case(pipe: AnalystPipeline, case: Case, age: int) -> Row:
    if case.skip_if_missing:
        missing = []
        if case.image is not None and not case.image.is_file():
            missing.append(str(case.image.name))
        if case.audio is not None and not case.audio.is_file():
            missing.append(str(case.audio.name))
        if missing:
            return Row(
                case=case.name,
                decision="skip",
                expect=case.expect,
                match="skip",
                text_s1=0.0,
                vision_s1=0.0,
                fused=None,
                total_ms=0.0,
                ocr_ms=0.0,
                asr_ms=0.0,
                clip_ms=0.0,
                notes="",
                skipped="missing " + ", ".join(missing),
            )

    frame, audio_b = load_pair(case.image, case.audio)
    result: AnalystRunResult = pipe.analyze(
        child_age=age,
        overlay_text=case.overlay,
        image_bytes=frame,
        audio_bytes=audio_b,
        app_exe="Discord.exe",
        app_category="chat",
    )
    fused = None if result.stage2 is None else result.stage2.get("fused")
    if case.expect == "any":
        match = "ok"
    elif result.decision == case.expect:
        match = "ok"
    else:
        match = "MISMATCH"
    env = result.envelope.model_dump() if result.envelope else None
    lat = result.latency_ms or {}
    return Row(
        case=case.name,
        decision=result.decision,
        expect=case.expect,
        match=match,
        text_s1=float(result.stage1.get("text_score", 0.0)),
        vision_s1=float(result.stage1.get("vision_score", 0.0)),
        fused=fused,
        total_ms=float(lat.get("total_ms", 0.0)),
        ocr_ms=float(lat.get("ocr_ms", 0.0)),
        asr_ms=float(lat.get("asr_ms", 0.0)),
        clip_ms=float(lat.get("clip_ms", 0.0)),
        notes=", ".join(result.notes),
        envelope=env,
    )


def _table(rows: list[Row]) -> str:
    header = (
        f"{'case':<22} {'dec':<9} {'exp':<9} {'ok':<8} "
        f"{'t1':>6} {'v1':>6} {'fuse':>6} {'ms':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        if r.skipped:
            lines.append(f"{r.case:<22} skip      {r.expect:<9} skip     {r.skipped}")
            continue
        fused = f"{r.fused:.3f}" if r.fused is not None else "-"
        lines.append(
            f"{r.case:<22} {r.decision:<9} {r.expect:<9} {r.match:<8} "
            f"{r.text_s1:6.3f} {r.vision_s1:6.3f} {fused:>6} {r.total_ms:8.1f}"
        )
    return "\n".join(lines)


def _md_report(rows: list[Row], backends: dict[str, str], age: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Analyst end-to-end demo (Step 5)",
        "",
        f"Generated: {ts}  ",
        f"Persona age: {age}",
        "",
        "## Backends",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for k, v in backends.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += [
        "",
        "## Cases",
        "",
        "| case | decision | expect | match | text_s1 | vision_s1 | fused | total_ms | ocr_ms | asr_ms | clip_ms | notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.skipped:
            lines.append(
                f"| `{r.case}` | skip | {r.expect} | skip |  |  |  |  |  |  |  | {r.skipped} |"
            )
            continue
        fused = "" if r.fused is None else f"{r.fused:.4f}"
        lines.append(
            f"| `{r.case}` | {r.decision} | {r.expect} | {r.match} | "
            f"{r.text_s1:.4f} | {r.vision_s1:.4f} | {fused} | "
            f"{r.total_ms:.1f} | {r.ocr_ms:.1f} | {r.asr_ms:.1f} | {r.clip_ms:.1f} | {r.notes} |"
        )
    envelopes = [r for r in rows if r.envelope]
    if envelopes:
        lines += ["", "## hate.detected envelopes", ""]
        for r in envelopes:
            lines.append(f"### {r.case}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(r.envelope, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C2 Analyst end-to-end demo (Step 5)")
    p.add_argument("--age", type=int, default=10)
    p.add_argument("--no-report", action="store_true", help="Do not write markdown report")
    p.add_argument("--json", action="store_true", help="Print case rows as JSON")
    return p


def run_demo(age: int = 10, write_report: bool = True) -> tuple[list[Row], dict[str, str]]:
    pipe = AnalystPipeline()
    rows = [_run_case(pipe, case, age) for case in default_cases()]
    backends = pipe.backends()
    if write_report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(_md_report(rows, backends, age), encoding="utf-8")
    return rows, backends


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, backends = run_demo(age=args.age, write_report=not args.no_report)
    print("backends:", json.dumps(backends))
    print()
    print(_table(rows))
    print()
    hates = [r for r in rows if r.envelope]
    if hates:
        print("--- hate.detected ---")
        for r in hates:
            print(f"\n# {r.case}")
            print(json.dumps(r.envelope, indent=2))
    if not args.no_report:
        print(f"\nreport: {REPORT}")
    mismatches = [r for r in rows if r.match == "MISMATCH"]
    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, default=str))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
