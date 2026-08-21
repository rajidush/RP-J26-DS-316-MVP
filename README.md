# J26-DS-316 — Component 2: The Analyst

**Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety**

| Field | Value |
|---|---|
| **Project** | J26-DS-316 · IT4010 · SLIIT CoEAI · Data Science |
| **This repo focus** | **C2 — Hate-Speech Detection Module (The Analyst)** |
| **Owner** | Liyanage D. S. · IT23209152 |
| **Architecture source** | Engineering Plan v0.1 (19 Aug 2026) |
| **Runtime** | Local Python process, **CPU only**, **zero cloud** |

This repository is being built as the **correct production path for Component 2**, not as a browser-based parental-control product. The Analyst runs on-device, consumes screen/audio triggers (from C1 later, or replay/files now), and emits the team contract `hate.detected`.

---

## Correct system vision (hub-and-spoke)

Per the engineering plan, the full Guardian system is **not** a Next.js app that “detects hate”. It is four processes around a local bus:

```
C1 Sentinel   — OS activity + Capture Service (owns screen frames)
C2 Analyst    — THIS COMPONENT (OCR/ASR + cascade + fusion → hate.detected)
C3 Educator   — Socratic SLM (loads on demand)
C4 Recorder   — SQLite + XAI parent dashboard

Bus: ZeroMQ (later). Until C1/bus exist, C2 uses --replay / CLI / file ingest.
```

**Hard rules we follow**

- Zero outbound network at inference time  
- CPU only (no VRAM / no GPU required for demo numbers)  
- Application-independent (screen + speaker audio, not URL blocklists)  
- Raw frames/audio **never written to disk** — RAM hold → analyse → delete  
- C2 **consumes** frames; C1 owns capture (we do not pretend the web UI is the monitor)

---

## What C2 does

On a trigger:

1. Extract text (OCR) and speech (ASR / loopback later)  
2. **Stage 1** (cheap, high recall): lexicon + fast text (+ image probe when ready)  
3. Most content stops here as **not-hate**  
4. **Stage 2** (precision): stronger text + cross-modal fusion  
5. Emit **`hate.detected`** or **`hate.cleared`** with child-safe summary (never echo raw hate to the child)

### Decision flow

```
frame / audio / overlay text
        │
        ├─ OCR + ASR (parallel)
        ├─ CLIP embed (when model registered)
        │
        ▼
 Stage 1: lexicon ‖ text_fast ‖ image_fast
        │
   score < θ1 ──► clear (stopped_at_stage1)
        │
   score ≥ θ1
        ▼
 Stage 2: text_full + fusion(image, text)
        │
   score ≥ θ2(persona) ──► hate.detected → C3 + C4
        │
        └── else ──► hate.cleared (FP telemetry)
```

Persona thresholds (θ2): ages 8–10 → 0.55 · 11–13 → 0.65 · 14–15 → 0.75.

---

## Repository layout (C2)

```
analyst/                 ← production Analyst engine (ONLY C2 path — start here)
├── ingest/              frame/audio loaders (replay until C1 bus)
├── extract/             ocr.py · asr.py · embed.py (CLIP plug)
├── stage1/              lexicon · text_fast · image_fast
├── stage2/              fusion (+ text_full plug)
├── decide.py            hate.detected / hate.cleared contract
├── pipeline.py          cascade
├── buffer.py            RAM-only media, wiped after each run
├── schemas.py           Engineering Plan message shapes
├── main.py              CLI: python -m analyst
├── models/              ONNX weights (gitignored)
├── evaluation/          benchmark reports (A1/A4)
└── tests/

offline_backend/         C3 Educator + C4 dashboard demo API (teammates) — not C2
app/                     Shared Next.js sandbox for C3/C4 demo UI — not C2 detection
```

**Cleanup note:** The old duplicate `offline_backend/analyst/` and web `AnalystPanel` were removed so only root `analyst/` owns hate-speech detection. Socratic (`socratic_agent.py`) and parent dashboard endpoints were left untouched.
---

## Current status (honest)

| Piece | Status |
|---|---|
| Cascade + RAM buffer + delete | **Working** |
| Lexicon stage-1 (threat / bullying / gaming benign) | **Working** |
| `hate.detected` payload + child_safe_summary | **Working** |
| CLI + `--replay` | **Working** |
| OCR (RapidOCR) / ASR (Whisper tiny) | **Optional plugs** — install when needed |
| CLIP / image_fast / text_full ONNX | **Deferred plugs** — Milestone A2/A3 |
| ZeroMQ bus + C1 frame subscribe | **Later integration** |
| Next.js / offline_backend web demo | **Teammate C3/C4 sandbox** — not used for C2 detection |

---

## Quick start

```powershell
# From repo root
cd analyst
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional (recommended for screen/voice demos)
pip install rapidocr-onnxruntime faster-whisper numpy

cd ..
python -m analyst --text "you should kys" --age 10
python -m analyst --text "gg ez noob" --age 12
python -m analyst --image .\path\to\frame.png --age 10 --json
python -m analyst --replay .\path\to\frames_folder --age 10
```

### Tests

```powershell
python -m unittest analyst.tests.test_pipeline -v
```

### Example `hate.detected` (shape)

```json
{
  "score": 0.88,
  "category": "threat",
  "stage": 2,
  "modalities": { "text": 1.0, "image": 0.0, "audio": 0.0 },
  "evidence": { "ocr_snippet": "", "lexicon_hits": ["kys"] },
  "child_safe_summary": "Someone in this chat is using language that sounds like a threat or harm.",
  "recommended_action": "block",
  "persona_threshold": 0.55
}
```

---

## Build milestones (from Engineering Plan §4.4)

| Milestone | Focus | Done means |
|---|---|---|
| **A1** | Extraction + replay CLI | OCR/ASR/CLIP plugs; latency notes in `evaluation/` |
| **A2** | Stage 1 | Lexicon + `text_fast` ONNX + `image_fast`; θ1 tuned |
| **A3** | Stage 2 research | Dataset C + `text_full` ablations + fusion (main thesis) |
| **A4** | Eval + integrate | CPU report; wire `hate.detected` to C3/C4 |

**Now:** A1 foundation is in place (cascade, contracts, CLI, lexicon). Next: optional OCR/ASR install, then A2 ONNX training.

---

## Integration with the rest of the team

| Component | How C2 talks to it |
|---|---|
| **C1 Sentinel** | Subscribes to `frame.captured` / activity (bus). Until then: `--replay` |
| **C3 Educator** | Publishes `hate.detected` with `child_safe_summary` (no raw hate text) |
| **C4 Recorder** | Same event + optional `hate.cleared` for FP analytics |
| **Their web UI** | May call a thin local HTTP adapter later — UI never owns detection |

Do **not** put detection models inside the browser. Do **not** store child frames in the web app.

---

## Privacy

- Frames/audio: process RAM only → wiped after the run  
- Logs: scores + redacted snippets ≤ 200 chars  
- English classifiers in v1; Sinhala/Singlish = future work (do not promise to panel)

---

## References (component)

- Engineering Plan v0.1 — J26-DS-316 (system contracts & C2 build guide)  
- TAF V2.2 — Topic Assessment Form  
- Hateful Memes · HateXplain · CONDA · Jigsaw (training/eval sets for A2–A4)

---

*Owners update this README when milestones land. Engineering truth for C2 lives in `analyst/` + the Engineering Plan.*
