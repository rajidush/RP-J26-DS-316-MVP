# Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety

**Project ID:** J26-DS-316  
**Module:** IT4010 Research Project (2026)  
**Research Cluster:** CoEAI  
**Specialization:** Data Science  
**Industry Verticals:** Cybersecurity, EdTech, Digital Health & Wellness

This repository is the MVP for the **Hate-Speech Detection Module (The Analyst)** — Component 2 of a privacy-first, Zero-Cloud Multi-Agent Edge AI parental control system.

| Field | Detail |
| --- | --- |
| Component owner | Liyanage D. S. |
| Registration no. | IT23209152 |
| Component | Hate-Speech Detection Module (The Analyst) |
| Target users | Children aged 8–15 |
| Deployment constraint | Fully on-device, CPU-only, no cloud, no GPU required |

---

## Research Problem

Legacy parental-control tools (Qustodio, Bark, Microsoft Family Safety, Google Family Link) rely on cloud processing, URL/app-name blocklists, and text-only filters. They fail in three ways that this component addresses:

1. **They cannot see the screen.** Hate speech in games, memes, Discord overlays, and unsupported apps is invisible to URL and account-based filters.
2. **They check image and text separately.** A harmless image plus harmless caption can become toxic only when combined (hateful memes). Unimodal models miss that case.
3. **They send a child's private screen, chat, and voice data to the cloud**, conflicting with COPPA / GDPR-K data-minimization rules and excluding families without a stable internet connection.

This module detects hate speech **on the child's device**, from **what is actually on screen and in voice chat**, and emits a structured risk result that later components use for Socratic intervention and parent analytics.

---

## System Context

The full research system uses a **Chained Activation** cascade: heavy models run only after a cheaper upstream trigger. Four agents share local IPC; no telemetry leaves the device.

```
Component 1  Sentinel          OS-level monitoring + violence detection
        |  trigger (risk event)
        v
Component 2  Analyst   <--- this MVP / this member
        |  structured hate-speech result (score, category)
        v
Component 3  Educator          Socratic SLM intervention
        |
        v
Component 4  Record Keeper     Behavioural profiling + XAI parent reports
```

**This component's contract**

- **Input:** a trigger from Component 1 (screen/audio capture is not continuous).
- **Output:** a structured payload `{ risk_score, category, hate | not-hate }` sent to Component 3 (intervention) and Component 4 (analytics).

---

## Hate-Speech Detection Module (The Analyst)

The Analyst uses **cascaded inference** and **cross-modal fusion**. A cheap first stage screens most frames as harmless. Only flagged content wakes the heavier vision and text models. Voice chat is transcribed and evaluated on the same text path as on-screen OCR.

### Why this is different

Existing tools mainly check text or known-unsafe URLs and cannot inspect actual on-screen content. They also score images and text independently, so they miss memes where each modality looks safe but the combination is hateful. Generic toxicity models further fail on children's slang and gaming language.

This component:

- Fuses **on-screen image + OCR text** so hate that exists only in the combination is still caught.
- Transcribes **voice chat** and routes it through the same text classifier.
- Runs a **lightweight first stage** (text + frame in parallel) so a harmful image with **no text** still gets a closer look.
- Fine-tunes the text model on **hate speech, cyberbullying, children's slang, and gaming language**.
- Runs the full pipeline **offline on an ordinary laptop CPU** — no GPU, no cloud.

### Cascaded inference (resource model)

```
Trigger from Sentinel
        |
        v
  Capture frame + mic (Phase 1)
        |
        +-- OCR + ASR  -->  fast text check  --+
        |                                      |  either flags risk
        +-- lightweight frame check  ----------+
                        |
            no risk -->  drop (not-hate)
            risk    -->  vision model + fine-tuned text classifier
                        |
                        v
              cross-modal fusion --> hate / not-hate
                        |
                        v
              { risk_score, category } --> Educator + Record Keeper
```

---

## Component plan (IT23209152)

This is the execution plan for Component 2. Work is sequential: capture is idle until Component 1 fires; Stage 1 must reject most content before Stage 2 is allowed to run; fine-tuning and CPU evaluation (Phase 4) are the primary research output; Phase 5 is the integration contract with the rest of the team.

### Objective

Detect hate speech and cyberbullying from **on-screen image + extracted text + spoken audio**, on a consumer Windows laptop, **CPU only**, with **zero data transit**. Emit a typed risk result that Component 3 can intercept on and Component 4 can log.

### Constraints (non-negotiable)

- No cloud APIs for OCR, ASR, classification, or logging.
- Models must run on a standard family laptop without a discrete GPU.
- Capture is **trigger-based**, never a continuous recorder.
- A harmful image with **no text** must still be eligible for Stage 2.
- A meme where image and text are each “safe” but toxic **together** must be eligible for a hate decision.

### Phase 1 — Screen and audio capture

**Goal:** Take one screen frame and a short microphone window only when the Sentinel sends a trigger, so idle CPU stays near zero.

| Item | Plan |
| --- | --- |
| Trigger | Local IPC / HTTP event from Component 1 (`risk_event`). No polling loop of our own. |
| Screen | One screenshot at trigger time, held in a **transient buffer** until this trigger’s analysis finishes. |
| Audio | Short rolling buffer (about 2–4 s) held with the same lifetime as the frame. |
| Privacy | Buffer lives in the local Analyst process (RAM). The web app never receives the image or audio. After the decision, both are deleted. Nothing is uploaded. |
| Deliverable | `capture_on_trigger(event) -> {frame, audio_pcm, timestamp, source_app_hint}` |

#### Transient buffer (hold → analyse → delete)

Frames **must** exist until the models have finished with that trigger. They must **not** exist after that. This is a working copy, not a gallery and not a parent evidence archive.

```
trigger
   → put frame + audio in RAM slot (keyed by trigger_id)
   → Stage 1 (and Stage 2 if escalated) read that slot
   → write JSON decision { risk_score, category, hate|not-hate }
   → delete slot  (success, not-hate, error, or timeout — always)
```

| Rule | Why |
| --- | --- |
| One slot per trigger, overwrite or bound the queue (e.g. max 1–2 pending) | A child laptop must not queue minutes of screenshots |
| Store in process RAM (`bytes` / NumPy / PIL). Do not write `screenshot.png` under the web `public/` folder, Downloads, or IndexedDB | Disk copies survive the demo and break zero-retention |
| Delete in a `finally` block | A model crash must not leave the frame behind |
| Delete on **not-hate** as well as on **hate** | Clean content is still a child’s screen |
| After delete, only the JSON score may leave this process | Component 3/4 and the web UI need the verdict, not the pixels |

If RAM is tight and a temp file is unavoidable: write to a process-private temp path, `flush`, analyse, then `unlink` in `finally`. Never serve that path over HTTP.

**Exit criteria:** CPU usage with no trigger is negligible; one trigger produces one frame + one audio clip within a hard latency budget (target &lt; 200 ms to have buffers ready); after each decision the buffer for that `trigger_id` is empty.

### Phase 2 — First-level check (cheap cascade)

**Goal:** Clear harmless content quickly. Run two small checks **in parallel** so a risk is caught even when there is no text.

| Stream | Method | Role |
| --- | --- | --- |
| Text | OCR on the frame (Tesseract or EasyOCR, CPU) + ASR on the clip (Whisper-tiny / faster-whisper `tiny` or `base.en`) | Recover chat, captions, usernames, meme text, voice-chat abuse |
| Fast text screen | Keyword / lexicon + a tiny toxicity head (quantized MiniLM or DistilBERT-class, INT8) | High-recall, low-cost flag on recovered text |
| Fast frame screen | Lightweight image probe (tiny CNN / MobileNet-class, or CLIP-image embedding vs a small toxic-visual bank) | Flags memes, slurs-as-image, hate symbols when OCR returns empty |

**Decision rule**

- Both streams below a low threshold → **not-hate**, stop. Heavy models do not load.
- Either stream above threshold → wake Phase 3.

**Deliverable:** `stage1(frame, audio) -> {ocr_text, transcript, text_score, frame_score, escalate: bool}`

**Exit criteria:** Most clean frames never reach Phase 3; empty-text hate images still escalate via the frame check.

### Phase 3 — Cross-modal analysis

**Goal:** Decide whether the **combination** of image meaning and text is harmful, including cases where each modality alone looks safe.

| Branch | Plan |
| --- | --- |
| Vision | Compact vision encoder (CLIP ViT-B/32 distilled, or a small ViT ONNX) produces an image-intent embedding / toxicity score. |
| Text | Fine-tuned hate/cyberbullying classifier from Phase 4 on `ocr_text + transcript`. |
| Fusion | Late fusion: compare the two scores and a joint feature (concatenated embeddings or a small CPU MLP). Hate if the fused score exceeds the operating threshold, even if unimodal scores are sub-threshold. |
| Categories | At minimum: `hate_speech`, `cyberbullying`, `identity_attack`, `threat`, `none`. |

This is the meme case from the TAF: safe picture + safe caption → toxic only when read together.

**Deliverable:** `stage2(frame, ocr_text, transcript) -> {decision: hate|not-hate, risk_score, category, modality_votes}`

**Exit criteria:** Documented fusion rule; at least one held-out hateful-meme set where unimodal classifiers fail and fusion succeeds.

### Phase 4 — Fine-tuning and evaluation (main research output)

**Goal:** Specialize the text model for children's slang and gaming language, then prove it is accurate and cheap enough on CPU-only hardware compared with pretrained baselines.

**Data (scientifically published sets only; no live child data without ethics clearance)**

- Hate speech / toxicity: HateXplain, Davidson, OLID / OffensEval.
- Cyberbullying: publicly released cyberbullying corpora used in the literature.
- Multimodal hold-out: Hateful Memes (Facebook / Kiela et al.) for the image+text case.
- Domain adaptation: curated slang / gaming-language lexicon and labelled examples (gg, ez, slur variants, Discord/Twitch chat style) mixed into fine-tuning, not as the only train set.

**Training**

- Start from a small pretrained encoder (MiniLM / DistilBERT / equivalent).
- Fine-tune for binary hate / not-hate plus coarse category.
- Export ONNX or GGUF-class quantized weights for CPU inference.

**Evaluation protocol (must be reported)**

| Metric | Why it matters |
| --- | --- |
| Accuracy, Precision, Recall, F1 | Core detection quality |
| False-positive rate | Avoids punishing the child for slang / trash talk that is not hate |
| Latency (ms, CPU, single thread and modest multithread) | Edge feasibility |
| Peak RAM (MB) | Fits a family laptop beside Components 1, 3, 4 |
| Fine-tuned vs pretrained (same backbone, no extra data) | Isolates the slang/gaming adaptation gain |
| Unimodal vs fused (meme split) | Isolates the cross-modal claim |

**Hardware for the report:** one consumer Windows laptop, CPU only, no discrete GPU used for the published numbers.

**Deliverable:** evaluation table + model card (dataset mix, threshold, FPR operating point). This table is the main research output of Component 2.

### Phase 5 — Alert output

**Goal:** Hand a structured result to Component 3 (Socratic intercept) and Component 4 (analytics). No raw screenshot or audio leaves this process.

```json
{
  "component": "analyst",
  "threat_type": "hate_speech",
  "decision": "hate",
  "risk_score": 0.92,
  "category": "cyberbullying",
  "threshold": 0.85,
  "child_age": 10,
  "source": { "ocr": true, "asr": true, "vision": true },
  "session_hint": "trigger-uuid-from-sentinel"
}
```

- Score **> 0.85** (current MVP gate) → Component 3 starts a `hate_speech` dialogue.
- The same record is appended locally for Component 4. Image/audio blobs are not attached.

**Deliverable:** stable schema consumed by `/api/perception/trigger` (`hate_speech_score`, `threat_type`) and by the parent dashboard counters.

### Work sequence

| Order | Work | Depends on | Status in this MVP |
| --- | --- | --- | --- |
| 1 | Agree IPC trigger + output JSON with Components 1, 3, 4 | Team contract | Live `hate_speech_score` + `/api/analyst/analyze` |
| 2 | Phase 5 stub in the sandbox (slider / preset → intercept → parent log) | Schema | Done |
| 3 | Phase 1 capture-on-trigger | Component 1 event | RAM buffer + optional server screenshot; media deleted after each run |
| 4 | Phase 2 OCR + ASR + dual fast checks | Phase 1 | OCR live (RapidOCR). ASR: Whisper tiny on CPU; voice clip or 6s mic record |
| 5 | Phase 4 dataset mix, fine-tune, CPU eval vs pretrained | Public datasets | Slot ready: `models/text_hate.onnx` |
| 6 | Phase 3 vision branch + fusion, Hateful Memes hold-out | Phases 2 and 4 | Fusion rule live; ONNX deferred: `models/vision_stage1.onnx` |
| 7 | Replace the sandbox slider with live `risk_score` from the Analyst | Phases 1–5 | Analyst panel writes the slider, then can intercept |

### Success criteria for this component

1. Harmless content is dropped at Stage 1 in the majority of triggers (cascade actually saves CPU).
2. Hate in **text-only**, **image-only**, and **meme (joint)** conditions is detected.
3. Spoken abuse is classified after ASR on the same text model.
4. Fine-tuned model beats the pretrained baseline on slang/gaming language without a large FPR increase.
5. End-to-end inference (Stage 1 always; Stage 2 when escalated) meets the CPU/RAM budget on the test laptop.
6. Component 3 receives `hate_speech` and Component 4 can chart it — already shown by this MVP's preset path.

---

## What this MVP demonstrates

The current MVP runs the Analyst cascade locally. OCR, Whisper, and vision ONNX are **plugs**: if they are missing, overlay/chat text still scores and the API stays up.

| Capability | Status in this MVP |
| --- | --- |
| Hate-speech risk score as a perception vector | Implemented (`hate_speech_score`, gate **> 0.85**) |
| Live Analyst cascade (hold → analyse → delete) | Implemented (`POST /api/analyst/analyze`) |
| Lexicon text scorer (slang/threat; gaming trash talk stays below gate) | Implemented |
| OCR / Whisper / vision ONNX | OCR live. Whisper tiny installed for voice. Vision ONNX still deferred |
| Hate-speech preset and parent-dashboard telemetry | Implemented |
| Threshold breach → local Socratic intercept (Component 3 stub) | Implemented via FastAPI + LM Studio |
| Zero data transit (local FastAPI, no cloud LLM) | Implemented |
| Fine-tuned MiniLM vs pretrained CPU eval | Phase 4 — drop weights into `models/text_hate.onnx` when ready |

In the Prototype Sandbox:

1. Use **Analyst — Hate Speech**: paste chat, upload a frame, or grab the server screen.
2. Click **Analyse locally**. The frame is wiped from RAM; only JSON remains.
3. If the decision is **hate**, click **Send score to Socratic intercept** (or use the sliders as before).
4. The Educator overlay opens a local Socratic dialogue when the gate is breached.

---

## Repository layout

```
RP-J26-DS-316-MVP/
├── app/
│   ├── page.tsx              # Prototype Sandbox
│   ├── analyst-panel.tsx     # Hate-speech Analyst UI (JSON only; no frame store)
│   ├── parent/page.tsx       # Parent dashboard (hate-speech analytics view)
│   └── api/socratic/route.ts # Cloud path disabled (410) — offline-only
├── offline_backend/
│   ├── main.py               # FastAPI: analyst + perception + dialogue
│   ├── analyst/              # Component 2 cascade
│   ├── models/               # Optional ONNX (gitignored weights)
│   ├── socratic_agent.py     # Deterministic Socratic state machine (Component 3 stub)
│   └── requirements.txt
└── README.md                 # This file
```

Perception payload consumed by the backend:

```json
{
  "violence_score": 0.05,
  "hate_speech_score": 0.92,
  "adult_content_score": 0.12,
  "child_age": 10
}
```

If `hate_speech_score` is the highest vector and exceeds `0.85`, the system opens a `hate_speech` session and returns a structured intercept to the UI.

---

## How to run

### Prerequisites

- Node.js 20+
- Python 3.8–3.11 (a local `offline_backend/venv` is fine)
- [LM Studio](https://lmstudio.ai/) with a small instruct model loaded (for example `Llama-3.2-1B-Instruct`) and the local server started on `http://localhost:1234`

### 1. Python backend

```bash
cd offline_backend
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
python main.py
```

Backend: [http://127.0.0.1:8001](http://127.0.0.1:8001) (UI reads `app/lib/backend.ts`; use 8000 if that port is free)

### 2. Frontend

From the repository root:

```bash
npm install
npm run dev
```

UI: [http://localhost:3000](http://localhost:3000)  
Parent dashboard: [http://localhost:3000/parent](http://localhost:3000/parent)

### 3. Exercise the hate-speech path

Live Analyst (no models required):

1. Open the Prototype Sandbox.
2. In **Analyst — Hate Speech**, paste chat such as `you should kys` (hate) or `gg ez noob` (must stay clean).
3. Click **Analyse locally**. Confirm `media deleted` and that the Hate Speech slider updates.
4. If the decision is hate, click **Send score to Socratic intercept**.

Slider fallback (still works):

1. Click **Hate Speech** (sets `hate_speech_score = 0.92`) or drag the slider above `0.85`.
2. Trigger interception.
3. Open the Parent Dashboard for hate-speech counts.

Without LM Studio, Analyst scoring and the perception trigger still run; dialogue turns need the local SLM.

Optional later: `pip install -r requirements-analyst.txt` then drop ONNX files into `offline_backend/models/`. The same endpoints keep working.

---

## Alignment

| SDG | How this component contributes |
| --- | --- |
| SDG 3 Good Health & Well-being | Intercepts toxic / bullying content before it reaches the child unchallenged |
| SDG 4 Quality Education | Hands a typed hate-speech event to the Socratic agent instead of silent blocking |
| SDG 9 Industry, Innovation & Infrastructure | Cascaded, CPU-only models on ordinary family laptops |
| SDG 16 Peace, Justice & Strong Institutions | Zero-cloud capture and inference; screen/audio never leave the device |

---

## References (component-specific)

1. Jevremovic, A., Veinovic, M., Cabarkapa, M., et al. “Keeping Children Safe Online with Limited Resources: Analyzing What Is Seen and Heard.” *IEEE Access* 9 (2021): 132723–32.
2. Gomez, R., Gibert, J., Gomez, L., and Karatzas, D. “Exploring Hate Speech Detection in Multimodal Publications.” *IEEE WACV*, 2020.
3. Arya, G., et al. “Multimodal Hate Speech Detection in Memes Using Contrastive Language-Image Pre-Training.” *IEEE Access* 12 (2024).
4. Kiela, D., et al. “The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes.” *NeurIPS*, 2020.
5. Wang, X., and Jia, W. “Optimizing Edge AI: A Comprehensive Survey on Data, Model, and System Strategies.” arXiv:2501.03265, 2025.
6. Sun, R., Zheng, Y., Xiong, Z., et al. “More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection.” 2026.

Related team components: Zero-Trust process / violence monitoring (Dinethra, IT23377844); Socratic pedagogical agent (Hettige R. D., IT23155466); behavioural profiling with XAI (Sandunika H.A.K.H., IT23135116).
