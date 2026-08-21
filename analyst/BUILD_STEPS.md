# C2 Analyst — build steps (do in order)

Source of truth: Engineering Plan v0.1 §4.  
Rule: **each step must leave a working CLI**. No dead ends.

| Step | Goal | Status |
|---|---|---|
| **0** | Clean `main` + `analyst/` skeleton | Done |
| **1** | Pretrained Stage-1 text + demo assets | Done |
| **2** | OCR path verified on demo screenshots | Done |
| **3** | ASR path (Whisper) on short clips | **Done (this session)** |
| **4** | Pretrained CLIP / image_fast stub → real | **Next** |
| **5** | Fusion + end-to-end demo script | Later |
| **6** | After demo: fine-tune + Dataset C (research) | After demo |

---

## Step 1 — Pretrained Stage-1 text

Lexicon + optional HF `martin-ha/toxic-comment-model`.

```powershell
python -m analyst --text "gg ez noob" --age 12
```

---

## Step 2 — OCR on screenshots

```powershell
python -m analyst.demo_assets.generate
python -m unittest analyst.tests.test_ocr_demo -v
python -m analyst --replay analyst/demo_assets --age 10
```

---

## Step 3 — ASR (Whisper)

**What we built**

- Windows SAPI synthetic WAVs (`generate_audio.py`) — no real child voice
- `faster-whisper` tiny on CPU; missing package → skip audio, no crash
- Same cascade: transcript → Stage-1 text → `hate.detected`
- Tests: `analyst/tests/test_asr_demo.py`

**How to run**

```powershell
pip install faster-whisper
python -m analyst.demo_assets.generate_audio
python -m unittest analyst.tests.test_asr_demo -v
python -m analyst --audio analyst/demo_assets/02_hate_threat.wav --age 10
```

Replay still prefers matching `.wav` next to each PNG if present.

**Done means**

- [x] Hate WAV → non-empty transcript → decision `hate`
- [x] Injected ASR path works without Whisper installed
- [x] Media wiped after audio-only analyse

---

## Step 4 — CLIP / image_fast (next)

- Pretrained MobileCLIP or CLIP ONNX for image branch
- Empty-text harmful image can escalate without OCR
