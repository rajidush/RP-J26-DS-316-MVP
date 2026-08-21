# C2 Analyst — build steps (do in order)

Source of truth: Engineering Plan v0.1 §4.  
Rule: **each step must leave a working CLI** (`python -m analyst ...`). No dead ends.

| Step | Goal | Status |
|---|---|---|
| **0** | Clean `main` + `analyst/` skeleton | Done |
| **1** | Pretrained Stage-1 text + demo assets | Done |
| **2** | OCR path verified on demo screenshots | **Done (this session)** |
| **3** | ASR path (Whisper) on short clips | **Next** |
| **4** | Pretrained CLIP / image_fast stub → real | Later |
| **5** | Fusion + `hate.detected` demo script | Later |
| **6** | After demo: fine-tune + Dataset C (research) | After demo |

---

## Step 1 — Pretrained Stage-1 text

- Lexicon always on; optional HF model `martin-ha/toxic-comment-model`
- Score = `max(lexicon, pretrained)`
- Demo PNGs in `analyst/demo_assets/`

```powershell
python -m analyst --text "gg ez noob" --age 12
```

---

## Step 2 — OCR on screenshots

**What we built**

- High-contrast synthetic chat PNGs (OCR-friendly)
- RapidOCR extract → Stage-1 text → `hate.detected` / not-hate
- Automated checks: `analyst/tests/test_ocr_demo.py`
- Notes: `analyst/evaluation/baseline_extract.md`

**How to run**

```powershell
pip install rapidocr-onnxruntime   # if needed
python -m analyst.demo_assets.generate
python -m unittest analyst.tests.test_ocr_demo -v
python -m analyst --replay analyst/demo_assets --age 10
```

**Done means**

- [x] OCR recovers key tokens from each demo PNG
- [x] Hate screenshot → decision `hate` with non-empty `ocr_text`
- [x] Clean gaming screenshot → `not-hate`
- [x] Missing OCR package still allows `--text` path (no crash)

---

## Step 3 — ASR (next)

- Whisper tiny/base on short `.wav` next to demo images
- Same cascade: transcript → Stage-1 text
- Keep working if Whisper missing
