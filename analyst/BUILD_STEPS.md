# C2 Analyst — build steps (do in order)

Source of truth: Engineering Plan v0.1 §4.  
Rule: **each step must leave a working CLI**. No dead ends.

| Step | Goal | Status |
|---|---|---|
| **0** | Clean `main` + `analyst/` skeleton | Done |
| **1** | Pretrained Stage-1 text + demo assets | Done |
| **2** | OCR path verified on demo screenshots | Done |
| **3** | ASR path (Whisper) on short clips | Done |
| **4** | Pretrained CLIP / image_fast zero-shot | Done |
| **5** | Fusion + end-to-end demo script | **Done (this session)** |
| **6** | After demo: fine-tune + Dataset C (research) | After demo |

---

## Step 1 — Pretrained Stage-1 text

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

```powershell
python -m analyst.demo_assets.generate_audio
python -m unittest analyst.tests.test_asr_demo -v
python -m analyst --audio analyst/demo_assets/02_hate_threat.wav --age 10
```

---

## Step 4 — CLIP / image_fast

```powershell
python -m analyst.demo_assets.generate
python -m unittest analyst.tests.test_image_fast -v
```

Without torch, backends report `clip=deferred` — text path still works.

---

## Step 5 — Fusion + end-to-end demo

**What we built**

- Rule-based late fusion (0.6 text + 0.4 vision, meme bump when both mid)
- `python -m analyst.demo_e2e` runs a fixed matrix: text, OCR, ASR, vision-only, multimodal
- Prints a latency table + `hate.detected` JSON
- Writes `analyst/evaluation/demo_e2e_report.md` each run
- Tests: `analyst/tests/test_demo_e2e.py`

**How to run**

```powershell
python -m analyst.demo_assets.generate
python -m analyst.demo_assets.generate_audio
python -m unittest analyst.tests.test_demo_e2e -v
python -m analyst.demo_e2e --age 10
```

`vision_only_05` is `expect=any` until the trained probe (Step 6).

**Done means**

- [x] Text hate/clean match in the demo table
- [x] `hate.detected` envelope includes `child_safe_summary`
- [x] Latency columns: ocr / asr / clip / total
- [x] Markdown report written under `analyst/evaluation/`

---

## Step 6 — Fine-tune + Dataset C (after demo)

- Logistic probe on CLIP embeddings
- Fusion MLP vs this rule-based head
- Do not block the live demo on this step
