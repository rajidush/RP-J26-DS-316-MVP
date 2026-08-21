# C2 Analyst — build steps (do in order)

Source of truth: Engineering Plan v0.1 §4.  
Rule: **each step must leave a working CLI** (`python -m analyst --text "..."`). No dead ends.

| Step | Goal | Status |
|---|---|---|
| **0** | Clean `main` + `analyst/` skeleton | Done |
| **1** | Pretrained Stage-1 text + demo assets | **This branch** |
| **2** | OCR path verified on demo screenshots | Next |
| **3** | ASR path (Whisper) on short clips | Later |
| **4** | Pretrained CLIP / image_fast stub → real | Later |
| **5** | Fusion + `hate.detected` demo script | Later |
| **6** | After demo: fine-tune + Dataset C (research) | After demo |

---

## Step 1 — Pretrained Stage-1 text (current)

**What we build**

- Load a **public pretrained** toxicity/hate classifier on CPU (HuggingFace).
- Always keep **lexicon** as backup / boost (gaming slang stays low).
- Final Stage-1 text score = `max(lexicon, pretrained)`.
- If `transformers` or model download fails → **lexicon only** (demo still works).

**Default model**

- `martin-ha/toxic-comment-model` (DistilBERT, toxic vs non-toxic)
- Override: env `ANALYST_TEXT_MODEL=<hf-id>`

**How to run**

```powershell
pip install -r analyst/requirements.txt
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
# first run downloads the model (~250MB), then works offline from cache

python -m analyst --text "you should kys" --age 10
python -m analyst --text "gg ez noob" --age 12
python -m analyst --replay analyst/demo_assets --age 10
```

**Done means**

- [x] Pretrained plug in `stage1/text_fast.py`
- [x] Lexicon never removed
- [x] `demo_assets/` with clean + hate screenshots
- [x] Tests still pass without GPU / without transformers installed
- [x] `BUILD_STEPS.md` roadmap for Steps 2–6
