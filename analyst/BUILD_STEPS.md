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
| **5** | Fusion + end-to-end demo script | Done |
| **5b** | Live capture + SQLite + localhost panel | **Done** |
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

> **One-time network step.** The first transcription downloads
> `Systran/faster-whisper-tiny` (~75 MB) from Hugging Face into
> `~/.cache/huggingface`. After that ASR is fully offline. Pre-warm it before
> an offline demo:
> `python -c "from faster_whisper import WhisperModel; WhisperModel('tiny')"`
>
> OCR needs no download — the RapidOCR wheel ships its ONNX models.

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

## Step 4b — Image meaning via a local VLM (meme path)

`image_fast` scores a CLIP embedding and is deferred until A2, which leaves the
**meme case** uncovered: a picture whose harm is not in any text OCR can reach.
A small local vision-language model closes that gap.

Two findings shaped the design, both measured, both worth knowing before you
trust any output:

### 1. Never send the whole desktop

A bullying poster was embedded in a browser window and both forms were shown to
`lfm2.5-vl-450m`:

| Input | "What is happening?" | "Transcribe the text" |
|---|---|---|
| Full desktop | *"a red square with the text GUARANTEE"* — **invented** | an invented website nav menu — **fabricated** |
| Cropped to the poster | *"yellow text that says nobody likes you, go back to your country"* — correct | `NO BODY LIKES YOU / GO BACK TO YOUR COUNTRY` — correct |

A model this size does not hedge when it cannot see clearly — it invents
confidently, and a fabrication is indistinguishable from a real reading once it
reaches the cascade. `extract/region.py` therefore finds the dominant picture
first (saturation + local contrast on a coarse grid, largest connected blob) and
the branch **declines to ask** when no distinct picture is present. On a plain
text page it correctly returns nothing — OCR already covers that.

### 2. The model reads, the lexicon judges

The VLM is never asked "is this hate". At 450M it cannot follow a strict label
format (it replies `ONLY`), and delegating the verdict would move the definition
of hate out of the auditable lexicon. It answers two questions instead:

* `image_caption` — what the picture means
* `image_text` — what words are written in it

Both are scored by the **same** Stage-1 scorer as OCR and ASR text.

### Running it

```powershell
# LM Studio -> Developer -> Local Server -> Start (port 1234), load the model
$env:ANALYST_VLM_URL   = "http://127.0.0.1:1234/v1"
$env:ANALYST_VLM_MODEL = "lfm2.5-vl-450m"
python -m analyst.serve
```

Unset `ANALYST_VLM_URL` and the branch is inert; the suite passes identically
either way.

**Not gated on escalation.** A meme is exactly the case where Stage 1 sees
nothing, so the vision branch runs whenever a picture is on screen. The cost
control is the crop, not a gate: ~1.7–6.6 s per frame on CPU. This is a
supervision tool, not a real-time one — accuracy is worth the seconds.

**Model choice** (cropped, 512 px, CPU):

| Model | Latency | Reads overlay text? |
|---|---|---|
| `lfm2.5-vl-450m` | ~2–4 s | yes, verbatim — **default** |
| `lfm2-vl-450m` | ~0.4 s | no — paraphrases, loses the slur |
| `google/gemma-4-e4b` | 18–42 s | returned empty in this setup |

**Known limits.** No local VLM tested here reads heavily rotated or distorted
text — `lfm2.5-vl-450m` honestly reports "a small, unreadable word". Word-level
transcription is also imperfect (`NOBODY LIKES YOU` came back as `NOT BACK TO
YOUR COUNTRY GO BACK TO YOUR COUNTRY`), which is fine for lexicon matching but
must not be quoted as ground truth in the thesis.

Everything stays on-device: LM Studio is a localhost server.

---

## Data retention (what the store keeps, and for how long)

Raw frames and audio are **never** written to disk — they are held in a bounded
RAM buffer, analysed, then zeroed. What SQLite keeps is a detection row plus, for
the panel's preview strip, a **downscaled (320 px) and Gaussian-blurred JPEG** —
not the original screenshot.

That preview is still a picture of the screen, so it expires first:

| Data | Default TTL | Env override |
|---|---|---|
| Blurred thumbnail | 24 hours | `ANALYST_THUMB_TTL_H` |
| Detection row (scores, redacted <=200-char OCR/ASR snippet) | 30 days | `ANALYST_RUN_TTL_DAYS` |
| Hard row ceiling | 5000 rows | `ANALYST_MAX_RUNS` |

Pruning runs at panel startup and amortised on insert. The connection sets
`PRAGMA secure_delete=ON` and pruning `VACUUM`s, so cleared thumbnails are
overwritten rather than left recoverable in free pages.

Manual control:

```powershell
curl -X POST http://127.0.0.1:8765/api/retention/prune   # apply policy now
curl -X POST http://127.0.0.1:8765/api/retention/purge   # delete every run
curl http://127.0.0.1:8765/api/retention                 # show current policy
```

OCR/ASR snippets are redacted for `password` / `token` / `api_key` patterns
before storage, but redaction is pattern-based — it will not catch private
messages or names. Treat `analyst/data/analyst.db` as sensitive. It is
gitignored and must never be committed or shared.

---

## Step 5b — Live capture + SQLite + panel

**What we built**

- Temporary C2 capture: `mss` screen + `soundcard` WASAPI loopback (until C1)
- SQLite at `analyst/data/analyst.db` — events, snippets, scores, blurred thumb only
- Local panel: `python -m analyst.serve` → http://127.0.0.1:8765
- Tests: `analyst/tests/test_store_capture.py`

**How to run**

```powershell
pip install fastapi uvicorn python-multipart mss soundcard
python -m unittest analyst.tests.test_store_capture -v
python -m analyst.serve
# open http://127.0.0.1:8765 → Start
```

Privacy: raw frames/audio stay in RAM then wipe. Panel is localhost-only.

---

## Step 6 — Fine-tune + Dataset C (after demo)

- Logistic probe on CLIP embeddings. Until it exists `image_fast` reports
  `calibrated = False` and is excluded from the fused score — measured, it
  returns 0.324-0.393 for every image, which is noise.
- Fusion MLP vs this rule-based head
- Do not block the live demo on this step

**The pretrained baseline already exists.** `analyst/evaluation/` benchmarks the
current cascade against Jigsaw, Davidson and Berkeley with content-derived
train/dev/test splits, so a fine-tuned model has something honest to beat:

```powershell
python -m analyst.evaluation.benchmark --corpus berkeley:test --scorer cascade
python -m analyst.evaluation.corpora     # provenance + class balance
```

Two rules when working on this step:

1. Tune on `:train`, sanity-check on `:dev`, report on `:test`. Splits come from
   a hash of the text, so they survive re-sampling.
2. `jigsaw:*` is **contaminated** — `unitary/toxic-bert` was trained on it.
   Never quote it as a generalisation result for a stack containing that head.
