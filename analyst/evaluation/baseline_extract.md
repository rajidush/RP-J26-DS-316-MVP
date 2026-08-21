# Step 2–3 — Extraction baseline

## OCR (Step 2)

Measured with RapidOCR on synthetic `analyst/demo_assets/*.png`.

| Asset | Expect |
|---|---|
| `01_clean_gaming.png` | clean → not-hate |
| `02_hate_threat.png` | hate via OCR |
| `03_hate_identity.png` | hate via OCR |
| `04_benign_chat.png` | not-hate |

## ASR (Step 3)

Synthetic WAVs from Windows SAPI (`generate_audio.py`). Backend: `faster-whisper` tiny, CPU int8.

| Asset | Expect |
|---|---|
| `02_hate_threat.wav` | transcript contains kill/yourself → hate |
| Injected ASR tests | work even if Whisper missing |

```powershell
python -m analyst.demo_assets.generate_audio
python -m unittest analyst.tests.test_asr_demo -v
python -m analyst --audio analyst/demo_assets/02_hate_threat.wav --age 10
```

## Vision / CLIP (Step 4)

| Asset / path | Expect |
|---|---|
| Injected `image_fast` | vision-only → hate |
| `05_vision_only.png` | no chat text; CLIP optional (may stay deferred) |
| Missing torch | `clip=deferred`, text/OCR/ASR still work |

```powershell
python -m analyst.demo_assets.generate
python -m unittest analyst.tests.test_image_fast -v
```

## End-to-end demo (Step 5)

```powershell
python -m analyst.demo_e2e --age 10
```

Writes `analyst/evaluation/demo_e2e_report.md` (latency table + `hate.detected` JSON).
