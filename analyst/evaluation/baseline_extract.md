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
