# Demo assets (synthetic)

No real child data.

**Images (OCR):**
```powershell
python -m analyst.demo_assets.generate
```

**Audio (ASR, Windows SAPI TTS):**
```powershell
python -m analyst.demo_assets.generate_audio
```

**Replay images (+ matching wav if present):**
```powershell
python -m analyst --replay analyst/demo_assets --age 10
```

**Audio only:**
```powershell
python -m analyst --audio analyst/demo_assets/02_hate_threat.wav --age 10
```
