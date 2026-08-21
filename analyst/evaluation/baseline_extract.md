# Step 2 — OCR extraction baseline

Measured with RapidOCR on synthetic `analyst/demo_assets/` screenshots (high-contrast).

| Asset | Expect | Result |
|---|---|---|
| `01_clean_gaming.png` | gaming slang tokens | See unittest `test_ocr_demo` |
| `02_hate_threat.png` | threat tokens → cascade **hate** | See unittest |
| `03_hate_identity.png` | identity tokens | See unittest |
| `04_benign_chat.png` | benign tokens → **not-hate** | See unittest |

**Backend:** `rapidocr` (fallback: tesseract / none).  
**Rule:** empty OCR must not crash; overlay `--text` still works.

Re-run:

```powershell
python -m analyst.demo_assets.generate
python -m unittest analyst.tests.test_ocr_demo -v
python -m analyst --replay analyst/demo_assets --age 10
```
