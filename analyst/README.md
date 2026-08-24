# J26-DS-316 Component 2 — The Analyst
#
# Step-by-step guide: analyst/BUILD_STEPS.md
#
# Requires Python 3.10-3.14. On 3.13+ requirements.txt installs `rapidocr`
# instead of `rapidocr-onnxruntime` (which caps at <3.13); ocr.py drives both.
# Windows: clone to a SHORT path (e.g. C:\dev\...). Deep paths break the
# pyclipper DLL load and OCR silently falls back to "none".
#
# Quick start (from repo root):
#   pip install -r analyst/requirements.txt
#   python -m analyst.demo_assets.generate
#   python -m analyst --text "you should kys" --age 10
#   python -m analyst --replay analyst/demo_assets --age 10
#   python -m analyst.demo_e2e --age 10
#
# Live panel (Step 5b — temporary C2 capture until C1):
#   pip install fastapi uvicorn python-multipart mss soundcard
#   python -m analyst.serve
#   open http://127.0.0.1:8765
#
# Step 1 pretrained text + Step 4 CLIP (optional):
#   pip install transformers
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Tests:
#   python -m unittest analyst.tests.test_pipeline analyst.tests.test_demo_e2e analyst.tests.test_store_capture -v
