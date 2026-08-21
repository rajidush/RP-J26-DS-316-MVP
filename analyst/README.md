# J26-DS-316 Component 2 — The Analyst
#
# Production-oriented hate-speech cascade (Engineering Plan v0.1 §4).
# Does NOT depend on the Next.js web demo.
#
# Quick start (from repo root):
#   cd analyst
#   python -m venv .venv
#   .venv\Scripts\Activate.ps1
#   pip install -r requirements.txt
#   pip install rapidocr-onnxruntime faster-whisper   # optional
#   cd ..
#   python -m analyst --text "you should kys" --age 10
#   python -m analyst --text "gg ez noob" --age 12
#   python -m analyst --image path\to\frame.png --age 10
#
# Replay a folder of screenshots:
#   python -m analyst --replay path\to\frames --age 10
#
# Tests:
#   python -m unittest analyst.tests.test_pipeline -v
