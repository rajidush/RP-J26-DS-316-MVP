# J26-DS-316 Component 2 — The Analyst
#
# Step-by-step guide: analyst/BUILD_STEPS.md
#
# Quick start (from repo root):
#   pip install -r analyst/requirements.txt
#   python -m analyst.demo_assets.generate
#   python -m analyst --text "you should kys" --age 10
#   python -m analyst --replay analyst/demo_assets --age 10
#   python -m analyst.demo_e2e --age 10
#
# Step 1 pretrained text + Step 4 CLIP (optional):
#   pip install transformers
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Tests:
#   python -m unittest analyst.tests.test_pipeline analyst.tests.test_demo_e2e -v
