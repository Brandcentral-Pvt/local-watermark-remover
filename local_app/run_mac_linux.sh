#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Watermark Remover - launcher for macOS / Linux
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Python 3.10+ is required. Install it first (macOS: brew install python; Debian/Ubuntu: sudo apt install python3 python3-venv)."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[1/3] creating virtual environment..."
  "$PY" -m venv .venv
fi

echo "[2/3] installing dependencies (first run only)..."
./.venv/bin/python -m pip install --upgrade pip -q
./.venv/bin/python -m pip install -r requirements.txt

if [ ! -f "models/lama_fp32.onnx" ]; then
  echo "[3/3] downloading the model (208 MB, once)..."
  ./.venv/bin/python watermark_gui.py --download
fi

echo "starting the app - your browser will open at http://127.0.0.1:7860"
./.venv/bin/python watermark_gui.py --port 7860
