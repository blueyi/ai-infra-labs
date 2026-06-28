#!/usr/bin/env bash
# L4.4 — ONNX export + ORT infer smoke
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PY="${VENV:-../.venv}/bin/python"
[ -x "$PY" ] || PY=python3
"$PY" export_tiny_lm.py
"$PY" ort_infer.py
echo "[ok] onnx lab passed"
