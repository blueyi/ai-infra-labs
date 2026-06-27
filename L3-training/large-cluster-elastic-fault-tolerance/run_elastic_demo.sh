#!/usr/bin/env bash
# Short elastic training demo (8 steps, ~8s) — full lab uses ELASTIC_DEMO_STEPS=60
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/elastic_demo.log"
: > "$LOG"
source "$HOME/workspace/repos/ai-infra-labs/.venv/bin/activate"
rm -f /tmp/elastic_ckpt.pt
export ELASTIC_DEMO_STEPS=8
torchrun --nproc_per_node=1 elastic_train.py 2>&1 | tee "$LOG"
grep -q '\[ckpt\]' "$LOG"
grep -q 'DONE at step=8' "$LOG"
echo "[ok] elastic demo passed" | tee -a "$LOG"
