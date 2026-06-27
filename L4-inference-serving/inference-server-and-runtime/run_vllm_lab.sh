#!/usr/bin/env bash
# L4 vLLM serve + async bench client (Qwen2.5-0.5B, 6GB GPU)
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/vllm_bench.log"
: > "$LOG"
VENV="${VENV:-$HOME/workspace/repos/ai-infra-labs/.venv-sys}"
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-$LAB/.hf_cache}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"

uv pip install -q vllm openai 2>&1 | tail -3 | tee -a "$LOG"

pkill -f "vllm serve" 2>/dev/null || true
sleep 2
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

vllm serve "$MODEL_ID" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 1024 \
  --port 8000 \
  > vllm_server.log 2>&1 &
VLLM_PID=$!

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[vllm] server ready after ${i}0s" | tee -a "$LOG"
    break
  fi
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -30 vllm_server.log; exit 1; }

python bench_vllm_client.py 2>&1 | tee -a "$LOG"
kill $VLLM_PID 2>/dev/null || true
echo "[ok] vllm bench passed" | tee -a "$LOG"
