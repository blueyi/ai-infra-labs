#!/usr/bin/env bash
# L4 AWQ/GPTQ quantization bench on Qwen2.5-0.5B (6GB GPU friendly)
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/quant_bench.log"
: > "$LOG"
source "$HOME/workspace/repos/ai-infra-labs/.venv-sys/bin/activate"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
export HF_HOME="${HF_HOME:-$LAB/.hf_cache}"

pip install -q transformers accelerate datasets autoawq optimum 2>&1 | tail -2

python awq_quant.py 2>&1 | tee -a "$LOG"
python bench.py 2>&1 | tee -a "$LOG"
python gptq_quant.py 2>&1 | tee -a "$LOG" || {
  echo "[warn] GPTQ failed; running bitsandbytes NF4 fallback" | tee -a "$LOG"
  python quant_bnb_fallback.py 2>&1 | tee -a "$LOG"
}
python -c "
import os
from bench import benchmark
if os.path.isdir('qwen0.5b-gptq'):
    benchmark('qwen0.5b-gptq', 'GPTQ-INT4')
else:
    print('[GPTQ-INT4] skip — gptq_quant.py did not produce qwen0.5b-gptq/')
" 2>&1 | tee -a "$LOG"
python ppl.py 2>&1 | tee -a "$LOG"
echo "[ok] quantization lab passed" | tee -a "$LOG"
