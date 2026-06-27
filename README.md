# ai-infra-labs

Hands-on code companion for [AI Infra Docs](https://github.com/blueyi/hexoblog/tree/master/docs/ai-infra-docs) (`docs/ai-infra-docs` in hexoblog).

Each directory mirrors a doc chapter and contains runnable scripts extracted from the **动手实践** sections.

## Quick start

```bash
git clone https://github.com/blueyi/ai-infra-labs.git
cd ai-infra-labs
python3 -m venv .venv && source .venv/bin/activate
pip install torch  # add --index-url for CUDA wheel if needed
pip install -r requirements.txt

# Smoke test environment
python L0-foundations/hands-on-environment-baseline/hello_infra.py

# Run all verifiable labs (32 core + 7 optional service labs)
python scripts/run_all_labs.py
cat results/lab_results.json
cat VERIFICATION.md
```

**Heavy labs (AWQ / vLLM)** need a second venv with system Python 3.12 (has `lzma`):

```bash
uv venv --python /usr/bin/python3 .venv-sys && source .venv-sys/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install vllm transformers autoawq gptqmodel torchvision bitsandbytes
```

## Layout

```
L0-foundations/          # NumPy MLP, pybind11, DataLoader, hello_infra
L1-compute-orchestration/  # GEMM TC benchmark, chip compare, checkpoint I/O
L2-data-operators-compilers/  # CUDA GEMM, Triton, torch.compile, ETL, HNSW
L3-training/             # DDP, TP, FSDP, DeepSpeed ZeRO
L4-inference-serving/    # Quantization, vLLM client, routing gateway
L5-mlops-llmops/         # MLflow, guardrails, CI gate, observability stack
L6-application-architecture/  # ReAct agent, RAG eval, multi-tenant gateway, UniIR
scripts/                 # run_all_labs.py
tools/                   # extract_from_mdx.py, normalize_labs.py
```

## Hardware notes (verified on RTX 3060 Laptop 6GB)

| Lab | Requirement |
|-----|-------------|
| `tc_vs_cuda_gemm.py` | GPU; matrix size tuned to 4096 for 6GB VRAM |
| `flash_attn_triton.py` | GPU + Triton |
| `train_zero.py` | GPU; run with `deepspeed --num_gpus=1` on single-GPU machines |
| `ddp_demo.py` / `tp_demo.py` | 2 processes via `torchrun` (CPU gloo OK) |
| `hnsw_bench.py` | Docker Qdrant on `:6333` |
| AWQ/GPTQ/vLLM | `.venv-sys` (Python 3.12); see `run_quant_lab.sh` / `run_vllm_lab.sh` |
| LLVM Pass | Docker `silkeh/clang:20` via `run_llvm_pass.sh` |
| kind + Volcano | `run_kind_gang.sh` (~10 min) |

## Regenerate from docs

When hexoblog docs change:

```bash
python tools/extract_from_mdx.py
python tools/normalize_labs.py
```

## License

MIT — code snippets originate from ai-infra-docs educational content.
