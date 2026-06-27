# ai-infra-labs

Hands-on code companion for **[AI Infra 全栈从入门到精通](https://blueyi.github.io/ai-infra-docs/)** — the published learning site (L0–L6).

Each directory mirrors a doc chapter and contains runnable scripts extracted from the **动手实践** sections.

| Layer | Doc chapter (site) | Lab directory |
|-------|-------------------|---------------|
| L0 | [基础与环境](https://blueyi.github.io/ai-infra-docs/L0-foundations/hands-on-environment-baseline) | `L0-foundations/` |
| L1 | [算力与编排](https://blueyi.github.io/ai-infra-docs/L1-compute-orchestration/single-chip-microarchitecture) | `L1-compute-orchestration/` |
| L2 | [数据·算子·编译](https://blueyi.github.io/ai-infra-docs/L2-data-operators-compilers/cuda-programming-essentials-and-advanced) | `L2-data-operators-compilers/` |
| L3 | [训练](https://blueyi.github.io/ai-infra-docs/L3-training/distributed-training-and-3d-parallelism) | `L3-training/` |
| L4 | [推理与 Serving](https://blueyi.github.io/ai-infra-docs/L4-inference-serving/inference-acceleration-mechanisms) | `L4-inference-serving/` |
| L5 | [MLOps / LLMOps](https://blueyi.github.io/ai-infra-docs/L5-mlops-llmops/engineering-methodology) | `L5-mlops-llmops/` |
| L6 | [应用架构](https://blueyi.github.io/ai-infra-docs/L6-application-architecture/rag-retrieval-augmented-generation) | `L6-application-architecture/` |

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

**Heavy labs (AWQ / vLLM / GPTQ)** need a second venv with system Python 3.12 (has `lzma`):

```bash
uv venv --python /usr/bin/python3 .venv-sys && source .venv-sys/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install vllm transformers autoawq gptqmodel optimum torchvision bitsandbytes
```

## Layout

```
L0-foundations/          # NumPy MLP, pybind11, DataLoader, hello_infra
L1-compute-orchestration/  # GEMM TC benchmark, chip compare, checkpoint I/O
L2-data-operators-compilers/  # CUDA GEMM, Triton, torch.compile, ETL, HNSW
L3-training/             # DDP, TP, FSDP, DeepSpeed ZeRO, LLVM Pass, elastic
L4-inference-serving/    # AWQ/GPTQ quant, vLLM client, routing gateway
L5-mlops-llmops/         # MLflow, guardrails, CI gate, observability stack
L6-application-architecture/  # ReAct agent, RAG eval, multi-tenant gateway, UniIR
scripts/                 # run_all_labs.py
tools/                   # extract_from_mdx.py, normalize_labs.py
```

## Hardware notes (verified on RTX 3060 Laptop 6GB)

See [VERIFICATION.md](./VERIFICATION.md) for full **39/39 pass** results.

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

When the published docs are updated locally (clone [blueyi.github.io](https://github.com/blueyi/blueyi.github.io) or sibling `hexoblog/docs/ai-infra-docs`):

```bash
python tools/extract_from_mdx.py
python tools/normalize_labs.py
```

## License

MIT — code snippets originate from [ai-infra-docs](https://blueyi.github.io/ai-infra-docs/) educational content.
