# Verification Report

**Date:** 2026-06-27  
**Machine:** WSL2, NVIDIA GeForce RTX 3060 Laptop GPU (6GB)  
**Driver:** 610.43.02 / CUDA UMD 13.3  
**PyTorch:** 2.12.1+cu130 (`.venv` core labs; `.venv-sys` for AWQ/vLLM)

## Summary

`python scripts/run_all_labs.py` → **39 pass / 0 fail / 0 skip**

| Tier | Count | Notes |
|------|-------|-------|
| Core automated labs | 32 | Python/CUDA/torchrun, no Docker |
| Optional service labs | 7 | kind/Volcano, LLVM Pass, elastic, AWQ, vLLM, Prometheus, promote |

## Verified optional lab outputs (RTX 3060 6GB)

| Lab | Key output |
|-----|------------|
| L1.6 kind_gang | `running_pods=6`, PodGroup `Running minMember=6` |
| L3.5 llvm_pass | `[semantic] before=8 after=8` (Docker `silkeh/clang:20`, LLVM 20.1.8) |
| L3.6 elastic_demo | `[ckpt] step=0/5`, `DONE at step=8` |
| L4.1 quant_bench | FP16 `1.00GB/42tok/s`; AWQ `0.48GB/36tok/s`; GPTQ `0.47GB/32tok/s`; PPL `10.14→10.90` |
| L4.2 vllm_serve | `6.6 req/s`, `P50=871ms P99=1867ms` (`gpu_memory_utilization=0.70`) |
| L5.5 obs_stack | Prometheus ready, Grafana health OK, `alerts count=1` |
| L5.6 promote_best | `accuracy=1.0000`, `iris-rf-classifier v1 -> Production` |

## Environment-specific adjustments

| Lab | Adjustment |
|-----|------------|
| `tc_vs_cuda_gemm.py` | `n=4096` (was 8192) to fit 6GB VRAM |
| `ddp_demo.py` / `tp_demo.py` | Use `gloo`+CPU when `device_count < world_size` |
| `fsdp_ckpt.py` | `torchrun --nproc_per_node=1` on single-GPU (FSDP requires accelerator) |
| `ds_config.json` | `train_batch_size=4` for `--num_gpus=1` |
| `hnsw_bench.py` | Local `QdrantClient(path=...)` instead of Docker; `N=2000` |
| `train_experiments.py` | Fallback `mlruns_lite/` when Python lacks `_lzma` |
| `elastic_train.py` | All-reduce tensors on `cuda` when backend is `nccl` |
| `ppl.py` | `load_dataset("Salesforce/wikitext", ...)` (legacy `wikitext` id broken on HF Hub) |
| LLVM Pass | Docker `silkeh/clang:20` when host lacks LLVM 20 dev packages |
| vLLM | `.venv-sys` (Python 3.12 + lzma); `gpu_memory_utilization=0.70`, `max_model_len=1024` on 6GB |
| AWQ bench | `.venv-sys` + `torchvision` + `gptqmodel` (transformers AWQ integration) |

## LLVM / Triton toolchain note

- **Teaching lab:** LLVM **20** (Docker `silkeh/clang:20` or `apt install llvm-20`)
- **Triton 3.7.1** (bundled with torch 2.12.1+cu130) pins LLVM commit `7f77ca0dbda4` → **LLVM 23.0.0git**
- Build script: `scripts/build_llvm_triton_pin.sh` (source build, ~1h+)

Full JSON output: `results/lab_results.json`
