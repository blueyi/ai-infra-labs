# Verification Report

**Date:** 2026-06-27  
**Machine:** WSL2, NVIDIA GeForce RTX 3060 Laptop GPU (6GB)  
**Driver:** 610.43.02 / CUDA UMD 13.3  
**PyTorch:** 2.12.1+cu130 (venv)

## Summary

`python scripts/run_all_labs.py` → **32 pass / 0 fail / 0 skip**

## Environment-specific adjustments

| Lab | Adjustment |
|-----|------------|
| `tc_vs_cuda_gemm.py` | `n=4096` (was 8192) to fit 6GB VRAM |
| `ddp_demo.py` / `tp_demo.py` | Use `gloo`+CPU when `device_count < world_size` |
| `fsdp_ckpt.py` | `torchrun --nproc_per_node=1` on single-GPU (FSDP requires accelerator) |
| `ds_config.json` | `train_batch_size=4` for `--num_gpus=1` |
| `hnsw_bench.py` | Local `QdrantClient(path=...)` instead of Docker; `N=2000` |
| `train_experiments.py` | Fallback `mlruns_lite/` when Python lacks `_lzma` |

## Optional labs (not in automated runner)

- L1 Volcano/kind gang scheduling — requires kind + Helm
- L3 LLVM Pass — requires LLVM 18 dev packages
- L4 AWQ/GPTQ/vLLM — requires large model downloads
- L5 Prometheus stack — requires `docker compose up`

Full JSON output: `results/lab_results.json`
