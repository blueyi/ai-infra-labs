#!/usr/bin/env python3
"""Rename mis-extracted lab files to canonical names."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# chapter_dir -> list of (src_glob_or_name, dst_name)
RENAMES: dict[str, list[tuple[str, str]]] = {
    "L0-foundations/ai-system-end-to-end-landscape": [("torch.cu", "gemm_profile.py")],
    "L0-foundations/data-pipeline-infrastructure-basics": [("torch.cu", "dataloader_throughput.py")],
    "L0-foundations/math-and-ml-foundations": [
        ("snippet_1.py", "mlp_numpy.py"),
        ("torch.cu", "mlp_torch_autograd.py"),
    ],
    "L0-foundations/physical-bottlenecks-three-walls": [("torch.cu", "bandwidth_roofline.py")],
    "L1-compute-orchestration/ai-storage-architecture-gds-checkpoint": [
        ("snippet_1.py", "ckpt_io_lab.py"),
    ],
    "L1-compute-orchestration/multi-region-hybrid-cloud-compute-network": [
        ("snippet_1.py", "multi_region_dr_calc.py"),
    ],
    "L1-compute-orchestration/single-chip-microarchitecture": [("torch.cu", "tc_vs_cuda_gemm.py")],
    "L2-data-operators-compilers/cuda-programming-essentials-and-advanced": [
        ("snippet_0.cu", "gemm.cu"),
        ("snippet_2.py", "tiled_numpy.py"),
    ],
    "L2-data-operators-compilers/dl-compiler-and-mlir-lowering": [
        ("torch.cu", "compile_bench.py"),
        ("snippet_2.py", "compile_bench_dynamo.py"),
    ],
    "L2-data-operators-compilers/feature-store-and-vector-db": [("snippet_2.py", "hnsw_bench.py")],
    "L2-data-operators-compilers/operator-shape-generalization-and-profiling": [
        ("torch.cu", "profile_infer.py"),
    ],
    "L2-data-operators-compilers/sota-kernel-flashattention-moe": [
        ("torch.cu", "flash_attn_triton.py"),
        ("torch_1.cu", "flash_attn_bench.py"),
        ("snippet_3.py", "flash_attn_sdpa_cpu.py"),
    ],
    "L2-data-operators-compilers/triton-compiler-source-level-practice": [
        ("torch.cu", "softmax.py"),
        ("snippet_2.py", "softmax_cpu.py"),
    ],
    "L3-training/memory-optimization-and-zero": [
        ("snippet_1.json", "ds_config.json"),
        ("torch.cu", "train_zero.py"),
    ],
    "L3-training/hardcore-compiler-llvm-mlir-practice": [
        ("CountAndDCEPass_1.cpp", "CMakeLists.txt"),
    ],
    "L4-inference-serving/inference-server-and-runtime": [("snippet_2.py", "bench_vllm_client.py")],
    "L4-inference-serving/performance-cost-and-hybrid-deployment": [("snippet_1.py", "gateway.py")],
    "L5-mlops-llmops/engineering-methodology": [("snippet_1.py", "bench_gemm.py")],
    "L5-mlops-llmops/llmops-specifics": [("snippet_1.py", "gateway_guardrails.py")],
    "L5-mlops-llmops/pipeline-and-release": [("model_1.json", "model_v1.json")],
    "L6-application-architecture/agent-and-multi-step-reasoning": [("snippet_1.py", "react_agent_lab.py")],
    "L6-application-architecture/rag-retrieval-augmented-generation": [("snippet_1.py", "min_rag_eval.py")],
}


def main() -> None:
    for rel, pairs in RENAMES.items():
        d = ROOT / rel
        for src, dst in pairs:
            s, t = d / src, d / dst
            if not s.exists():
                print(f"skip missing {s}")
                continue
            if t.exists() and t != s:
                t.unlink()
            s.rename(t)
            print(f"renamed {s.relative_to(ROOT)} -> {t.name}")

    # remove junk
    junk = ROOT / "L1-compute-orchestration/multi-region-hybrid-cloud-compute-network/snippet_0.mermaid"
    if junk.exists():
        junk.unlink()
        print(f"removed {junk.name}")

    # dedupe developer-prerequisites duplicates
    dev = ROOT / "L0-foundations/developer-prerequisites-baseline"
    for dup in ("add_1.cpp", "setup_1.py", "snippet_0.sh"):
        p = dev / dup
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
