#!/usr/bin/env python3
"""Run ai-infra-docs hands-on labs and report pass/fail/skip."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lab_results.json"


@dataclass
class Lab:
    id: str
    cmd: list[str]
    cwd: Path
    timeout: int = 120
    needs_gpu: bool = False
    needs_multi_gpu: bool = False
    needs_docker: bool = False
    needs_services: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    build_cmd: list[str] | None = None


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def run_one(lab: Lab) -> dict:
    out: dict = {"id": lab.id, "status": "unknown", "duration_s": 0.0}
    if lab.needs_gpu and not has_gpu():
        out["status"] = "skip"
        out["reason"] = "no GPU"
        return out
    if lab.needs_multi_gpu:
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                out["status"] = "skip"
                out["reason"] = "needs 2+ GPUs"
                return out
        except ImportError:
            out["status"] = "skip"
            out["reason"] = "torch missing"
            return out
    if lab.needs_docker and not has_cmd("docker"):
        out["status"] = "skip"
        out["reason"] = "docker not installed"
        return out

    env = {**os.environ, **lab.env}
    t0 = time.time()
    try:
        if lab.build_cmd:
            subprocess.run(
                lab.build_cmd, cwd=lab.cwd, env=env, check=True,
                capture_output=True, text=True, timeout=lab.timeout,
            )
        proc = subprocess.run(
            lab.cmd, cwd=lab.cwd, env=env, check=False,
            capture_output=True, text=True, timeout=lab.timeout,
        )
        out["duration_s"] = round(time.time() - t0, 2)
        out["stdout"] = proc.stdout[-4000:]
        out["stderr"] = proc.stderr[-2000:]
        out["returncode"] = proc.returncode
        out["status"] = "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        out["status"] = "fail"
        out["reason"] = f"timeout>{lab.timeout}s"
        out["duration_s"] = round(time.time() - t0, 2)
    except Exception as e:
        out["status"] = "fail"
        out["reason"] = str(e)
        out["duration_s"] = round(time.time() - t0, 2)
    return out


def labs() -> list[Lab]:
    py = sys.executable
    L = Lab
    return [
        L("L0.6 hello_infra", [py, "hello_infra.py"], ROOT / "L0-foundations/hands-on-environment-baseline"),
        L("L0.1 gemm_profile", [py, "gemm_profile.py"], ROOT / "L0-foundations/ai-system-end-to-end-landscape"),
        L("L0.2 bandwidth_roofline", [py, "bandwidth_roofline.py"], ROOT / "L0-foundations/physical-bottlenecks-three-walls", timeout=180),
        L("L0.3 mlp_numpy", [py, "mlp_numpy.py"], ROOT / "L0-foundations/math-and-ml-foundations", timeout=180),
        L("L0.4 pybind11", [py, "setup.py", "build_ext", "--inplace"], ROOT / "L0-foundations/developer-prerequisites-baseline", build_cmd=None),
        L("L0.5 dataloader", [py, "dataloader_throughput.py"], ROOT / "L0-foundations/data-pipeline-infrastructure-basics", timeout=300, needs_gpu=True),
        L("L1.1 tc_vs_cuda_gemm", [py, "tc_vs_cuda_gemm.py"], ROOT / "L1-compute-orchestration/single-chip-microarchitecture", needs_gpu=True, timeout=180),
        L("L1.2 chip_compare", [py, "chip_compare.py"], ROOT / "L1-compute-orchestration/ai-chip-vendor-comparison"),
        L("L1.3 ckpt_io", [py, "ckpt_io_lab.py"], ROOT / "L1-compute-orchestration/ai-storage-architecture-gds-checkpoint", timeout=180),
        L("L1.4 multi_region_dr", [py, "multi_region_dr_calc.py"], ROOT / "L1-compute-orchestration/multi-region-hybrid-cloud-compute-network"),
        L("L1.5 allreduce_gloo", [py, "allreduce_gloo.py"], ROOT / "L1-compute-orchestration/interconnect-topology-nvlink-ib-roce"),
        L("L2.1 tiled_numpy", [py, "tiled_numpy.py"], ROOT / "L2-data-operators-compilers/cuda-programming-essentials-and-advanced"),
        L("L2.1 gemm.cu", ["nvcc", "-O3", "-arch=sm_86", "gemm.cu", "-o", "gemm"], ROOT / "L2-data-operators-compilers/cuda-programming-essentials-and-advanced", needs_gpu=True, build_cmd=None),
        L("L2.2 compile_bench", [py, "compile_bench.py"], ROOT / "L2-data-operators-compilers/dl-compiler-and-mlir-lowering", timeout=300, needs_gpu=True),
        L("L2.3 softmax_triton", [py, "softmax.py"], ROOT / "L2-data-operators-compilers/triton-compiler-source-level-practice", needs_gpu=True),
        L("L2.4 profile_infer", [py, "profile_infer.py"], ROOT / "L2-data-operators-compilers/operator-shape-generalization-and-profiling", needs_gpu=True, timeout=180),
        L("L2.5 flash_attn", [py, "flash_attn_triton.py"], ROOT / "L2-data-operators-compilers/sota-kernel-flashattention-moe", needs_gpu=True, timeout=300),
        L("L2.6 etl_gate", [py, "etl_with_gate.py"], ROOT / "L2-data-operators-compilers/data-pipeline-and-storage"),
        L("L2.7 hnsw_bench", [py, "hnsw_bench.py"], ROOT / "L2-data-operators-compilers/feature-store-and-vector-db", needs_docker=True, timeout=300),
        L("L3.1 ddp", ["torchrun", "--nproc_per_node=2", "ddp_demo.py"], ROOT / "L3-training/distributed-training-and-3d-parallelism"),
        L("L3.1 tp", ["torchrun", "--nproc_per_node=2", "tp_demo.py"], ROOT / "L3-training/distributed-training-and-3d-parallelism"),
        L("L3.2 fsdp", ["torchrun", "--nproc_per_node=1", "fsdp_ckpt.py"], ROOT / "L3-training/mainstream-frameworks-practice", timeout=180, needs_gpu=True),
        L("L3.3 deepspeed", ["deepspeed", "--num_gpus=1", "train_zero.py"], ROOT / "L3-training/memory-optimization-and-zero", needs_gpu=True, timeout=300),
        L("L4.3 gateway", [py, "gateway.py"], ROOT / "L4-inference-serving/performance-cost-and-hybrid-deployment", timeout=30),
        L("L5.1 bench_gemm", [py, "bench_gemm.py"], ROOT / "L5-mlops-llmops/engineering-methodology"),
        L("L5.2 mlflow_train", [py, "train_experiments.py"], ROOT / "L5-mlops-llmops/experiment-and-model-management", timeout=120),
        L("L5.3 gateway_guardrails", [py, "gateway_guardrails.py"], ROOT / "L5-mlops-llmops/llmops-specifics"),
        L("L5.4 validate_model", [py, "validate_model.py"], ROOT / "L5-mlops-llmops/pipeline-and-release"),
        L("L6.1 react_agent", [py, "react_agent_lab.py"], ROOT / "L6-application-architecture/agent-and-multi-step-reasoning"),
        L("L6.2 mock_gateway", [py, "mock_gateway.py"], ROOT / "L6-application-architecture/platform-governance-and-patterns"),
        L("L6.3 min_rag", [py, "min_rag_eval.py"], ROOT / "L6-application-architecture/rag-retrieval-augmented-generation", timeout=300),
        L("L6.4 uniir_demo", [py, "uniir_demo.py"], ROOT / "L6-application-architecture/ultimate-engineering-vision"),
    ]


def main() -> int:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    # pybind11 import test after build
    pybind_lab = next(x for x in labs() if x.id == "L0.4 pybind11")
    r = run_one(pybind_lab)
    if r["status"] == "pass":
        test = subprocess.run(
            [sys.executable, "-c", "import myadd; assert myadd.vector_add([1,2],[3,4])==[4,6]"],
            cwd=pybind_lab.cwd, capture_output=True, text=True,
        )
        r["status"] = "pass" if test.returncode == 0 else "fail"
        if test.returncode != 0:
            r["stderr"] = test.stderr

    # gemm.cu build+run
    gemm_dir = ROOT / "L2-data-operators-compilers/cuda-programming-essentials-and-advanced"
    gemm_bin = gemm_dir / "gemm"

    results = []
    for lab in labs():
        if lab.id == "L0.4 pybind11":
            results.append(r)
            continue
        if lab.id == "L2.1 gemm.cu":
            if not has_gpu() or not has_cmd("nvcc"):
                results.append({"id": lab.id, "status": "skip", "reason": "no nvcc or GPU"})
                continue
            br = subprocess.run(lab.cmd, cwd=lab.cwd, capture_output=True, text=True)
            if br.returncode != 0:
                results.append({"id": lab.id, "status": "fail", "stderr": br.stderr, "stdout": br.stdout})
                continue
            rr = subprocess.run([str(gemm_bin)], cwd=lab.cwd, capture_output=True, text=True, timeout=120)
            results.append({
                "id": lab.id,
                "status": "pass" if rr.returncode == 0 else "fail",
                "stdout": rr.stdout,
                "stderr": rr.stderr,
            })
            continue
        if lab.id == "L4.3 gateway":
            # gateway is uvicorn app — run import smoke test
            rr = subprocess.run(
                [sys.executable, "-c", "import gateway; print('gateway ok')"],
                cwd=lab.cwd, capture_output=True, text=True,
            )
            results.append({"id": lab.id, "status": "pass" if rr.returncode == 0 else "fail", "stdout": rr.stdout, "stderr": rr.stderr})
            continue
        results.append(run_one(lab))

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(1 for x in results if x["status"] == "pass")
    failed = sum(1 for x in results if x["status"] == "fail")
    skipped = sum(1 for x in results if x["status"] == "skip")
    print(f"\n=== Lab Results: {passed} pass / {failed} fail / {skipped} skip ===")
    for x in results:
        mark = {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(x["status"], "?")
        extra = f" ({x.get('reason', '')})" if x.get("reason") else ""
        print(f"{mark} {x['id']}{extra}")
        if x["status"] == "fail":
            if x.get("stderr"):
                print("  stderr:", x["stderr"][-500:])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
