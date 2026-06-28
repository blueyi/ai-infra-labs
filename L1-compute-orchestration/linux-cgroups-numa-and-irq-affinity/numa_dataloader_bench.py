#!/usr/bin/env python3
"""L1.8 — compare DataLoader throughput with/without numactl (CPU-only OK)."""
import os
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

N, D, STEPS = 50_000, 4096, 200
BATCH, WORKERS = 256, 4


def bench() -> float:
    data = TensorDataset(torch.randn(N, D))
    loader = DataLoader(data, batch_size=BATCH, num_workers=WORKERS, pin_memory=False)
    t0 = time.perf_counter()
    for i, (x,) in enumerate(loader):
        if i >= STEPS:
            break
        _ = x.sum()
    return time.perf_counter() - t0


def main() -> int:
    baseline = bench()
    print(f"pid={os.getpid()} baseline_elapsed={baseline:.3f}s")

    if shutil_which("numactl"):
        for node in (0, 1):
            cmd = [
                "numactl", f"--cpunodebind={node}", f"--membind={node}",
                sys.executable, __file__, "--inner",
            ]
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=120)
                print(out.strip())
            except subprocess.CalledProcessError as exc:
                print(f"node{node}: {exc.output.strip()}")
    else:
        print("[info] numactl not found — baseline only (install numactl on dual-socket hosts)")

    print("[ok] numa dataloader bench finished")
    return 0


def shutil_which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


if __name__ == "__main__":
    if "--inner" in sys.argv:
        print(f"pid={os.getpid()} numa_elapsed={bench():.3f}s")
        raise SystemExit(0)
    raise SystemExit(main())
