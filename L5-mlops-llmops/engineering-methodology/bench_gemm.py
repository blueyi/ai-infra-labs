import json
import platform
import statistics
import time
import os

import numpy as np

# ---------- 1) 可复现性：固定种子 + 记录环境 ----------
SEED = 42
np.random.seed(SEED)

def collect_env() -> dict:
    """采集环境清单——可复现报告的「身份证」。"""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "seed": SEED,
    }

# ---------- 2) 被测操作：一次 GEMM ----------
N = 1024
A = np.random.rand(N, N).astype(np.float64)
B = np.random.rand(N, N).astype(np.float64)

def workload():
    return A @ B

# ---------- 3) 测量协议：warmup + 多次采样 ----------
WARMUP = 5      # 丢弃首批样本：排除缓存冷启 / 库初始化
REPEAT = 30     # 采样次数：足够做稳健统计

def benchmark(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):          # 预热，不计入
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()     # 单调高精度时钟
        fn()
        samples.append(time.perf_counter() - t0)
    return samples

# ---------- 4) 统计聚合：均值/方差/中位数/p99/置信区间 ----------
def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    mean = statistics.mean(s)
    stdev = statistics.stdev(s) if n > 1 else 0.0
    # 95% 置信区间（正态近似，z=1.96）；标准误 = std / sqrt(n)
    half = 1.96 * stdev / (n ** 0.5)
    def pct(p):
        idx = min(n - 1, int(round(p / 100 * (n - 1))))
        return s[idx]
    return {
        "n": n,
        "mean_s": mean,
        "stdev_s": stdev,
        "cv_pct": (stdev / mean * 100) if mean else 0.0,  # 变异系数：方差是否可接受
        "median_s": statistics.median(s),
        "p99_s": pct(99),
        "ci95_low_s": mean - half,
        "ci95_high_s": mean + half,
    }

# ---------- 5) 出报告 ----------
def main():
    samples = benchmark(workload, WARMUP, REPEAT)
    report = {
        "env": collect_env(),
        "workload": f"GEMM {N}x{N} float64 (A @ B)",
        "protocol": {"warmup": WARMUP, "repeat": REPEAT},
        "stats": summarize(samples),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    st = report["stats"]
    print("\n=== 人类可读摘要 ===")
    print(f"中位数  : {st['median_s']*1e3:8.3f} ms")
    print(f"均值    : {st['mean_s']*1e3:8.3f} ms  ± {st['stdev_s']*1e3:.3f} ms (std)")
    print(f"95% CI  : [{st['ci95_low_s']*1e3:.3f}, {st['ci95_high_s']*1e3:.3f}] ms")
    print(f"p99     : {st['p99_s']*1e3:8.3f} ms")
    print(f"变异系数: {st['cv_pct']:8.2f} %  (>5% 说明方差偏大，需锁频/排查隐藏状态)")

if __name__ == "__main__":
    main()
