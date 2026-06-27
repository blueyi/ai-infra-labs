import time

# ---------- 通用：计算 Roofline 拐点 ----------
def ridge_point(peak_flops, peak_bw_bytes):
    """拐点算术强度 = 峰值算力 / 峰值带宽 (FLOP/Byte)"""
    return peak_flops / peak_bw_bytes

# ---------- 路径选择 ----------
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

N = 256 * 1024 * 1024  # 256M 个元素，约 1GB (float32)
BYTES = N * 4

if HAS_GPU:
    # ===== NVIDIA GPU 路径：测 HBM 拷贝带宽 =====
    import torch
    dev = "cuda"
    src = torch.randn(N, device=dev)
    dst = torch.empty_like(src)
    for _ in range(3):                 # 预热，触发 CUDA 上下文初始化
        dst.copy_(src)
    torch.cuda.synchronize()           # 关键：异步 kernel 必须同步后再计时

    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters):
        dst.copy_(src)                 # device-to-device 拷贝：1 读 + 1 写
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    bw = (BYTES * 2 * iters) / dt / 1e9   # 读+写算 2 倍流量
    print(f"[GPU] 实测 HBM 拷贝带宽 ≈ {bw:.1f} GB/s")
    # H100 理论峰值：算力 990 TFLOPS(FP16), 带宽 3.35 TB/s
    print(f"[GPU] H100 理论拐点 AI* = {ridge_point(990e12, 3.35e12):.0f} FLOP/Byte")
else:
    # ===== Mac / CPU 替代方案：测主存带宽 =====
    import numpy as np
    dev = "cpu"
    src = np.random.randn(N).astype(np.float32)
    dst = np.empty_like(src)
    for _ in range(2):                 # 预热，填满 cache 行
        np.copyto(dst, src)

    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        np.copyto(dst, src)            # 主存拷贝：1 读 + 1 写
    dt = time.perf_counter() - t0

    bw = (BYTES * 2 * iters) / dt / 1e9
    print(f"[CPU] 实测主存拷贝带宽 ≈ {bw:.1f} GB/s")
    # 典型笔记本 DDR4/5：算力 ~1 TFLOPS, 带宽 ~50 GB/s
    print(f"[CPU] 示例拐点 AI* = {ridge_point(1e12, 50e9):.0f} FLOP/Byte")

print(f"[device] {dev} | 数据规模 {BYTES/1e9:.2f} GB")
