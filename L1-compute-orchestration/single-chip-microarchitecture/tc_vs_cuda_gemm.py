import torch

def benchmark_gemm(dtype, n=4096, iters=50, warmup=10):
    """测量 n×n @ n×n GEMM 的 TFLOPS。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)

    # 一次 GEMM 的浮点运算量：2 * n^3（每个输出元素 n 次乘 + n 次加）
    flops = 2.0 * n ** 3

    if dev == "cuda":
        # 预热：触发 cuBLAS handle 初始化与 kernel 选择
        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            c = a @ b
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters          # 平均每次耗时(ms)
    else:
        import time
        for _ in range(3):                             # CPU 维度需调小，见 Gotchas
            _ = a @ b
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        ms = (time.perf_counter() - t0) / iters * 1e3

    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    # 让 cuBLAS 允许使用 TF32（Ampere+ 的 Tensor Core 加速 FP32 GEMM）
    torch.backends.cuda.matmul.allow_tf32 = True

    if dev == "cuda":
        configs = [("FP32 (CUDA Core)", torch.float32),
                   ("TF32 (Tensor Core)", torch.float32),  # 由 allow_tf32 控制
                   ("FP16 (Tensor Core)", torch.float16),
                   ("BF16 (Tensor Core)", torch.bfloat16)]
        # FP32 纯 CUDA Core 基线：关闭 TF32 单独测一次
        torch.backends.cuda.matmul.allow_tf32 = False
        ms, tf = benchmark_gemm(torch.float32)
        print(f"{'FP32 (CUDA Core)':<22} {ms:8.2f} ms   {tf:8.1f} TFLOPS")
        torch.backends.cuda.matmul.allow_tf32 = True
        for name, dt in configs[1:]:
            ms, tf = benchmark_gemm(dt)
            print(f"{name:<22} {ms:8.2f} ms   {tf:8.1f} TFLOPS")
    else:
        # CPU 路径：维度调小，仅对比 float32 / bfloat16 说明原理
        for name, dt in [("FP32 (CPU)", torch.float32),
                         ("BF16 (CPU)", torch.bfloat16)]:
            ms, tf = benchmark_gemm(dt, n=2048)
            print(f"{name:<22} {ms:8.2f} ms   {tf:8.1f} TFLOPS")
