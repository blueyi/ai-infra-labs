# hello_infra.py —— 本站第一个 Infra 健康检查脚本
import time
import torch

print("=" * 48)
print(f"[Hello-Infra] torch version : {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"[Hello-Infra] CUDA available: {cuda_ok}")

dev = "cuda" if cuda_ok else "cpu"
if cuda_ok:
    print(f"[Hello-Infra] GPU name     : {torch.cuda.get_device_name(0)}")
print(f"[Hello-Infra] using device  : {dev}")

# 一次小矩阵乘 + 计时（验证算子能落到设备上执行）
a = torch.randn(256, 256, device=dev)
b = torch.randn(256, 256, device=dev)

# 预热（首次会触发 kernel JIT / cuBLAS handle 初始化）
for _ in range(3):
    _ = a @ b
if cuda_ok:
    torch.cuda.synchronize()

t0 = time.perf_counter()
c = a @ b
if cuda_ok:
    torch.cuda.synchronize()  # 异步 kernel 必须同步后再停表
dt = (time.perf_counter() - t0) * 1e3

print(f"[Hello-Infra] 256x256 matmul: {dt:.3f} ms  -> result sum={c.sum().item():.2f}")
print("=" * 48)
print(">>> 环境基线就绪，欢迎进入 AI Infra 实战。")
