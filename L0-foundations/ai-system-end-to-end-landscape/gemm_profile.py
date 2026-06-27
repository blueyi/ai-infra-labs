import torch
from torch.profiler import profile, ProfilerActivity

# 选择设备：有 GPU 走 cuda，否则 cpu
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {dev}")

a = torch.randn(2048, 2048, device=dev)
b = torch.randn(2048, 2048, device=dev)

# 预热（首次会触发 kernel JIT / cuBLAS handle 初始化）
for _ in range(3):
    _ = a @ b
if dev == "cuda":
    torch.cuda.synchronize()

acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
with profile(activities=acts, record_shapes=True) as prof:
    c = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()

# 打印按设备耗时排序的算子，看 matmul 落到了哪个底层 kernel
key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=key, row_limit=8))
