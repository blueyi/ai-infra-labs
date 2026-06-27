import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {dev}")

# 一个小 Transformer Encoder 作为推理负载（也可换成 torchvision.models.resnet18）
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                               batch_first=True),
    num_layers=4,
).to(dev).eval()

x = torch.randn(8, 256, 512, device=dev)  # [batch, seq_len, d_model]

# 预热：触发 cuBLAS handle / kernel JIT，避免首次开销污染测量
with torch.inference_mode():
    for _ in range(5):
        _ = model(x)
    if dev == "cuda":
        torch.cuda.synchronize()

# 有 GPU 抓 CUDA 活动，无 GPU 抓 CPU 活动（aten 算子）
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])

with torch.inference_mode():
    with profile(
        activities=acts,
        record_shapes=True,     # 记录每个算子的输入 shape，便于归因
        with_stack=False,
        profile_memory=True,    # 记录显存分配，辅助判 Memory-Bound
    ) as prof:
        for _ in range(10):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()

# ① 按设备耗时排序，打印 top-k 算子表
sort_key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by=sort_key, row_limit=10))

# ② 导出 chrome trace —— 拖进 chrome://tracing 或 https://ui.perfetto.dev 看 timeline
prof.export_chrome_trace("trace.json")
print("[ok] 已导出 trace.json，可用 Perfetto / chrome://tracing 打开看 kernel timeline")
