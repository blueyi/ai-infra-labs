import torch
import torch.nn as nn
import time

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {dev}")


# 一个故意制造融合机会的小模型：多个逐元素算子串联
class TinyMLP(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x) * torch.sigmoid(x)  # 逐元素链，编译期可融合
        x = self.fc2(x)
        return x.sum()


model = TinyMLP().to(dev).eval()
x = torch.randn(512, 2048, device=dev)


def bench(fn, name, iters=50):
    # 预热：触发 JIT / 编译 / cuBLAS 初始化
    for _ in range(10):
        fn(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    print(f"[{name}] {dt:.3f} ms/iter")
    return dt


with torch.no_grad():
    eager = bench(model, "eager")
    compiled_fn = torch.compile(model)  # 三驾马车上线
    comp = bench(compiled_fn, "compiled")

print(f"[speedup] {eager / comp:.2f}x")
