import time
import torch
from torch.utils.data import Dataset, DataLoader

# 合成数据集：模拟「读一个样本需要一点 CPU 工作」的场景
class SyntheticDataset(Dataset):
    def __init__(self, n=20000, dim=3 * 224 * 224):
        self.n, self.dim = n, dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 故意做一点 CPU 计算，模拟解码 / tokenize 的开销
        x = torch.randn(self.dim)
        x = torch.sin(x) * torch.cos(x)        # 假装在做预处理
        return x.reshape(3, 224, 224), idx % 1000

def benchmark(num_workers, pin_memory, dev, batch_size=128, max_batches=120):
    ds = SyntheticDataset()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(2 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
    )

    seen, t0 = 0, None
    for i, (x, y) in enumerate(loader):
        if i == 5:                  # 跳过前几个 batch 预热，避开 worker 启动毛刺
            t0 = time.perf_counter()
        if i < 5:
            continue
        x = x.to(dev, non_blocking=pin_memory)   # 模拟搬上 GPU
        if dev == "cuda":
            torch.cuda.synchronize()
        seen += x.size(0)
        if i >= max_batches:
            break
    dt = time.perf_counter() - t0
    return seen / dt

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}\n")
    print(f"{'num_workers':>12} | {'pin_memory':>10} | {'samples/sec':>12}")
    print("-" * 42)
    for nw in [0, 2, 4, 8]:
        for pin in ([False, True] if dev == "cuda" else [False]):
            tput = benchmark(nw, pin, dev)
            print(f"{nw:>12} | {str(pin):>10} | {tput:>12.1f}")
