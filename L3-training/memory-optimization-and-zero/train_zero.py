import time, json, torch, torch.nn as nn, deepspeed

# 一个故意「优化器状态偏重」的小模型，便于观测 ZeRO 切分效果
class Net(nn.Module):
    def __init__(self, d=4096, layers=8):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)])
    def forward(self, x):
        return self.net(x)

def main():
    model = Net()
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json",
    )
    dev = engine.device
    torch.cuda.reset_peak_memory_stats(dev)

    steps, bs, d = 20, 4, 4096
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for _ in range(steps):
        x = torch.randn(bs, d, device=dev, dtype=torch.float16)
        y = torch.randn(bs, d, device=dev, dtype=torch.float16)
        loss = loss_fn(engine(x), y)
        engine.backward(loss)     # ZeRO 在此做 Reduce-Scatter
        engine.step()             # 在此做参数更新 + All-Gather
    torch.cuda.synchronize(dev)
    dt = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
    thpt = steps * bs / dt
    stage = json.load(open("ds_config.json"))["zero_optimization"]["stage"]
    print(f"[ZeRO-{stage}] peak_mem={peak_gb:.2f} GB  throughput={thpt:.1f} samples/s")

if __name__ == "__main__":
    main()
