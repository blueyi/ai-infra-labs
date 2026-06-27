# ddp_demo.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    ngpu = torch.cuda.device_count()

    # Single-GPU machines: fall back to gloo+CPU when ranks > GPUs
    use_nccl = torch.cuda.is_available() and ngpu >= world
    backend = "nccl" if use_nccl else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world)

    dev = f"cuda:{rank}" if use_nccl else "cpu"
    if use_nccl:
        torch.cuda.set_device(rank)

    model = nn.Linear(8, 8).to(dev)
    ddp = DDP(model, device_ids=[rank] if backend == "nccl" else None)

    # 每个 rank 喂不同的数据分片
    x = torch.randn(4, 8, device=dev)
    loss = ddp(x).sum()
    loss.backward()  # ← 反向结束时 DDP 自动触发梯度 All-Reduce

    # 验证：All-Reduce 后所有 rank 的梯度完全一致
    g = next(ddp.parameters()).grad
    print(f"[rank {rank}] backend={backend} grad_mean={g.mean().item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
