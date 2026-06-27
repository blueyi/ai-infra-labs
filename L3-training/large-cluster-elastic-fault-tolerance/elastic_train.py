# elastic_train.py
import os, time, torch, torch.distributed as dist

CKPT = "/tmp/elastic_ckpt.pt"

def setup():
    # GPU 走 nccl，CPU/Mac 走 gloo —— 弹性逻辑完全一致
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"[rank {rank}] joined, world_size={world}, backend={backend}", flush=True)
    return rank, world, use_cuda

def load_step():
    # 无重启热恢复的核心：从最新 checkpoint 续训，而非从 0 开始
    if os.path.exists(CKPT):
        step = torch.load(CKPT)["step"]
        print(f"[resume] 从 checkpoint 恢复，续训 step={step}", flush=True)
        return step
    return 0

def main():
    rank, world, use_cuda = setup()
    device = torch.device("cuda", rank % torch.cuda.device_count()) if use_cuda else torch.device("cpu")
    step = load_step()
    MAX_STEPS = int(os.environ.get("ELASTIC_DEMO_STEPS", "60"))
    while step < MAX_STEPS:
        # 模拟一个 training step：构造梯度张量并做 All-Reduce（梯度同步）
        g = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(g, op=dist.ReduceOp.SUM)   # 故障点会暴露在这里
        if rank == 0 and step % 5 == 0:
            torch.save({"step": step}, CKPT)        # rank0 周期 checkpoint
            print(f"[ckpt] step={step} saved, allreduce_sum={g.item():.0f}", flush=True)
        step += 1
        time.sleep(1)  # 放慢节奏，方便手动 kill
    print(f"[rank {rank}] DONE at step={step}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
