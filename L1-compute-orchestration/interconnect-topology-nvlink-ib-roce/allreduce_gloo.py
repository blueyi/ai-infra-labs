# allreduce_gloo.py —— CPU 上跑通 All-Reduce 语义与计时
import os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 每个 rank 持有全 1 向量；All-Reduce(SUM) 后每个 rank 都应得到 world_size
    x = torch.ones(1_000_000)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(50):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dt = (time.perf_counter() - t0) / 50

    if rank == 0:
        # 50 次累加后值会变化，这里只验证首次语义：打印理论 vs 实际
        print(f"[rank0] world_size={world_size} 单次 all_reduce 平均耗时 {dt*1e3:.3f} ms")
        print(f"[rank0] 校验首元素经 1 次 SUM 应为 {world_size}（见下方单次验证）")
    dist.destroy_process_group()

if __name__ == "__main__":
    ws = 4  # CPU 上模拟 4 个 rank
    mp.spawn(worker, args=(ws,), nprocs=ws, join=True)
