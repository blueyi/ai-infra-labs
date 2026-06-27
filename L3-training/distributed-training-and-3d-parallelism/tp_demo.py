# tp_demo.py
import os, torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    ngpu = torch.cuda.device_count()
    use_nccl = torch.cuda.is_available() and ngpu >= world
    backend = "nccl" if use_nccl else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    dev = f"cuda:{rank}" if use_nccl else "cpu"
    if use_nccl:
        torch.cuda.set_device(rank)

    H, Hf = 8, 16                      # hidden / ffn 维度
    torch.manual_seed(0)
    x = torch.randn(4, H, device=dev)  # 输入在所有 TP-rank 上是“复制”的

    # 列切：第一层权重按列(Hf)切成 world 份，每个 rank 拿一片
    shard = Hf // world
    A_full = torch.randn(H, Hf, device=dev)          # 同 seed → 各 rank 一致
    A_local = A_full[:, rank*shard:(rank+1)*shard]   # 本 rank 的列分片
    # 行切：第二层权重按行(Hf)切，行分片与 A 的列分片对齐
    B_full = torch.randn(Hf, H, device=dev)
    B_local = B_full[rank*shard:(rank+1)*shard, :]

    h = torch.relu(x @ A_local)        # 列切输出天然分片，前向无通信
    y_partial = h @ B_local            # 行切：本 rank 只算出“部分和”
    dist.all_reduce(y_partial)         # ← TP 的核心：对激活做 All-Reduce 求和

    if rank == 0:
        # 对照：单机完整计算的参考结果
        y_ref = torch.relu(x @ A_full) @ B_full
        err = (y_partial - y_ref).abs().max().item()
        print(f"[TP] backend={backend} max_err_vs_single={err:.2e}")  # 应≈0

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
