#!/usr/bin/env python3
"""L2.8 — file sharding + DistributedSampler overlap check."""
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def file_sharding_demo() -> None:
    shards = [f"shard_{i:03d}.jsonl" for i in range(16)]
    world_size = 4
    for rank in range(world_size):
        assigned = shards[rank::world_size]
        print(f"rank {rank}: {len(assigned)} files, first={assigned[0]}, last={assigned[-1]}")
    # verify no overlap
    seen = set()
    for rank in range(world_size):
        for f in shards[rank::world_size]:
            assert f not in seen, f"overlap on {f}"
            seen.add(f)
    assert len(seen) == len(shards)
    print("[ok] file sharding: 16 shards, 4 ranks, zero overlap")


def distributed_sampler_demo() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29507")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    data = TensorDataset(torch.arange(100))
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False)
    loader = DataLoader(data, batch_size=8, sampler=sampler)
    indices = []
    for (batch,) in loader:
        indices.extend(batch.tolist())

    if rank == 0:
        print(f"rank0 sample indices (first 16): {indices[:16]}")
    dist.barrier()
    # gather counts
    t = torch.tensor([len(indices)], dtype=torch.long)
    gather = [torch.zeros(1, dtype=torch.long) for _ in range(world)] if rank == 0 else None
    dist.gather(t, gather, dst=0)
    if rank == 0:
        total = sum(int(x.item()) for x in gather)
        print(f"[ok] distributed sampler: {total} samples across {world} ranks (expect 100)")
        assert total == 100
    dist.destroy_process_group()


def main() -> int:
    file_sharding_demo()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or "--dist" in __import__("sys").argv:
        distributed_sampler_demo()
    else:
        print("[info] run with: torchrun --nproc_per_node=2 shard_demo.py --dist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
