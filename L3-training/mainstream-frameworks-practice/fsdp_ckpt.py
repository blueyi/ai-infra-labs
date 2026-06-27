# fsdp_ckpt.py —— torchrun --nproc_per_node=2 fsdp_ckpt.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

CKPT_DIR = "ckpt_fsdp"


def setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ngpu = torch.cuda.device_count()
    use_nccl = torch.cuda.is_available() and ngpu >= world
    backend = "nccl" if use_nccl else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if use_nccl:
        torch.cuda.set_device(local_rank)
    return use_nccl, local_rank, world


class TinyTransformer(nn.Module):
    def __init__(self, d=64, layers=4):
        super().__init__()
        self.emb = nn.Linear(16, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, batch_first=True)
            for _ in range(layers)
        ])
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x).mean(dim=1)


def train_steps(model, opt, dev, steps=5):
    rank = dist.get_rank()
    for s in range(steps):
        x = torch.randn(8, 10, 16, device=dev)
        y = torch.randn(8, 1, device=dev)
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if rank == 0:
            print(f"[train] step {s} loss={loss.item():.4f}")


def main():
    use_nccl, local_rank, world = setup()
    rank = dist.get_rank()
    dev = torch.device(f"cuda:{local_rank}" if use_nccl else "cpu")
    torch.manual_seed(0)

    base = TinyTransformer()
    if use_nccl:
        base = base.to(dev)
    model = FSDP(base, device_id=local_rank if use_nccl else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_steps(model, opt, dev, steps=5)

    msd, osd = get_state_dict(model, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=CKPT_DIR)
    dist.barrier()
    if rank == 0:
        print(f"[save] sharded checkpoint -> {CKPT_DIR}/")

    model2 = FSDP(TinyTransformer().to(dev) if use_nccl else TinyTransformer(),
                  device_id=local_rank if use_nccl else None)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    msd2, osd2 = get_state_dict(model2, opt2)
    dcp.load({"model": msd2, "optim": osd2}, checkpoint_id=CKPT_DIR)
    set_state_dict(model2, opt2, model_state_dict=msd2, optim_state_dict=osd2)

    p1 = next(model.parameters()).detach()
    p2 = next(model2.parameters()).detach()
    same = torch.allclose(p1.cpu(), p2.cpu(), atol=1e-6)
    if rank == 0:
        print(f"[verify] params identical after restore: {same}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
