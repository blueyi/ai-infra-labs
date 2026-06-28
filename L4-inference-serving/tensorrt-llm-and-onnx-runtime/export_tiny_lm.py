#!/usr/bin/env python3
"""L4.4 — export minimal LM to ONNX (opset 17, dynamic axes)."""
import torch
import torch.nn as nn


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32000, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.linear(x.mean(dim=1))


def main() -> int:
    model = TinyLM().eval()
    dummy = torch.randint(0, 1000, (1, 16))
    out_path = "tiny_lm.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"[ok] exported {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
