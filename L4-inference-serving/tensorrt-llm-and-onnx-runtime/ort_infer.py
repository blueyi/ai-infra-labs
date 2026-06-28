#!/usr/bin/env python3
"""L4.4 — ORT inference on tiny_lm.onnx (CPU EP by default)."""
import numpy as np
import onnxruntime as ort
from pathlib import Path


def main() -> int:
    model = Path("tiny_lm.onnx")
    if not model.is_file():
        raise SystemExit("missing tiny_lm.onnx — run export_tiny_lm.py first")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model), providers=providers)
    input_ids = np.random.randint(0, 1000, (2, 16), dtype=np.int64)
    logits = sess.run(None, {"input_ids": input_ids})[0]
    print(f"provider={sess.get_providers()[0]} logits_shape={logits.shape}")
    assert logits.shape == (2, 32000)
    print("[ok] ort infer lab passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
