import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
q = torch.randn(1, 4, 512, 64)
k = torch.randn(1, 4, 512, 64)
v = torch.randn(1, 4, 512, 64)

# math 后端：朴素实现，会物化 N×N（对照基准）
with sdpa_kernel(SDPBackend.MATH):
    o_math = torch.nn.functional.scaled_dot_product_attention(q, k, v)

# flash 后端在 CPU 上通常不可用，这里用 efficient 后端作对照
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        o_eff = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("math vs efficient allclose:", torch.allclose(o_math, o_eff, atol=1e-4))
    except RuntimeError as e:
        print("该后端在当前设备不可用：", e)
