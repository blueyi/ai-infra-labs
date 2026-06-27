import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    # 每个 program 负责一个 (batch, head) 的一个 Q 行块（BLOCK_M 行）
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // tl.num_programs(2) if False else off_bh  # 简化：bh 合并维
    # 基址偏移（按 batch*head 展平）
    q_base = Q + off_bh * stride_qh
    k_base = K + off_bh * stride_kh
    v_base = V + off_bh * stride_vh
    o_base = O + off_bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # 载入 Q 行块到 SRAM（寄存器/共享内存由 Triton 调度）
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # online softmax 状态：running max m、running sum ℓ、累加输出 acc
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # 内循环：扫过所有 K/V 列块
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # 本块分数 S = scale * Q·Kᵀ
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))

        # —— online softmax 增量更新 ——
        m_block = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_block)
        p = tl.exp(s - m_new[:, None])              # 本块概率（未归一）
        alpha = tl.exp(m_i - m_new)                 # 旧状态 rescale 因子
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 收尾归一化：O = acc / ℓ
    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_attn_triton(q, k, v):
    # q,k,v: [B, H, N, D]，合并 B*H 为一维网格
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    q2 = q.reshape(B * H, N, D)
    k2 = k.reshape(B * H, N, D)
    v2 = v.reshape(B * H, N, D)
    o2 = o.reshape(B * H, N, D)
    grid = (triton.cdiv(N, BLOCK_M), B * H, 1)
    _flash_attn_fwd[grid](
        q2, k2, v2, o2,
        *q2.stride()[:1], q2.stride(0), q2.stride(1), q2.stride(2),
        *k2.stride()[:1], k2.stride(0), k2.stride(1), k2.stride(2),
        *v2.stride()[:1], v2.stride(0), v2.stride(1), v2.stride(2),
        *o2.stride()[:1], o2.stride(0), o2.stride(1), o2.stride(2),
        N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
    )
    return o


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Triton kernel 需要 NVIDIA GPU"
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 1024, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    o_triton = flash_attn_triton(q, k, v)
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # —— 对数值 ——
    print("allclose:", torch.allclose(o_triton, o_sdpa, atol=1e-2, rtol=1e-2))
    print("max abs diff:", (o_triton - o_sdpa).abs().max().item())

    import time

    def bench(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t_triton = bench(lambda: flash_attn_triton(q, k, v))
    t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"Triton flash : {t_triton:.3f} ms")
    print(f"PyTorch SDPA : {t_sdpa:.3f} ms")
