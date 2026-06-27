import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 块级编程模型：每个 program 负责输入矩阵的「一整行」
    row_idx = tl.program_id(axis=0)
    row_start = in_ptr + row_idx * in_row_stride

    # 一条语句向量化加载整行；mask 处理 n_cols 不是 BLOCK_SIZE 整数倍的尾部
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = row_start + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    # 数值稳定的 softmax：先减去行最大值，避免 exp 溢出
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # 写回结果
    out_row_start = out_ptr + row_idx * out_row_stride
    tl.store(out_row_start + col_offsets, softmax_out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    # BLOCK_SIZE 取大于等于列数的最近 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    # grid：启动 n_rows 个 program，每行一个
    grid = (n_rows,)
    kernel = softmax_kernel[grid](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out, kernel


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    x = torch.randn(1823, 781, device=dev)

    y_triton, kernel = triton_softmax(x)
    y_ref = torch.softmax(x, axis=1)

    # 正确性校验（解释模式下也能跑这一步）
    print("[correctness] max abs diff =",
          (y_triton - y_ref).abs().max().item())
