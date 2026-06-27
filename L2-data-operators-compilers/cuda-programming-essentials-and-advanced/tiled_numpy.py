import numpy as np

N, TILE = 512, 64
A = np.ones((N, N), dtype=np.float32)
B = np.full((N, N), 2.0, dtype=np.float32)
C = np.zeros((N, N), dtype=np.float32)

# 模拟 GPU tiled GEMM 的三层分块循环：搬一块 -> 复用一块
for i0 in range(0, N, TILE):
    for j0 in range(0, N, TILE):
        acc = np.zeros((TILE, TILE), dtype=np.float32)
        for k0 in range(0, N, TILE):
            a_tile = A[i0:i0+TILE, k0:k0+TILE]   # 对应搬进 Shared Memory 的 As
            b_tile = B[k0:k0+TILE, j0:j0+TILE]   # 对应 Bs
            acc += a_tile @ b_tile               # 这一块数据被 TILE^2 次乘加复用
        C[i0:i0+TILE, j0:j0+TILE] = acc

assert np.allclose(C, A @ B)
print("分块结果与直接 matmul 一致；a_tile/b_tile 即对应 GPU 的 Shared Memory 复用单元")
