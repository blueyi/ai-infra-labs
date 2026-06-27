#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16          // 每个 Block 处理 16x16 的输出 tile
#define N    1024         // 方阵边长（C = A * B，均为 N x N）

// ---------- Kernel 1：Naive GEMM（每个输出元素直读全局显存）----------
__global__ void gemm_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += A[row * n + k] * B[k * n + col];  // 每次乘加都读两次 HBM
        C[row * n + col] = acc;
    }
}

// ---------- Kernel 2：Tiled GEMM（Shared Memory 分块复用）----------
// 注意 +1 列 padding 以免除 bank conflict
__global__ void gemm_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    // 沿 K 维分块：每次把 A、B 的一个 tile 搬进 Shared Memory 复用
    for (int t = 0; t < n / TILE; ++t) {
        As[ty][tx] = A[row * n + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * n + col];
        __syncthreads();                       // 等全 Block 搬完

        #pragma unroll
        for (int k = 0; k < TILE; ++k)         // 这 TILE 次乘加全部命中 Shared Memory
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                       // 等全 Block 算完再搬下一块
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

// ---------- 计时辅助 ----------
float time_kernel(void (*launch)(const float*, const float*, float*, int),
                  const float* dA, const float* dB, float* dC, int n,
                  dim3 grid, dim3 block) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // 预热
    launch<<<grid, block>>>(dA, dB, dC, n); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) launch<<<grid, block>>>(dA, dB, dC, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 20.0f;   // 单次平均毫秒
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    float t_naive = time_kernel(gemm_naive, dA, dB, dC, N, grid, block);
    float t_tiled = time_kernel(gemm_tiled, dA, dB, dC, N, grid, block);

    double gflop = 2.0 * N * N * N / 1e9;   // 2*N^3 次浮点运算
    printf("Naive : %.3f ms  | %.1f GFLOPS\n", t_naive, gflop / (t_naive / 1e3));
    printf("Tiled : %.3f ms  | %.1f GFLOPS\n", t_tiled, gflop / (t_tiled / 1e3));
    printf("Speedup (naive/tiled) = %.2fx\n", t_naive / t_tiled);

    free(hA); free(hB); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
