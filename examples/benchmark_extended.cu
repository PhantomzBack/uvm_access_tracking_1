#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "tracking.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ── 0. SAXPY ─────────────────────────────────────────────────────────────────
// Three coalesced streams (x read, y read+write). All three pointers are
// affine AddRecs → BatchMarkAccess fires for each, all hoisted to preheader.
__global__ void test_saxpy(float* x, float* y, float alpha, int n) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)n; i += grid_stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

// ── 1. Reduction ──────────────────────────────────────────────────────────────
// One coalesced read into shared memory (skipped by pass: addrspace 3),
// then a tree reduction inside shared mem, then one invariant atomic write.
// Pass sees: data[i] → Variant → BatchMarkAccess; result pointer → Invariant → hoist.
__global__ void test_reduction(int* data, int* result, int n) {
    extern __shared__ int sdata[];
    int    tid = threadIdx.x;
    size_t i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < (size_t)n) ? data[i] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// ── 2. Histogram ─────────────────────────────────────────────────────────────
// Coalesced sequential read of data (Variant → BatchMarkAccess), then a
// non-affine atomic scatter into hist (index = data[i] % bins → Unknown → fallback).
// AtomicRMW is not a Load/Store so the hist write isn't tracked directly;
// only the load of data[i] is instrumented.
__global__ void test_histogram(int* data, int* hist, int n, int bins) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)n; i += grid_stride) {
        atomicAdd(&hist[data[i] % bins], 1);
    }
}

// ── 3. Out-of-place Matrix Transpose ─────────────────────────────────────────
// Uses a shared-memory tile to avoid uncoalesced global writes.
// Without a grid-stride loop the memory ops fall to per-instruction fallback
// (no loop → LI.getLoopFor returns null). Tests the thread-local last_page_cache.
#define TILE 32
__global__ void test_transpose(int* in, int* out, int N) {
    __shared__ int tile[TILE][TILE + 1];  // +1 avoids shared-mem bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

// ── 4. Dense Matrix-Vector Multiply (GEMV) ───────────────────────────────────
// Outer grid-stride loop over rows; inner dense loop over columns.
// Inner loop:
//   A[row*N+k] → Variant, SCEV BTC = N-1 (computable: start=0 step=1) → BatchMarkAccess
//   x[k]       → Variant, same → BatchMarkAccess, hoisted to outer preheader (x invariant there)
//   y[row]     → Invariant for inner loop → hoistMarkPage
// Outer loop:
//   y[row]     → Variant → BatchMarkAccess with runtime count
__global__ void test_gemv(float* A, float* x, float* y, int M, int N) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < (size_t)M; row += grid_stride) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * x[k];
        }
        y[row] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <test_id>\n  0=saxpy 1=reduction 2=histogram 3=transpose 4=gemv\n", argv[0]);
        return 1;
    }
    int test_id = atoi(argv[1]);

    const int N   = 1 << 20;  // 1M elements (base)
    const int MAT = 2048;     // matrix side for transpose and GEMV

    float *d_x, *d_y, *d_A, *d_gx, *d_gy;
    int   *d_data, *d_hist, *d_result, *d_ti, *d_to;

    CUDA_CHECK(cudaMallocManaged(&d_x,    N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_y,    N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_A,    (size_t)MAT * MAT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_gx,   MAT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_gy,   MAT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_ti,   (size_t)MAT * MAT * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_to,   (size_t)MAT * MAT * sizeof(int)));

    // Initialise with deterministic values
    std::mt19937 rng(42);
    for (int i = 0; i < N; i++) {
        d_x[i] = (float)(rng() % 1000) / 100.0f;
        d_y[i] = (float)(rng() % 1000) / 100.0f;
        d_data[i] = rng() % 256;
    }
    for (int i = 0; i < MAT * MAT; i++) {
        d_A[i]  = (float)(rng() % 100) / 10.0f;
        d_ti[i] = rng() % 1000;
    }
    for (int i = 0; i < MAT; i++) d_gx[i] = (float)(rng() % 100) / 10.0f;
    memset(d_hist,   0, 256 * sizeof(int));
    memset(d_result, 0, sizeof(int));

#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif

    dim3 block_1d(256);
    dim3 grid_1d(256);
    dim3 block_2d(TILE, TILE);
    dim3 grid_2d((MAT + TILE - 1) / TILE, (MAT + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto launch = [&](int id) {
        switch (id) {
            case 0:
                printf("Selected saxpy");
                test_saxpy<<<grid_1d, block_1d>>>(d_x, d_y, 2.5f, N);
                break;
            case 1:
                printf("Selected reduction");
                test_reduction<<<grid_1d, block_1d, block_1d.x * sizeof(int)>>>(d_data, d_result, N);
                break;
            case 2:
                printf("Selected histogram");
                test_histogram<<<grid_1d, block_1d>>>(d_data, d_hist, N, 256);
                break;
            case 3:
                printf("Selected transpose");
                test_transpose<<<grid_2d, block_2d>>>(d_ti, d_to, MAT);
                break;
            case 4:
                printf("Selected gemv");
                test_gemv<<<grid_1d, block_1d>>>(d_A, d_gx, d_gy, MAT, MAT);
                break;
        }
    };

    launch(test_id);  // warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch(test_id);  // timed
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nBENCHMARK_TIME: %.3f ms\n", ms);

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_l1);
#endif

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_A);
    cudaFree(d_gx); cudaFree(d_gy);
    cudaFree(d_data); cudaFree(d_hist); cudaFree(d_result);
    cudaFree(d_ti); cudaFree(d_to);
    return 0;
}
