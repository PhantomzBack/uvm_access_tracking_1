#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fstream>
#include "common.h"
#include "tracking.h"

// ─────────────────────────────────────────────────────────────────────────────
// End-to-End Verification & Benchmark:
// Static Analysis (LLVM SCEV Bytecode) + Host-Side Tripwire Temporal Tracking
// ─────────────────────────────────────────────────────────────────────────────

#define TILE_DIM 32

// 1. Tiled Matrix Multiplication (with outer K-tile loop)
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = K / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 2. Multi-Epoch Streaming Stencil Kernel (with outer epoch loop)
__global__ void iterative_stencil_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int num_epochs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = tid; i < n; i += stride) {
            float prev = (i > 0) ? in[i - 1] : in[i];
            float curr = in[i];
            float next = (i + 1 < n) ? in[i + 1] : in[i];
            out[i] = 0.25f * prev + 0.50f * curr + 0.25f * next + 0.001f * (float)epoch;
        }
        __syncthreads();
    }
}

static uint64_t count_logged_pages(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return 0;

    PageLogHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || hdr.magic != PAGELOG_MAGIC) return 0;

    std::vector<PageLogIndexEntry> index(hdr.num_leaves);
    f.read(reinterpret_cast<char*>(index.data()), hdr.num_leaves * sizeof(PageLogIndexEntry));

    uint64_t total_pages = 0;
    std::vector<uint64_t> bitmap(hdr.l3_bytes / 8);
    for (uint64_t k = 0; k < hdr.num_leaves; ++k) {
        f.seekg(index[k].offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(bitmap.data()), hdr.l3_bytes);
        for (uint64_t w : bitmap) {
            total_pages += __builtin_popcountll(w);
        }
    }
    return total_pages;
}

static uint64_t expected_pages_for_range(uintptr_t base, size_t size_bytes) {
    if (size_bytes == 0) return 0;
    uintptr_t start_page = base >> 12;
    uintptr_t end_page = (base + size_bytes - 1) >> 12;
    return (end_page - start_page + 1);
}

extern "C" void uvm_export_static_temporal_trace(const char* filename);

int main(int argc, char** argv) {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024; // 32 outer loop tiles (TILE_DIM = 32)
    const size_t MAT_BYTES = (size_t)M * N * sizeof(float); // 4 MB per matrix (1024 pages)

    const int STENCIL_N = 2 * 1024 * 1024; // 8 MB per buffer (2048 pages)
    const size_t STENCIL_BYTES = (size_t)STENCIL_N * sizeof(float);
    const int STENCIL_EPOCHS = 16;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *d_S_in = nullptr, *d_S_out = nullptr;

    CUDA_CHECK(cudaMallocManaged(&d_A, MAT_BYTES));
    CUDA_CHECK(cudaMallocManaged(&d_B, MAT_BYTES));
    CUDA_CHECK(cudaMallocManaged(&d_C, MAT_BYTES));
    CUDA_CHECK(cudaMallocManaged(&d_S_in, STENCIL_BYTES));
    CUDA_CHECK(cudaMallocManaged(&d_S_out, STENCIL_BYTES));

    for (size_t i = 0; i < (size_t)M * N; ++i) {
        d_A[i] = 1.0f;
        d_B[i] = 0.5f;
        d_C[i] = 0.0f;
    }
    for (int i = 0; i < STENCIL_N; ++i) {
        d_S_in[i] = 2.0f;
        d_S_out[i] = 0.0f;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef TRACKING_ENABLED
    void*** d_l1 = nullptr;
    init_tracking(&d_l1);
    uvm_temporal_reset();
#endif

    dim3 mm_block(TILE_DIM, TILE_DIM);
    dim3 mm_grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    dim3 st_block(256);
    dim3 st_grid(256);

    // Warmup
    tiled_matmul_kernel<<<mm_grid, mm_block>>>(d_A, d_B, d_C, M, N, K);
    iterative_stencil_kernel<<<st_grid, st_block>>>(d_S_in, d_S_out, STENCIL_N, STENCIL_EPOCHS);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef TRACKING_ENABLED
    uvm_temporal_reset();
#endif

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int RUNS = 5;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < RUNS; ++r) {
        tiled_matmul_kernel<<<mm_grid, mm_block>>>(d_A, d_B, d_C, M, N, K);
        iterative_stencil_kernel<<<st_grid, st_block>>>(d_S_in, d_S_out, STENCIL_N, STENCIL_EPOCHS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / RUNS;

    uint64_t exp_pages =
        expected_pages_for_range((uintptr_t)d_A, MAT_BYTES) +
        expected_pages_for_range((uintptr_t)d_B, MAT_BYTES) +
        expected_pages_for_range((uintptr_t)d_C, MAT_BYTES) +
        expected_pages_for_range((uintptr_t)d_S_in, STENCIL_BYTES) +
        expected_pages_for_range((uintptr_t)d_S_out, STENCIL_BYTES);

    uint64_t logged_pages = 0;
    uint32_t temporal_milestones = 0;
    uint32_t mailbox_seq = 0;

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "static_tripwire_pages.bin");
    logged_pages = count_logged_pages("static_tripwire_pages.bin");
    temporal_milestones = uvm_temporal_get_count();
    UvmTripwireMailbox* mb = uvm_get_tripwire_mailbox();
    if (mb) {
        mailbox_seq = mb->seq_num;
    }
    uvm_temporal_export_json("temporal_log.json");
#endif

    printf("=================================================================\n");
    printf("  Static Analysis + Tripwire Temporal Tracking E2E Results\n");
    printf("=================================================================\n");
    printf("Avg Pipeline Time (per iter): %.4f ms\n", avg_ms);
    printf("Expected Accessed Pages:      %llu\n", (unsigned long long)exp_pages);
    printf("Tracked Pages in Bitmap:      %llu (%.1f%% accuracy)\n",
           (unsigned long long)logged_pages,
           exp_pages ? (100.0 * (double)logged_pages / (double)exp_pages) : 0.0);
    printf("Temporal Milestones Logged:   %u (per run: %u)\n",
           temporal_milestones, temporal_milestones / RUNS);
    printf("Host Tripwire Mailbox Seq:    %u\n", mailbox_seq);
    printf("Verify C[0]=%.2f, S_out[0]=%.4f\n", d_C[0], d_S_out[0]);
    printf("=================================================================\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_S_in));
    CUDA_CHECK(cudaFree(d_S_out));
    return 0;
}
