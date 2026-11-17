// ═══════════════════════════════════════════════════════════════════════════════
// Temporal Access Tracking Microbenchmark:
// Comparing:
//   1. Clean Baseline (No instrumentation)
//   2. Tripwire using Page Faults (GPU accesses host-resident UVM page, triggering MMU fault)
//   3. Tiled Loop with Low-Latency Progress Signaling (PCIe-coherent system atomic st.global.release.sys)
//
// Evaluated across 3 distinct kernel patterns:
//   - Workload A: Iterative Time-Stepping Stencil (Multi-epoch outer loop)
//   - Workload B: Memory-Streaming Array Chunking (Bandwidth-bound loop)
//   - Workload C: Dense Matrix Multiplication K-Tiled Reduction (Compute-bound loop)
// ═══════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <emmintrin.h> // _mm_pause

// Helper: high-resolution CPU timestamp in microseconds
static inline uint64_t get_time_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// ─────────────────────────────────────────────────────────────────────────────
// WORKLOAD A: Iterative Time-Stepping Stencil (Diffusion/Heat)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stencil_clean(float* d_in, float* d_out, int N, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int s = 0; s < steps; ++s) {
        if (idx > 0 && idx < N - 1) {
            d_out[idx] = 0.25f * (d_in[idx - 1] + 2.0f * d_in[idx] + d_in[idx + 1]);
        }
        __syncthreads();
        // Ping-pong in local thread registers/pointers
        float* tmp = d_in; d_in = d_out; d_out = tmp;
    }
}

__global__ void stencil_tripwire(float* d_in, float* d_out, int N, int steps, int tile_step, volatile int* tripwire_pages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int s = 0; s < steps; ++s) {
        if (idx > 0 && idx < N - 1) {
            d_out[idx] = 0.25f * (d_in[idx - 1] + 2.0f * d_in[idx] + d_in[idx + 1]);
        }
        __syncthreads();
        float* tmp = d_in; d_in = d_out; d_out = tmp;

        // Tripwire: at every tile milestone, Thread 0 touches an un-prefetched CPU page
        if ((s + 1) % tile_step == 0) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                int tile_id = s / tile_step;
                // Writing to tripwire page causes hardware MMU fault interrupt
                tripwire_pages[tile_id * 1024] = tile_id + 1;
            }
            __syncthreads();
        }
    }
}

__global__ void stencil_tiled_signal(float* d_in, float* d_out, int N, int steps, int tile_step, volatile uint32_t* host_status) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int s = 0; s < steps; ++s) {
        if (idx > 0 && idx < N - 1) {
            d_out[idx] = 0.25f * (d_in[idx - 1] + 2.0f * d_in[idx] + d_in[idx + 1]);
        }
        __syncthreads();
        float* tmp = d_in; d_in = d_out; d_out = tmp;

        if ((s + 1) % tile_step == 0) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                int milestone_id = s / tile_step;
                uint32_t val = (uint32_t)(milestone_id + 1);
                asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(&host_status[milestone_id]), "r"(val));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKLOAD B: Memory-Streaming Array Chunking (Bandwidth-Bound Loop)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stream_clean(const float* d_in, float* d_out, int N, int num_passes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < num_passes; ++p) {
        for (int i = tid; i < N; i += stride) {
            d_out[i] = d_in[i] * 1.0001f + 0.05f;
        }
    }
}

__global__ void stream_tripwire(const float* d_in, float* d_out, int N, int num_passes, volatile int* tripwire_pages) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < num_passes; ++p) {
        for (int i = tid; i < N; i += stride) {
            d_out[i] = d_in[i] * 1.0001f + 0.05f;
        }
        // At pass boundary, trigger MMU page fault tripwire
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            tripwire_pages[p * 1024] = p + 1;
        }
        __syncthreads();
    }
}

__global__ void stream_tiled_signal(const float* d_in, float* d_out, int N, int num_passes, volatile uint32_t* host_status) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < num_passes; ++p) {
        for (int i = tid; i < N; i += stride) {
            d_out[i] = d_in[i] * 1.0001f + 0.05f;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            uint32_t val = (uint32_t)(p + 1);
            asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(&host_status[p]), "r"(val));
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKLOAD C: Tiled Matrix Multiplication (Compute-Bound Loop)
// ─────────────────────────────────────────────────────────────────────────────
#define TILE_DIM 32

__global__ void matmul_clean(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < K / TILE_DIM; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < M && (t * TILE_DIM + threadIdx.x) < K) ? A[row * K + t * TILE_DIM + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = ((t * TILE_DIM + threadIdx.y) < K && col < N) ? B[(t * TILE_DIM + threadIdx.y) * N + col] : 0.0f;
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

__global__ void matmul_tripwire(const float* A, const float* B, float* C, int M, int N, int K, int k_tile_milestone, volatile int* tripwire_pages) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = K / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < M && (t * TILE_DIM + threadIdx.x) < K) ? A[row * K + t * TILE_DIM + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = ((t * TILE_DIM + threadIdx.y) < K && col < N) ? B[(t * TILE_DIM + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();

        if ((t + 1) % k_tile_milestone == 0) {
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                int milestone_id = t / k_tile_milestone;
                tripwire_pages[milestone_id * 1024] = milestone_id + 1;
            }
            __syncthreads();
        }
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled_signal(const float* A, const float* B, float* C, int M, int N, int K, int k_tile_milestone, volatile uint32_t* host_status) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = K / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < M && (t * TILE_DIM + threadIdx.x) < K) ? A[row * K + t * TILE_DIM + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = ((t * TILE_DIM + threadIdx.y) < K && col < N) ? B[(t * TILE_DIM + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();

        if ((t + 1) % k_tile_milestone == 0) {
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                int milestone_id = t / k_tile_milestone;
                uint32_t val = (uint32_t)(milestone_id + 1);
                asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(&host_status[milestone_id]), "r"(val));
            }
        }
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Runner and Polling Infrastructure
// ─────────────────────────────────────────────────────────────────────────────

struct BenchmarkResult {
    std::string name;
    float clean_ms;
    float tripwire_ms;
    float tiled_signal_ms;
    float tripwire_overhead_pct;
    float tiled_overhead_pct;
    int num_milestones;
};

void run_stencil_benchmark(BenchmarkResult& res, int num_milestones = 16) {
    res.name = "Stencil (Time-Stepping)";
    res.num_milestones = num_milestones;

    const int N = 4 * 1024 * 1024; // 4M elements (16 MB)
    const int total_steps = 128;
    const int tile_step = total_steps / num_milestones;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Allocate tripwire pages via UVM Managed Memory
    // Prefetch to CPU so that GPU access triggers an MMU demand fault interrupt
    int* tripwire_pages = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&tripwire_pages, num_milestones * 4096));
    CUDA_CHECK(cudaMemAdvise(tripwire_pages, num_milestones * 4096, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

    // Allocate mapped zero-copy status array for low-latency signaling
    uint32_t* host_status = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&host_status, num_milestones * sizeof(uint32_t), cudaHostAllocMapped));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. Warmup & Clean Baseline
    for (int i = 0; i < 5; ++i) stencil_clean<<<grid, block>>>(d_in, d_out, N, total_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) stencil_clean<<<grid, block>>>(d_in, d_out, N, total_steps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.clean_ms, start, stop));
    res.clean_ms /= 10.0f;

    // 2. Tripwire Run (MMU page faults)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        // Evict tripwire pages to CPU before launch so every pass faults
        CUDA_CHECK(cudaMemPrefetchAsync(tripwire_pages, num_milestones * 4096, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        stencil_tripwire<<<grid, block>>>(d_in, d_out, N, total_steps, tile_step, tripwire_pages);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tripwire_ms, start, stop));
    res.tripwire_ms /= 10.0f;

    // 3. Tiled + Low-Latency Host Polling Run
    std::vector<uint64_t> milestone_timestamps(num_milestones, 0);
    std::atomic<bool> stop_poller{false};

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        memset(host_status, 0, num_milestones * sizeof(uint32_t));
        
        // Spawn background polling thread to listen for milestones
        std::thread poller([&]() {
            for (int m = 0; m < num_milestones && !stop_poller.load(std::memory_order_relaxed); ++m) {
                while (__atomic_load_n(&host_status[m], __ATOMIC_ACQUIRE) == 0 && !stop_poller.load(std::memory_order_relaxed)) {
                    _mm_pause();
                }
                milestone_timestamps[m] = get_time_us();
            }
        });

        stencil_tiled_signal<<<grid, block>>>(d_in, d_out, N, total_steps, tile_step, host_status);
        CUDA_CHECK(cudaDeviceSynchronize());
        stop_poller = true;
        if (poller.joinable()) poller.join();
        stop_poller = false;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tiled_signal_ms, start, stop));
    res.tiled_signal_ms /= 10.0f;

    res.tripwire_overhead_pct = ((res.tripwire_ms - res.clean_ms) / res.clean_ms) * 100.0f;
    res.tiled_overhead_pct = ((res.tiled_signal_ms - res.clean_ms) / res.clean_ms) * 100.0f;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(tripwire_pages));
    CUDA_CHECK(cudaFreeHost(host_status));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_stream_benchmark(BenchmarkResult& res, int num_milestones = 16) {
    res.name = "Memory Streaming (Bandwidth-Bound)";
    res.num_milestones = num_milestones;

    const int N = 16 * 1024 * 1024; // 16M floats (64 MB)
    const int passes = num_milestones;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    int* tripwire_pages = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&tripwire_pages, num_milestones * 4096));
    CUDA_CHECK(cudaMemAdvise(tripwire_pages, num_milestones * 4096, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

    uint32_t* host_status = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&host_status, num_milestones * sizeof(uint32_t), cudaHostAllocMapped));

    dim3 block(256);
    dim3 grid(1024); // Grid-stride loop

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. Clean
    for (int i = 0; i < 5; ++i) stream_clean<<<grid, block>>>(d_in, d_out, N, passes);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) stream_clean<<<grid, block>>>(d_in, d_out, N, passes);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.clean_ms, start, stop));
    res.clean_ms /= 10.0f;

    // 2. Tripwire
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemPrefetchAsync(tripwire_pages, num_milestones * 4096, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        stream_tripwire<<<grid, block>>>(d_in, d_out, N, passes, tripwire_pages);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tripwire_ms, start, stop));
    res.tripwire_ms /= 10.0f;

    // 3. Tiled Signal
    std::atomic<bool> stop_poller{false};
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        memset(host_status, 0, num_milestones * sizeof(uint32_t));
        std::thread poller([&]() {
            for (int m = 0; m < num_milestones && !stop_poller.load(std::memory_order_relaxed); ++m) {
                while (__atomic_load_n(&host_status[m], __ATOMIC_ACQUIRE) == 0 && !stop_poller.load(std::memory_order_relaxed)) {
                    _mm_pause();
                }
            }
        });

        stream_tiled_signal<<<grid, block>>>(d_in, d_out, N, passes, host_status);
        CUDA_CHECK(cudaDeviceSynchronize());
        stop_poller = true;
        if (poller.joinable()) poller.join();
        stop_poller = false;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tiled_signal_ms, start, stop));
    res.tiled_signal_ms /= 10.0f;

    res.tripwire_overhead_pct = ((res.tripwire_ms - res.clean_ms) / res.clean_ms) * 100.0f;
    res.tiled_overhead_pct = ((res.tiled_signal_ms - res.clean_ms) / res.clean_ms) * 100.0f;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(tripwire_pages));
    CUDA_CHECK(cudaFreeHost(host_status));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_matmul_benchmark(BenchmarkResult& res, int num_milestones = 16) {
    res.name = "Matmul (Compute-Bound GEMM)";
    res.num_milestones = num_milestones;

    const int M = 2048, N = 2048, K = 2048;
    const int total_k_tiles = K / TILE_DIM; // 64 tiles
    const int k_tile_milestone = total_k_tiles / num_milestones;

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, M * N * sizeof(float)));

    int* tripwire_pages = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&tripwire_pages, num_milestones * 4096));
    CUDA_CHECK(cudaMemAdvise(tripwire_pages, num_milestones * 4096, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

    uint32_t* host_status = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&host_status, num_milestones * sizeof(uint32_t), cudaHostAllocMapped));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. Clean
    for (int i = 0; i < 5; ++i) matmul_clean<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) matmul_clean<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.clean_ms, start, stop));
    res.clean_ms /= 10.0f;

    // 2. Tripwire
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemPrefetchAsync(tripwire_pages, num_milestones * 4096, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        matmul_tripwire<<<grid, block>>>(A, B, C, M, N, K, k_tile_milestone, tripwire_pages);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tripwire_ms, start, stop));
    res.tripwire_ms /= 10.0f;

    // 3. Tiled Signal
    std::atomic<bool> stop_poller{false};
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        memset(host_status, 0, num_milestones * sizeof(uint32_t));
        std::thread poller([&]() {
            for (int m = 0; m < num_milestones && !stop_poller.load(std::memory_order_relaxed); ++m) {
                while (__atomic_load_n(&host_status[m], __ATOMIC_ACQUIRE) == 0 && !stop_poller.load(std::memory_order_relaxed)) {
                    _mm_pause();
                }
            }
        });

        matmul_tiled_signal<<<grid, block>>>(A, B, C, M, N, K, k_tile_milestone, host_status);
        CUDA_CHECK(cudaDeviceSynchronize());
        stop_poller = true;
        if (poller.joinable()) poller.join();
        stop_poller = false;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.tiled_signal_ms, start, stop));
    res.tiled_signal_ms /= 10.0f;

    res.tripwire_overhead_pct = ((res.tripwire_ms - res.clean_ms) / res.clean_ms) * 100.0f;
    res.tiled_overhead_pct = ((res.tiled_signal_ms - res.clean_ms) / res.clean_ms) * 100.0f;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(tripwire_pages));
    CUDA_CHECK(cudaFreeHost(host_status));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int milestones = 16;
    if (argc > 1) {
        milestones = atoi(argv[1]);
        if (milestones <= 0) milestones = 16;
    }

    printf("=========================================================================================\n");
    printf(" TEMPORAL ACCESS TRACKING OVERHEAD BENCHMARK (NVIDIA A40)\n");
    printf(" Milestones / Epochs per kernel: %d\n", milestones);
    printf("=========================================================================================\n\n");

    std::vector<BenchmarkResult> results(3);
    run_stencil_benchmark(results[0], milestones);
    run_stream_benchmark(results[1], milestones);
    run_matmul_benchmark(results[2], milestones);

    printf("%-35s | %-12s | %-12s | %-12s | %-14s | %-14s\n",
           "WORKLOAD", "CLEAN (ms)", "TRIPWIRE (ms)", "TILED (ms)", "TRIPWIRE OVERHEAD", "TILED OVERHEAD");
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    for (const auto& r : results) {
        printf("%-35s | %10.3f ms | %10.3f ms | %10.3f ms | %+13.2f%% | %+13.2f%%\n",
               r.name.c_str(), r.clean_ms, r.tripwire_ms, r.tiled_signal_ms,
               r.tripwire_overhead_pct, r.tiled_overhead_pct);
    }
    printf("=========================================================================================\n");
    return 0;
}
