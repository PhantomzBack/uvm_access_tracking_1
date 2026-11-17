#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Layer forward kernel: simulates realistic weight-activation compute
__global__ void transformer_layer_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ weights,
    int n,
    int compute_reps
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        float val = in[i];
        float w = weights[i];
        // Perform compute loop to simulate matrix-vector GEMV / MLP computation
        #pragma unroll 4
        for (int r = 0; r < compute_reps; ++r) {
            val = val * w + 0.001f * (float)r;
        }
        out[i] = val;
    }
}

static void read_fault_stats(long long* cpu_faults, long long* gpu_faults) {
    *cpu_faults = 0;
    *gpu_faults = 0;
    FILE* f = fopen("/proc/driver/nvidia-uvm/cpu/fault_stats", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "uvm_cpu_faults: %lld", cpu_faults) == 1) continue;
        if (sscanf(line, "uvm_gpu_faults: %lld", gpu_faults) == 1) continue;
    }
    fclose(f);
}

static void reset_fault_stats() {
    FILE* f = fopen("/proc/driver/nvidia-uvm/cpu/fault_stats", "w");
    if (f) {
        fprintf(f, "0\n");
        fclose(f);
    }
}

int main(int argc, char** argv) {
    const int NUM_LAYERS = 16;
    // 4M floats = 16 MB per layer -> 16 layers = 256 MB weights
    const int N = 4 * 1024 * 1024;
    const size_t LAYER_BYTES = N * sizeof(float);
    const int NUM_ITERS = 8; // 8 token generation steps
    const int COMPUTE_REPS = 32; // GEMV intensity

    printf("=================================================================\n");
    printf("  Oversubscribed Pipeline Benchmark\n");
    printf("  Layers: %d, Size per layer: %.1f MB (Total: %.1f MB)\n",
           NUM_LAYERS, (double)LAYER_BYTES / (1024.0 * 1024.0),
           (double)(NUM_LAYERS * LAYER_BYTES) / (1024.0 * 1024.0));
    printf("  Token Steps: %d, Compute Reps per element: %d\n", NUM_ITERS, COMPUTE_REPS);
    printf("=================================================================\n");

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_in, LAYER_BYTES));
    CUDA_CHECK(cudaMallocManaged(&d_out, LAYER_BYTES));
    for (int i = 0; i < N; ++i) {
        d_in[i] = 1.0f;
        d_out[i] = 0.0f;
    }
    // Activations stay on GPU
    CUDA_CHECK(cudaMemPrefetchAsync(d_in, LAYER_BYTES, device_id, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(d_out, LAYER_BYTES, device_id, 0));

    // Allocate weights as Managed Memory, preferred on CPU
    float* weights[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; ++l) {
        CUDA_CHECK(cudaMallocManaged(&weights[l], LAYER_BYTES));
        for (int i = 0; i < N; ++i) {
            weights[l][i] = 1.0001f + 0.00001f * (l + 1);
        }
        // Designate CPU as preferred location so that weights reside in host DRAM
        CUDA_CHECK(cudaMemAdvise(weights[l], LAYER_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        // Evict/prefetch to CPU host memory initially
        CUDA_CHECK(cudaMemPrefetchAsync(weights[l], LAYER_BYTES, cudaCpuDeviceId, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block(256);
    dim3 grid(1024);

    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    // Reset UVM fault counters before timed benchmark
    reset_fault_stats();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, compute_stream));

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int l = 0; l < NUM_LAYERS; ++l) {
            const float* in_ptr = (l % 2 == 0) ? d_in : d_out;
            float* out_ptr = (l % 2 == 0) ? d_out : d_in;

            transformer_layer_kernel<<<grid, block, 0, compute_stream>>>(
                in_ptr, out_ptr, weights[l], N, COMPUTE_REPS
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    long long cpu_faults = 0, gpu_faults = 0;
    read_fault_stats(&cpu_faults, &gpu_faults);

    printf("\n----------------------- Results -----------------------\n");
    printf("Total Elapsed Time:      %.3f ms\n", total_ms);
    printf("Avg Time per Token Step: %.3f ms\n", total_ms / NUM_ITERS);
    printf("GPU Page Faults:         %lld\n", gpu_faults);
    printf("CPU Page Faults:         %lld\n", cpu_faults);
    printf("Verification out[0]:     %.4f\n", d_out[0]);
    printf("-------------------------------------------------------\n");

    for (int l = 0; l < NUM_LAYERS; ++l) {
        CUDA_CHECK(cudaFree(weights[l]));
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
