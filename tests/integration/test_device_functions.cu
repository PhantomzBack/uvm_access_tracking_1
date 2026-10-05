#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "tracking.h"

// Non-inlined device functions performing memory accesses
__device__ __noinline__ float helper_read_val(const float* in, int idx) {
    return in[idx];
}

__device__ __noinline__ void helper_write_val(float* out, int idx, float val) {
    out[idx] = val * 2.0f;
}

// Kernel calling device functions
__global__ void kernel_with_device_calls(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = helper_read_val(in, idx);
        helper_write_val(out, idx, val);
    }
}

int main() {
    const int N = 1024 * 1024; // 1M floats = 4MB
    size_t size = N * sizeof(float);

    void*** d_l1 = nullptr;
    init_tracking(&d_l1);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMallocManaged(&d_in, size);
    cudaMallocManaged(&d_out, size);

    for (int i = 0; i < N; i++) {
        d_in[i] = 1.0f;
        d_out[i] = 0.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    kernel_with_device_calls<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    printf("[test_device_functions] Completed successfully, sample out[0]=%.1f\n", d_out[0]);

    export_binary(d_l1, "access_log.bin");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
