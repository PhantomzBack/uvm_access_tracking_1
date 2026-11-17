#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// A simple elementwise MLP-like layer forward pass
__global__ void layer_compute_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     const float* __restrict__ weight,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float w = weight[idx];
        // Perform non-trivial compute to simulate layer operation
        float acc = x * w;
        for (int k = 0; k < 16; ++k) {
            acc = acc * 0.99f + x * 0.01f;
        }
        out[idx] = acc;
    }
}

int main(int argc, char** argv) {
    const int NUM_LAYERS = 8;
    const int ELEMS_PER_LAYER = 2 * 1024 * 1024; // 2M floats = 8 MB per layer
    const size_t BYTES_PER_LAYER = ELEMS_PER_LAYER * sizeof(float);
    const int NUM_ITERATIONS = 6; // 6 token steps

    printf("=================================================================\n");
    printf("  Record & Replay Test: %d Layers, %.1f MB per layer, %d Iterations\n",
           NUM_LAYERS, (double)BYTES_PER_LAYER / (1024.0 * 1024.0), NUM_ITERATIONS);
    printf("=================================================================\n");

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_in, BYTES_PER_LAYER));
    CUDA_CHECK(cudaMallocManaged(&d_out, BYTES_PER_LAYER));

    float* weights[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; ++l) {
        CUDA_CHECK(cudaMallocManaged(&weights[l], BYTES_PER_LAYER));
        // Initialize host-side to leave memory currently resident on CPU
        for (int i = 0; i < ELEMS_PER_LAYER; ++i) {
            weights[l][i] = 1.0f + 0.01f * l;
        }
    }
    for (int i = 0; i < ELEMS_PER_LAYER; ++i) {
        d_in[i] = 2.0f;
        d_out[i] = 0.0f;
    }

    dim3 block(256);
    dim3 grid((ELEMS_PER_LAYER + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Simulate autoregressive decode loop across tokens
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (int l = 0; l < NUM_LAYERS; ++l) {
            const float* in_ptr = (l % 2 == 0) ? d_in : d_out;
            float* out_ptr = (l % 2 == 0) ? d_out : d_in;
            layer_compute_kernel<<<grid, block>>>(in_ptr, out_ptr, weights[l], ELEMS_PER_LAYER);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Total Execution Time: %.3f ms (avg %.3f ms per token iteration)\n",
           ms, ms / NUM_ITERATIONS);

    // Verify output
    printf("Verification: out[0]=%.4f (Expected ~2.0)\n", d_out[0]);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        CUDA_CHECK(cudaFree(weights[l]));
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
