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

// Helper to get global thread ID and grid stride
// Index: $i = blockIdx.x \cdot blockDim.x + threadIdx.x$
// Stride: $s = blockDim.x \cdot gridDim.x$



// 2. Stride: Forces multiple cache line fetches
__global__ void test_stride(int* data, int n, int mem_stride) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        // We use 'i' as the logic index and 'mem_stride' for the memory gap
        data[i * mem_stride] = i;
    }
}



int main(int argc, char** argv) {
    int n = 1048576; // 1M elements

    int *d_in, *d_out, *d_indices;
    // Over-allocate for stride (mem_stride * n)
    CUDA_CHECK(cudaMallocManaged(&d_in, n * 64 * sizeof(int))); 
    CUDA_CHECK(cudaMallocManaged(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_indices, n * sizeof(int)));

    std::vector<int> h_indices(n);
    for(int i=0; i<n; i++) h_indices[i] = i;
    std::shuffle(h_indices.begin(), h_indices.end(), std::mt19937(42));
    cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif

    // Execution Configuration: Using a fixed grid size
    // 256 blocks is usually enough to saturate most modern GPUs
    dim3 block(256);
    dim3 grid(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout << "Warming up GPU..." << std::endl;
    // Warmup & Select Kernel
    test_stride<<<grid, block>>>(d_in, n, 32); // Pre-warm with a different kernel to stabilize clock speeds
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    test_stride<<<grid, block>>>(d_in, n, 32); // Timed run
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "BENCHMARK_TIME: " << ms << " ms" << std::endl;

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_l1);
#endif

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_indices);
    return 0;
}