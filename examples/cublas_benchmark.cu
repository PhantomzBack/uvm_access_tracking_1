#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tracking.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUBLAS_CHECK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"cuBLASassert: error %d in %s:%d\n", (int)code, file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv) {
    int N = 2048;
    int num_iters = 20;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_iters = atoi(argv[2]);

    int M = N;
    int K = N;

    printf("[cuBLAS Benchmark] Matrix size %dx%d (M=%d, N=%d, K=%d), iterations=%d\n",
           N, N, M, N, K, num_iters);

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *A, *B, *C;
    CUDA_CHECK(cudaMallocManaged(&A, bytes_A));
    CUDA_CHECK(cudaMallocManaged(&B, bytes_B));
    CUDA_CHECK(cudaMallocManaged(&C, bytes_C));

    // Initialize arrays
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = 2.0f;
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = 0.0f;

#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Warmup
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             A, M,
                             B, K,
                             &beta,
                             C, M));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < num_iters; it++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K,
                                 &alpha,
                                 A, M,
                                 B, K,
                                 &beta,
                                 C, M));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iters;

    double gflops = (2.0 * (double)M * (double)N * (double)K * 1e-9) / (avg_ms * 1e-3);

    printf("BENCHMARK_TIME: %0.6f\n", avg_ms);
    printf("GPU Runtime: %0.6fs\n", avg_ms * 1e-3);
    printf("Throughput: %0.2f GFLOPS (avg %0.4f ms per SGEMM)\n", gflops, avg_ms);

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
#endif

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
#ifdef TRACKING_ENABLED
    CUDA_CHECK(cudaFree(d_l1));
#endif

    return 0;
}
