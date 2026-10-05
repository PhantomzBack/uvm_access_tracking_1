#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include "tracking.h"

#define CUDA_CHECK(status)                                                   \
  {                                                                          \
    cudaError_t error = status;                                              \
    if (error != cudaSuccess) {                                              \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)      \
                << " at line: " << __LINE__ << std::endl;                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }

#define CUTLASS_CHECK(status)                                                \
  {                                                                          \
    cutlass::Status error = status;                                          \
    if (error != cutlass::Status::kSuccess) {                                \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)     \
                << " at line: " << __LINE__ << std::endl;                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }

// Define GEMM operator for single-precision GEMM on Ampere (sm_80 / sm_86)
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using ElementAccumulator = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80
>;

int main(int argc, char** argv) {
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int num_iters = 20;

    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) num_iters = atoi(argv[2]);

    printf("[CUTLASS Benchmark] Matrix size %dx%d (M=%d, N=%d, K=%d), iterations=%d\n",
           M, N, M, N, K, num_iters);

    size_t bytes_A = (size_t)M * K * sizeof(ElementInputA);
    size_t bytes_B = (size_t)K * N * sizeof(ElementInputB);
    size_t bytes_C = (size_t)M * N * sizeof(ElementOutput);

    ElementInputA *A = nullptr;
    ElementInputB *B = nullptr;
    ElementOutput *C = nullptr;

    CUDA_CHECK(cudaMallocManaged(&A, bytes_A));
    CUDA_CHECK(cudaMallocManaged(&B, bytes_B));
    CUDA_CHECK(cudaMallocManaged(&C, bytes_C));

    // Initialize host data
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = 2.0f;
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = 0.0f;

#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif

    CutlassGemm gemm_op;

    int lda = K;
    int ldb = N;
    int ldc = N;

    float alpha = 1.0f;
    float beta  = 0.0f;

    typename CutlassGemm::Arguments args(
        {M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {C, ldc},
        {alpha, beta}
    );

    // Warmup
    CUTLASS_CHECK(gemm_op(args));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < num_iters; it++) {
        CUTLASS_CHECK(gemm_op(args));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iters;
    double gflops = (2.0 * (double)M * (double)N * (double)K * 1e-9) / (avg_ms * 1e-3);

    printf("BENCHMARK_TIME: %0.6f\n", avg_ms);
    printf("GPU Runtime: %0.6fs\n", avg_ms * 1e-3);
    printf("Throughput: %0.2f GFLOPS (avg %0.4f ms per GEMM)\n", gflops, avg_ms);

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
#endif

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
}
