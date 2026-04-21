/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * (Trimmed: removed CUDA Samples helper dependencies)
 *
 * Matrix multiplication: C = A * B using shared-memory tiling.
 * Reference: V. Volkov and J. Demmel, SC '08.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "tracking.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Kernel ────────────────────────────────────────────────────────────────────
// C = A * B  (wA = width of A, wB = width of B)
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// ── Host helper ───────────────────────────────────────────────────────────────
static void fill(float *p, int n, float v) {
    for (int i = 0; i < n; i++) p[i] = v;
}

static int parseInt(int argc, char **argv, const char *flag, int def) {
    for (int i = 1; i < argc; i++) {
        int v;
        char pat[64];
        snprintf(pat, sizeof(pat), "%s=%%d", flag);
        if (sscanf(argv[i], pat, &v) == 1) return v;
    }
    return def;
}

// ── Run + time ────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    const int block_size = 32;
    const int nIter      = 10;

    int wA = parseInt(argc, argv, "-wA", 1024);
    int hA = parseInt(argc, argv, "-hA", 1024);
    int wB = parseInt(argc, argv, "-wB", 1024);
    int hB = parseInt(argc, argv, "-hB", 1024);  // must equal wA

    if (wA != hB) {
        fprintf(stderr, "Error: wA (%d) must equal hB (%d)\n", wA, hB);
        return EXIT_FAILURE;
    }
    if (wA % block_size || hA % block_size || wB % block_size) {
        fprintf(stderr, "Error: matrix dims must be multiples of %d\n", block_size);
        return EXIT_FAILURE;
    }

    printf("[MatrixMul] A(%d×%d) × B(%d×%d)  block=%d  iterations=%d\n",
           wA, hA, wB, hB, block_size, nIter);

    unsigned int szA = wA * hA, szB = wB * hB, szC = wB * hA;

    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, szA * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_B, szB * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C, szC * sizeof(float)));
    fill(h_A, szA, 1.0f);
    fill(h_B, szB, 0.01f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, szC * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, szA * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, szB * sizeof(float), cudaMemcpyHostToDevice, stream));

    dim3 threads(block_size, block_size);
    dim3 grid(wB / block_size, hA / block_size);

#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif

    // Warmup
    MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, wA, wB);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    for (int j = 0; j < nIter; j++)
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, wA, wB);

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float msTotal = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, ev_start, ev_stop));
    float msPerKernel = msTotal / nIter;

    double flops  = 2.0 * wA * hA * wB;
    double gflops = (flops * 1e-9) / (msPerKernel * 1e-3);
    printf("Performance= %.2f GFlop/s  Time= %.3f ms/kernel\n", gflops, msPerKernel);
    printf("BENCHMARK_TIME: %.3f ms\n", msPerKernel);

    // Correctness spot-check
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, szC * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ref = wA * 0.01f;
    int   bad = 0;
    for (int i = 0; i < (int)szC && bad < 5; i++) {
        float rel = fabsf(h_C[i] - ref) / ref;
        if (rel > 1e-4f) { printf("  mismatch[%d]: %.6f vs %.6f\n", i, h_C[i], ref); bad++; }
    }
    printf("Correctness: %s\n", bad == 0 ? "PASS" : "FAIL");

#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_l1);
#endif

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
