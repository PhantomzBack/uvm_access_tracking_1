#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include "common.h"
#include "tracking.h"

// ── Test kernels ──────────────────────────────────────────────────────────────
__global__ void myKernel()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[myKernel] Hello from GPU\n");
}

__global__ void stride_access(int* data, int n)
{
    check_shadow_l1();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid * 1024] = tid;
        LOG("[stride_access] tid=%d accessed data[%d]\n", tid, tid * 1024);
        MarkAccess((uintptr_t)(&data[tid * 1024]));
    }
    printf("[stride_access] tid=%d done\n", tid);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    int n = 5;
    int* d_data;
    CUDA_CHECK(cudaMallocManaged(&d_data, 1024ULL * 1024 * 100 * sizeof(int)));

    void*** d_l1;
    init_tracking(&d_l1);

    stride_access<<<1, n>>>(d_data, n);
    //check_shadow_l1_kernel<<<1,1>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("[main] sync error: %s\n", cudaGetErrorString(err));
    else
        printf("[main] kernels completed successfully\n");

    export_log(d_l1, "access_log.txt");
    export_binary(d_l1, "access_log.bin");

    cudaFree(d_data);
    cudaFree(d_l1);
    return 0;
}
