#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// Forward declarations of our runtime helpers
extern "C" void init_tracking(void*** d_l1_ptr);
extern "C" void export_log(void** d_l1, const char* filename);

__global__ void stride_access(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // This access will be instrumented by your pass
        data[tid * 1024] = tid; 
    }
    // std::cout << "blockIdx.x: " << blockIdx.x << "| threadIdx.x: " << threadIdx.x << std::endl;
    printf("blockIdx.x: %d, threadIdx.x: %d done.", blockIdx.x, threadIdx.x);
}
/*
int main() {
    int n = 10;
    size_t size = n * 1024 * sizeof(int);
    int *d_data;
    cudaMallocManaged(&d_data, size);

    void** d_l1;
    init_tracking(&d_l1);

    // Manually allocate one L2 leaf for the first 128MB range for testing
    void* d_l2;
    cudaMalloc(&d_l2, 4096);
    cudaMemset(d_l2, 0, 4096);
    cudaMemcpy(&(d_l1[0]), &d_l2, sizeof(void*), cudaMemcpyHostToDevice);

    stride_access<<<1, n>>>(d_data, n);
    cudaDeviceSynchronize();

    export_log(d_l1, "access_log.txt");
    std::cout << "Done! Check access_log.txt for results." << std::endl;

    return 0;
}
*/

int main() {
    int n = 10;
    int *d_data;
    // Allocate something far away to test dynamic allocation
    cudaMallocManaged(&d_data, 1024 * 1024 * 100 * sizeof(int)); 

    void** d_l1;
    //init_tracking(&d_l1); // Heap limit is set here
    printf("Starting\n");

    stride_access<<<1, n>>>(d_data, n);
    cudaDeviceSynchronize();
    printf("Ended\n");

   // export_log(d_l1, "access_log.txt");
    return 0;
}
