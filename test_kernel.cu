#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// #define NVCC_TESTING


// Forward declarations of our runtime helpers
#ifndef NVCC_TESTING
extern "C" __device__ int mark_access_invoked;
extern "C" void init_tracking(void*** d_l1_ptr);
extern "C" void export_log(void** d_l1, const char* filename);
extern "C" void check_invocation();
#endif

#ifndef INVOCATION_CANARY

/*extern "C" void get_canary_value(int* out_ptr);

extern "C" void check_invocation() {
    int h_val = 0;
    int *d_val_ptr;

    // 1. Allocate a tiny "mailbox" on the GPU
    cudaMalloc(&d_val_ptr, sizeof(int));
    cudaMemset(d_val_ptr, 0, sizeof(int));

    // 2. Launch the getter kernel (1 block, 1 thread)
    // Using the triple chevron syntax or the Clang equivalent



//    get_canary_value<<<1, 1>>>(d_val_ptr);
    void* args[] = { &d_val_ptr };

    cudaLaunchKernel(
            (void*)get_canary_value,
            dim3(1),
            dim3(1),
            args,
            0,
            0
            );
    cudaDeviceSynchronize();

    // 3. Copy the value back to host
    cudaMemcpy(&h_val, d_val_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_val == 1) {
        printf("[DEBUG] SUCCESS: Canary is 1. Instrumentation is LIVE!\n");
    } else {
        printf("[DEBUG] FAILURE: Canary is 0. Kernel ran but MarkAccess was not called.\n");
    }

    cudaFree(d_val_ptr);
}*/
#endif
#ifdef NVCC_TESTING
void init_tracking(void*** d_l1_ptr){
    return;
}

void export_log(void** d_l1, const char* filename){
    return;
}
#endif

__global__ void myKernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}


__global__ void stride_access(int* data, int n) {
    printf("Stride: Hello from GPU thread %d\n", threadIdx.x);
//    return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // This access will be instrumented by your pass
        data[tid * 1024] = tid; 
    }
    printf("TID: %d concluded", tid);
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
#ifndef NVCC_TESTING
    void** d_l1;
    init_tracking(&d_l1); // Heap limit is set here
#endif
    myKernel<<<1, 5>>>();
    stride_access<<<1, 5>>>(d_data, n);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("Kernel launch failed with error: %s\\n", cudaGetErrorString(cudaerr));
    }
    printf("Kernel launch successful"); 
    check_invocation();
#ifndef NVCC_TESTING
    export_log(d_l1, "access_log.txt");
#endif
    return 0;
}
