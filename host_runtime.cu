#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#define L1_ENTRIES 8192


/*
extern "C" void init_tracking(void*** d_l1_ptr) {
    cudaMalloc(d_l1_ptr, L1_ENTRIES * sizeof(void*));
    cudaMemset(*d_l1_ptr, 0, L1_ENTRIES * sizeof(void*));
    // Link to the GPU's global pointer
    void** temp_ptr = *d_l1_ptr;
    cudaMemcpyToSymbol("shadow_cr3", &temp_ptr, sizeof(void*));
}
*/

// Declare the kernel signature so the compiler is happy

/*
extern "C" void check_invocation() {

    int h_invoked = 0;
    // Get the symbol from the GPU
    void* dummy = (void*)&mark_access_invoked; 
    
    cudaError_t err = cudaMemcpyFromSymbol(&h_invoked, "mark_access_invoked", sizeof(int));
    printf("Mark Access Invoked! Starting\n");

    // In host_runtime.cpp
void* devPtr = nullptr;
    err = cudaGetSymbolAddress(&devPtr, "mark_access_invoked");

if (err == cudaSuccess) {
    // If we found the address, copy from that address instead of the symbol name
    cudaMemcpy(&h_invoked, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
} else {
    printf("Driver still cannot find symbol: %s\n", cudaGetErrorString(err));
}
 //   cudaMemcpyFromSymbol(&h_invoked, "mark_access_invoked", sizeof(int));
    if (h_invoked == 1) {
        printf("SUCCESS: MarkAccess was definitely called on the GPU!\n");
    } else {
        printf("FAILURE: MarkAccess was NEVER called.\n");
    }
}
*/

extern __device__ int mark_access_invoked;

extern "C" void check_invocation() {
    int host_flag = 0;

    cudaMemcpyFromSymbol(&host_flag, "mark_access_invoked",
                         sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (host_flag)
        printf("MarkAccess WAS invoked\n");
    else
        printf("MarkAccess was NOT invoked\n");
}
extern "C" void init_tracking(void*** d_l1_ptr) {
    // 1. Increase the internal GPU heap size (e.g., to 128MB or more)
    // This is required for dynamic 'malloc' in kernels to work.
    size_t heap_size = 128 * 1024 * 1024; 
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

    // 2. Allocate L1 Directory
    cudaMalloc(d_l1_ptr, 8192 * sizeof(void*));
    cudaMemset(*d_l1_ptr, 0, 8192 * sizeof(void*));
    
    void** temp_ptr = *d_l1_ptr;
    cudaMemcpyToSymbol("shadow_cr3", &temp_ptr, sizeof(void*));
}

extern "C" void export_log(void** d_l1, const char* filename) {
    std::cout << "Exiting and writing to " << filename << std::endl;
    std::ofstream f(filename);
    std::vector<void*> h_l1(L1_ENTRIES);
    cudaMemcpy(h_l1.data(), d_l1, L1_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (h_l1[i]) {
            std::cout << "Entry exists for " << i << ": " << std::endl;
            std::vector<uint64_t> bitmap(4096 / 8);
            cudaMemcpy(bitmap.data(), h_l1[i], 4096, cudaMemcpyDeviceToHost);
            for (int j = 0; j < bitmap.size(); j++) {
                if (bitmap[j]) {
                    for (int b = 0; b < 64; b++) {
                        if (bitmap[j] & (1ULL << b)) {
                            uint64_t vaddr = ((uint64_t)i << 27) | ((uint64_t)j * 64 + b) << 12;
                            f << std::hex << "0x" << vaddr << "\n";
                        }
                    }
                }
            }
        }
        else{
//            std::cout << "No entry for " << i << std::endl;
        }
    }
}
