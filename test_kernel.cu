#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

#define NVCC_TESTING
#define MA_TESTING
#define DBG printf("At line: %d\n",__LINE__);
#define DBG_RET printf("At line: %d\n",__LINE__); return;

#include <stdint.h>
extern "C" {
    
    // Manually declare the NVVM intrinsics to bypass broken Clang-20 headers
//    __device__ unsigned long long __nvvm_atom_or_gen_ll(unsigned long long* ptr, unsigned long long val);
//    __device__ unsigned long long __nvvm_atom_cas_gen_ll(unsigned long long* ptr, unsigned long long comp, unsigned long long val);

    __device__ void** shadow_cr3 = nullptr;
    __device__ uint64_t last_page_cache = 0xFFFFFFFFFFFFFFFFULL;
    
    __device__ int mark_access_invoked = 0; // The original canary



    __device__ void MarkAccess(uintptr_t addr) {
        mark_access_invoked = 1;
        printf("In MarkAccess\n");
        if (!shadow_cr3) {
            printf("No shadow cr3\n");
            return;
        }
        else{
            printf("Shadow CR3 found\n");
//            return;
        }

        if (addr % 1024 == 0) { // Throttle it so it doesn't spam
           printf("Tracking access ttoo: %p\n", (void*)addr);
        }
        DBG;
        uint64_t vpn = addr >> 12;
        DBG;
        uint32_t l1_idx = (vpn >> ); 
        uint32_t l2_bit_offset = vpn & 0x7FFF;
        DBG;
        if (l1_idx >= 8192) {
            printf("[WARN] l1_idx %u out of range for addr %p\n", l1_idx, (void*)addr);
            return;
        }
        // --- DYNAMIC L2 INSERTION (DEMAND PAGING) ---
        printf("Shadow cr3 l1_idx: %p %d\n", shadow_cr3, l1_idx);
        
        printf("Shadow cr3[l1_idx]: %p\n", &shadow_cr3[l1_idx]);

        return;
        if (shadow_cr3[l1_idx] == nullptr) {
            void* new_leaf = malloc(4096); 
            if (!new_leaf) return; 
            printf("malloced new leaf %p\n", new_leaf);
            // Initialize the new leaf to zero (C-style loop to avoid memset headers)
            unsigned long long* bitmap_init = (unsigned long long*)new_leaf;
            for(int i = 0; i < 512; i++) {
                bitmap_init[i] = 0;
            }

            // Atomic Compare-and-Swap directly using the intrinsic
            unsigned long long* target_l1 = (unsigned long long*)&shadow_cr3[l1_idx];
            void* already_there = (void*)atomicCAS(target_l1, 0ULL, (unsigned long long)new_leaf);
            
            if (already_there != nullptr) {
                free(new_leaf);
            }
        }

        // --- BIT FLIPPING ---
        unsigned long long* l2_bitmap = (unsigned long long*)shadow_cr3[l1_idx];
        if (l2_bitmap != nullptr) {
            unsigned long long* target_bit = &l2_bitmap[l2_bit_offset / 64];
            unsigned long long bit_mask = (1ULL << (l2_bit_offset % 64));
            
            // Call the OR intrinsic directly
            atomicOr(target_bit, bit_mask);
        }
    }
}

#ifdef MA_TESTING


#endif


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
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
#define L1_ENTRIES 8192
void init_tracking(void*** d_l1_ptr){
        // 1. Increase the internal GPU heap size (e.g., to 128MB or more)
    // This is required for dynamic 'malloc' in kernels to work.
    size_t heap_size = 128 * 1024 * 1024; 
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

    // 2. Allocate L1 Directory
    cudaMalloc(d_l1_ptr, 8192 * sizeof(void*));
    cudaMemset(*d_l1_ptr, 0, 8192 * sizeof(void*));
    
    void** temp_ptr = *d_l1_ptr;
    void* readback;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_cr3, sizeof(void*)));
    printf("Readback: %p\n", readback);

    printf("Shadow CR3 Initialised at: %p %p %p\n", d_l1_ptr, temp_ptr, &temp_ptr);
    // Trying with allocating
    CUDA_CHECK(cudaMemcpyToSymbol(shadow_cr3, &temp_ptr, sizeof(void*)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_cr3, sizeof(void*)));

    printf("Readback: %p\n", readback);
    return;
}

void export_log(void** d_l1, const char* filename){
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

    return;
}
#endif

__global__ void myKernel() {
    int kbc = 5;
    printf("Hello from GPU thread %d %d\n", threadIdx.x, kbc);
}


__global__ void stride_access(int* data, int n) {
    printf("Stride: Hello from GPU thread %d %d\n", threadIdx.x, 6);
//    asm("trap");
//    return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // This access will be instrumented by your pass
        data[tid * 1024] = tid; 
        MarkAccess((uintptr_t)(&(data[tid * 1024])));
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
//#ifndef NVCC_TESTING
    void** d_l1;
    init_tracking(&d_l1); // Heap limit is set here
//#endif
    myKernel<<<1, 5>>>();
    stride_access<<<1, 5>>>(d_data, n);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("Kernel launch failed with error: %s\\n", cudaGetErrorString(cudaerr));
    }
    printf("Kernel launch successful"); 
//#ifndef NVCC_TESTING
    export_log(d_l1, "access_log.txt");
//#endif
    return 0;
}
