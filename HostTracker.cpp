#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define L1_ENTRIES 8192 // Supports up to 1TB of address space
#define L2_SIZE_BYTES 4096

extern "C" {
    // Function to initialize the GPU table
    void** init_tracking(void*** d_l1_ptr) {
        void** h_l1 = (void**)calloc(L1_ENTRIES, sizeof(void*));
        cudaMalloc(d_l1_ptr, L1_ENTRIES * sizeof(void*));
        cudaMemset(*d_l1_ptr, 0, L1_ENTRIES * sizeof(void*));
        
        // Link the GPU symbol to our allocated table
        // Note: 'shadow_cr3' must be defined in the GPU library
        // cudaMemcpyToSymbol(shadow_cr3, d_l1_ptr, sizeof(void*));
        
        return h_l1;
    }

    // Function to export tracked pages to a CSV
    void export_to_file(void** h_l1, void** d_l1, const char* filename) {
        FILE* f = fopen(filename, "w");
        fprintf(f, "Virtual_Address\n");

        // Sync and pull L1 back to host
        cudaDeviceSynchronize();
        cudaMemcpy(h_l1, d_l1, L1_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);

        uint8_t leaf_temp[L2_SIZE_BYTES];
        for (int i = 0; i < L1_ENTRIES; i++) {
            if (h_l1[i] != nullptr) {
                cudaMemcpy(leaf_temp, h_l1[i], L2_SIZE_BYTES, cudaMemcpyDeviceToHost);
                unsigned long long* bitmap = (unsigned long long*)leaf_temp;
                
                for (int j = 0; j < (L2_SIZE_BYTES / 8); j++) {
                    if (bitmap[j] != 0) {
                        for (int bit = 0; bit < 64; bit++) {
                            if (bitmap[j] & (1ULL << bit)) {
                                uint64_t vaddr = ((uint64_t)i << 27) | ((uint64_t)j * 64 + bit) << 12;
                                fprintf(f, "0x%lx\n", vaddr);
                            }
                        }
                    }
                }
            }
        }
        fclose(f);
    }
}
