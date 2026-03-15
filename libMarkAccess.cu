#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {
    // Manually declare the NVVM intrinsics to bypass broken Clang-20 headers
    __device__ unsigned long long __nvvm_atom_or_gen_ll(unsigned long long* ptr, unsigned long long val);
    __device__ unsigned long long __nvvm_atom_cas_gen_ll(unsigned long long* ptr, unsigned long long comp, unsigned long long val);

    __device__ void** shadow_cr3 = nullptr;
    __device__ uint64_t last_page_cache = 0xFFFFFFFFFFFFFFFFULL;
    
    __device__ int mark_access_invoked = 0; // The original canary



    __device__ void MarkAccess(uintptr_t addr) {
        mark_access_invoked = 1;
        printf("In MarkAccess\n");
        if (!shadow_cr3) return;
        if (addr % 1024 == 0) { // Throttle it so it doesn't spam
           printf("Tracking access to: %p\n", (void*)addr);
        }
        uint64_t vpn = addr >> 12;
        uint32_t l1_idx = (vpn >> 15); 
        uint32_t l2_bit_offset = vpn & 0x7FFF;

        // --- DYNAMIC L2 INSERTION (DEMAND PAGING) ---
        if (shadow_cr3[l1_idx] == nullptr) {
            void* new_leaf = malloc(4096); 
            if (!new_leaf) return; 

            // Initialize the new leaf to zero (C-style loop to avoid memset headers)
            unsigned long long* bitmap_init = (unsigned long long*)new_leaf;
            for(int i = 0; i < 512; i++) {
                bitmap_init[i] = 0;
            }

            // Atomic Compare-and-Swap directly using the intrinsic
            unsigned long long* target_l1 = (unsigned long long*)&shadow_cr3[l1_idx];
            void* already_there = (void*)__nvvm_atom_cas_gen_ll(target_l1, 0ULL, (unsigned long long)new_leaf);
            
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
            __nvvm_atom_or_gen_ll(target_bit, bit_mask);
        }
    }
}
