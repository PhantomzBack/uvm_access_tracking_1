#include "common.h"



// ── Device globals ────────────────────────────────────────────────────────────
extern "C" {
    // shadow_l1 points to the L1 table (array of 512 void** pointers)
    __device__ void*** shadow_l1 = nullptr;


    // ── MarkAccess ────────────────────────────────────────────────────────────
    __device__ void MarkAccess(uintptr_t addr)
    {

        if (!shadow_l1) {
            printf("[MarkAccess] shadow_l1 not initialised\n");
            return;
        }

        // Decompose address
        uint32_t l1_idx    = (addr >> L1_SHIFT) & L1_MASK;
        uint32_t l2_idx    = (addr >> L2_SHIFT) & L2_MASK;
        uint32_t l3_offset = (addr >> L3_SHIFT) & L3_MASK; // bit index within L3 leaf

        if (addr % (1 << 20) == 0) // Throttle printf spam
            printf("[MarkAccess] addr=%p  l1=%u l2=%u l3=%u\n",
                   (void*)addr, l1_idx, l2_idx, l3_offset);

        LOG("[MarkAccess] addr=%p  l1=%u l2=%u l3=%u\n",
            (void*)addr, l1_idx, l2_idx, l3_offset);
        LOG("[MarkAccess] shadow_l1[l1_idx]=%p\n", shadow_l1[l1_idx]);

        // ── Level 1 → Level 2 (demand allocate L2 table) ─────────────────────
        if (shadow_l1[l1_idx] == nullptr) {
            void** new_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*));
            LOG("[MarkAccess] malloc for L2 table returned %p\n", new_l2);
            if (!new_l2) return;

            unsigned long long old = atomicCAS(
                (unsigned long long*)&shadow_l1[l1_idx],
                0ULL,
                (unsigned long long)new_l2
            );
            if (old != 0ULL) {
                free(new_l2);
            } else {
                LOG("[MarkAccess] allocated L2 table for L1 index %u at %p\n", l1_idx, new_l2);
                memset(new_l2, 0, L2_ENTRIES * sizeof(void*));
                int flag = 0;
                for (int i = 0; i != L2_ENTRIES; i++) {
                    if (((unsigned long long*)new_l2)[i] != 0ULL) {
                        flag = 1;
                        LOG("[MarkAccess] error: new L2 table at %p not zero-initialised! Entry %d is %p\n",
                            new_l2, i, ((void**)new_l2)[i]);
                        break;
                    }
                }
                if (!flag)
                    LOG("[MarkAccess] new L2 table at %p successfully zero-initialised\n", new_l2);
            }
        }

        void** l2_table = (void**)shadow_l1[l1_idx];

        // ── Level 2 → Level 3 (demand allocate L3 bitmap leaf) ───────────────
        if (l2_table[l2_idx] == nullptr) {
            void* new_l3 = malloc(L3_BYTES);
            if (!new_l3) return;

            unsigned long long old = atomicCAS(
                (unsigned long long*)&l2_table[l2_idx],
                0ULL,
                (unsigned long long)new_l3
            );
            if (old != 0ULL)
                free(new_l3);
            else{
                LOG("[MarkAccess] allocated L3 leaf for L1 index %u, L2 index %u\n",
                    l1_idx, l2_idx);
                memset(new_l3, 0, L3_BYTES);
            }

        }

        // ── Level 3: flip the bit for this page ──────────────────────────────
        unsigned long long* l3_bitmap = (unsigned long long*)l2_table[l2_idx];
        if (l3_bitmap) {
            LOG("[MarkAccess] marking page accessed in L3 bitmap for L1 index %u, L2 index %u, bit %u, was %lld\n",
                l1_idx, l2_idx, l3_offset, (l3_bitmap[l3_offset / 64] >> (l3_offset % 64)) & 1ULL);
            unsigned long long* word = &l3_bitmap[l3_offset / 64];
            unsigned long long  mask = 1ULL << (l3_offset % 64);
            atomicOr(word, mask);
            LOG("[MarkAccess] after atomicOr, bit is now %lld\n",
                ((*word) >> (l3_offset % 64)) & 1ULL);
        } else {
            printf("[MarkAccess] error: L3 bitmap not allocated for L1 index %u, L2 index %u\n",
                   l1_idx, l2_idx);
        }
    }
}
