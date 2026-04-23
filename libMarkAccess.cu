#include "common.h"
#include "tracking.h"
// ── Device globals ────────────────────────────────────────────────────────────
extern "C" {
    // shadow_l1 points to the L1 table (array of 512 void** pointers)
    __device__ void*** shadow_l1 = nullptr;


    // ── MarkAccess ────────────────────────────────────────────────────────────
    __device__ void MarkAccess(uintptr_t addr)
    {
        LOG("[MarkAccess] called with addr=%p\n", (void*)addr);
        if (!shadow_l1) {
            LOG("[MarkAccess] shadow_l1 not initialised %p\n", shadow_l1);
            return;
        }
        

        // Decompose address
        uint32_t l1_idx    = (addr >> L1_SHIFT) & L1_MASK;
        uint32_t l2_idx    = (addr >> L2_SHIFT) & L2_MASK;
        uint32_t l3_offset = (addr >> L3_SHIFT) & L3_MASK; // bit index within L3 leaf

        if (addr % (1 << 20) == 0) // Throttle printf spam
            LOG("[MarkAccess] addr=%p  l1=%u l2=%u l3=%u\n",
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

            // Warp-aggregated atomicOr: group threads in the warp by target 64-bit
            // word (identified by L1,L2,word_idx) and OR their bitmasks together
            // using shfl_xor reductions conditioned on matching keys. Only the
            // group's leader performs a single 64-bit atomicOr.
            unsigned full = __activemask();
            uint32_t word_idx = l3_offset / 64;
            unsigned long long my_mask = 1ULL << (l3_offset % 64);
            uint32_t key = (l1_idx << 21) | (l2_idx << 9) | word_idx; // fits in 30 bits

            unsigned long long agg = my_mask;
            // Efficient tree reduction using shfl_xor
            for (int off = 16; off > 0; off >>= 1) {
                unsigned long long other = __shfl_xor_sync(full, agg, off);
                uint32_t other_key = __shfl_xor_sync(full, key, off);
                if (other_key == key) agg |= other;
            }

            int lane = threadIdx.x & 31;
            uint32_t group_mask = __match_any_sync(full, key);
            int leader = __ffs(group_mask) - 1;
            if (lane == leader) {
                unsigned long long* word = &l3_bitmap[word_idx];
                unsigned long long prev = *word;
                unsigned long long delta = agg & ~prev;
                if (delta) atomicOr(word, delta);
                unsigned long long after = *word;
                LOG("[MarkAccess] after atomicOr, bit is now %lld (delta=0x%llx)\\n",
                    ((after >> (l3_offset % 64)) & 1ULL), delta);
            }
        } else {
            LOG("[MarkAccess] error: L3 bitmap not allocated for L1 index %u, L2 index %u\n",
                   l1_idx, l2_idx);
        }
    }

    // ── BatchMarkAccess ───────────────────────────────────────────────────────
    // Two key optimisations over the naive per-element loop:
    //
    //  1. O(count) → O(pages) for dense strides (stride < PAGE_SIZE):
    //     When stride < 4KB, consecutive elements share pages (no page gaps exist
    //     between start and end). Compute [start_page, end_page] directly and loop
    //     over pages — 1024x fewer iterations for float coalesced access (4B stride).
    //
    //  2. Warp-level deduplication:
    //     For coalesced access, all 32 threads in a warp access the same pages.
    //     The warp leader (lowest active lane) broadcasts its page range; any thread
    //     whose range is fully covered by the leader's can exit immediately, letting
    //     the leader handle the marking for the whole warp.
    __device__ void BatchMarkAccess(uintptr_t base_addr, int64_t stride, uint64_t count)
    {
        if (!shadow_l1 || count == 0) return;

        // Helper: traverse L1→L2→L3 and mark one page.
        // Declared as a lambda-style macro to avoid a device function call.
        // 'cached_l1', 'cached_l2', 'l2_table', 'l3_bitmap' are in the caller's scope.
#define MARK_PAGE(addr)                                                          \
        do {                                                                     \
            uint32_t _l1 = (uint32_t)(((addr) >> L1_SHIFT) & L1_MASK);          \
            uint32_t _l2 = (uint32_t)(((addr) >> L2_SHIFT) & L2_MASK);          \
            uint32_t _l3 = (uint32_t)(((addr) >> L3_SHIFT) & L3_MASK);          \
            if (_l1 != cached_l1) {                                              \
                if (!shadow_l1[_l1]) {                                           \
                    void** _nl2 = (void**)malloc(L2_ENTRIES * sizeof(void*));    \
                    if (!_nl2) break;                                             \
                    unsigned long long _old = atomicCAS(                         \
                        (unsigned long long*)&shadow_l1[_l1],                   \
                        0ULL, (unsigned long long)_nl2);                         \
                    if (_old != 0ULL) free(_nl2);                                \
                    else              memset(_nl2, 0, L2_ENTRIES * sizeof(void*)); \
                }                                                                \
                l2_table  = (void**)shadow_l1[_l1];                             \
                cached_l1 = _l1;  cached_l2 = (uint32_t)-1;  l3_bitmap = nullptr; \
            }                                                                    \
            if (_l2 != cached_l2) {                                              \
                if (!l2_table[_l2]) {                                            \
                    void* _nl3 = malloc(L3_BYTES);                               \
                    if (!_nl3) break;                                             \
                    unsigned long long _old = atomicCAS(                         \
                        (unsigned long long*)&l2_table[_l2],                    \
                        0ULL, (unsigned long long)_nl3);                         \
                    if (_old != 0ULL) free(_nl3);                                \
                    else              memset(_nl3, 0, L3_BYTES);                 \
                }                                                                \
                l3_bitmap = (unsigned long long*)l2_table[_l2];                 \
                cached_l2 = _l2;                                                 \
            }                                                                    \
            if (l3_bitmap) {                                                     \
                unsigned long long* _w = &l3_bitmap[_l3 / 64];                  \
                unsigned long long  _m = 1ULL << (_l3 % 64);                    \
                    if (!(*_w & _m)) atomicOr(_w, _m);                              \
            }                                                                    \
        } while (0)

        uint32_t            cached_l1 = (uint32_t)-1;
        uint32_t            cached_l2 = (uint32_t)-1;
        void**              l2_table  = nullptr;
        unsigned long long* l3_bitmap = nullptr;

        // ── Opt 1: dense stride → page-range loop ────────────────────────────
        // When 0 < stride < PAGE_SIZE there are no page gaps between first and
        // last element (adjacent elements are <4KB apart), so every page in
        // [start_page, end_page] is touched. Skip the per-element multiply.
        if (stride > 0 && (uint64_t)stride < (1ULL << L3_SHIFT)) {
            uintptr_t start_page = base_addr >> 12;
            uintptr_t end_page   = (base_addr + (uint64_t)(count - 1) * (uint64_t)stride) >> 12;

            // ── Opt 2: warp-level deduplication ──────────────────────────────
            // For coalesced access all warp lanes span the same page range.
            // The lowest active lane (leader) broadcasts its range; any thread
            // whose range is fully covered skips — the leader handles them.
            unsigned active      = __activemask();
            int      my_lane     = threadIdx.x & 31;
            int      leader_lane = __ffs(active) - 1;
            unsigned long long l0_start = __shfl_sync(active, (unsigned long long)start_page, leader_lane);
            unsigned long long l0_end   = __shfl_sync(active, (unsigned long long)end_page,   leader_lane);
            if (my_lane != leader_lane
                    && l0_start <= (unsigned long long)start_page
                    && l0_end   >= (unsigned long long)end_page)
                return;

            for (uintptr_t p = start_page; p <= end_page; p++)
                MARK_PAGE(p << 12);

            return;
        }

        // ── Sparse strides (stride ≥ PAGE_SIZE or ≤ 0): element-by-element ──
        // Elements may skip pages (stride ≥ 4KB) or go backwards; iterate per
        // element and use last_page to skip same-page repeats.
        uintptr_t last_page = (uintptr_t)-1;
        for (uint64_t i = 0; i < count; i++) {
            uintptr_t addr     = (uintptr_t)((intptr_t)base_addr + (int64_t)i * stride);
            uintptr_t cur_page = addr >> 12;
            if (cur_page == last_page) continue;
            last_page = cur_page;
            MARK_PAGE(addr);
        }

#undef MARK_PAGE
    }

    __global__ void copy_l2_to_host(void*** l1, int l1_idx, void** out, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
        if (tid < n)
            out[tid] = ((void**)l1[l1_idx])[tid];
        if (tid == n-1)
            LOG("[copy_l2_to_host] successful\n");
    }

    __global__ void copy_l3_to_host(void*** l1, int l1_idx, int l2_idx,
                                    unsigned long long* out, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
        if (tid < n)
            out[tid] = ((unsigned long long*)((void**)l1[l1_idx])[l2_idx])[tid];
        if (tid == n-1)
            LOG("[copy_l2_to_host] successful\n");

    }
}


// ── Host: initialise the L1 table ─────────────────────────────────────────────
void init_tracking(void**** d_l1_ptr)
{
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024 * 1024 * 2));

    // L1 table is just 512 pointers = 4KB
    CUDA_CHECK(cudaMalloc(d_l1_ptr, L1_ENTRIES * sizeof(void**)));
    CUDA_CHECK(cudaMemset(*d_l1_ptr, 0, L1_ENTRIES * sizeof(void**)));

    void*** temp = *d_l1_ptr;
    CUDA_CHECK(cudaMemcpyToSymbol(shadow_l1, &temp, sizeof(void***)));
    LOG("[init_tracking] shadow_l1 set to %p\n", temp);
    void*** readback = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_l1, sizeof(void***)));
    LOG("[init_tracking] shadow_l1 → %p (readback %p)\n", temp, readback);

    if(temp != readback) {
        fprintf(stderr, "[init_tracking] error: shadow_l1 readback mismatch!\n");
        exit(1);
    } else {
        LOG("[init_tracking] shadow_l1 readback successful\n");
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Host: walk the page table and dump accessed addresses ────────────────────
void export_log(void*** d_l1, const char* filename)
{
    std::ofstream f(filename);
    LOG("[export_log] writing to %s\n", filename);

    std::vector<void**> h_l1(L1_ENTRIES);
    CUDA_CHECK(cudaMemcpy(h_l1.data(), d_l1,
                          L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost));

    void**              d_l2_stage;
    unsigned long long* d_l3_stage;
    CUDA_CHECK(cudaMalloc(&d_l2_stage, L2_ENTRIES * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_l3_stage, L3_BYTES));

    std::vector<void*>              h_l2(L2_ENTRIES);
    std::vector<unsigned long long> bitmap(L3_BYTES / 8);

    int threads = 256;

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (!h_l1[i]) continue;

        LOG("[export_log] L1[%d] → %p\n", i, h_l1[i]);

        int blocks = (L2_ENTRIES + threads - 1) / threads;
        copy_l2_to_host<<<blocks, threads>>>(d_l1, i, d_l2_stage, L2_ENTRIES);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_l2.data(), d_l2_stage,
                              L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost));
        int accessed_l2 = 0;
        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;
                accessed_l2++;
            
            LOG("[export_log]   L2[%d] → %p\n", j, h_l2[j]);

            int l3_words = L3_BYTES / 8;
            blocks = (l3_words + threads - 1) / threads;
            copy_l3_to_host<<<blocks, threads>>>(d_l1, i, j, d_l3_stage, l3_words);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(bitmap.data(), d_l3_stage,
                                  L3_BYTES, cudaMemcpyDeviceToHost));
            int accessed_pages = 0;
            for (int w = 0; w < (int)bitmap.size(); w++) {
                if (!bitmap[w]) continue;
                accessed_pages += __builtin_popcountll(bitmap[w]);
                for (int b = 0; b < 64; b++) {
                    if (bitmap[w] & (1ULL << b)) {
                        uint64_t l3_offset = (uint64_t)w * 64 + b;
                        // Reconstruct VA: l3_offset is the VPN within this leaf
                        uint64_t vaddr = ((uint64_t)i << L1_SHIFT)
                                        | ((uint64_t)j << L2_SHIFT)
                                        | (l3_offset   << L3_SHIFT);
                        f << std::hex << "0x" << vaddr << "\n";
                    }
                }
            }
            LOG("[export_log]     L3 leaf had %d accessed pages\n", accessed_pages);
        }
        LOG("[export_log] L1 index %d had %d accessed L2 entries\n", i, accessed_l2);
    }

    cudaFree(d_l2_stage);
    cudaFree(d_l3_stage);
}

void export_binary(void*** d_l1, const char* filename)
{
    LOG("[export_binary] starting export\n");
    // check_shadow_l1_kernel<<<1,1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG("[export_binary] writing to %s\n", filename);

    // ── copy L1 to host ───────────────────────────────────────────────────────
    std::vector<void**> h_l1(L1_ENTRIES);
    CUDA_CHECK(cudaMemcpy(h_l1.data(), d_l1,
                          L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost));

    // ── staging buffers ───────────────────────────────────────────────────────
    void**               d_l2_stage;
    unsigned long long*  d_l3_stage;
    CUDA_CHECK(cudaMalloc(&d_l2_stage, L2_ENTRIES * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_l3_stage, L3_BYTES));

    std::vector<void*>              h_l2(L2_ENTRIES);
    std::vector<unsigned long long> bitmap(L3_BYTES / 8);

    int threads = 256;

    // ── first pass: collect all populated leaves ──────────────────────────────
    struct Leaf { uint16_t l1, l2; std::vector<uint8_t> data; };
    std::vector<Leaf> leaves;

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (!h_l1[i]) continue;

        int blocks = (L2_ENTRIES + threads - 1) / threads;
        copy_l2_to_host<<<blocks, threads>>>(d_l1, i, d_l2_stage, L2_ENTRIES);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_l2.data(), d_l2_stage,
                              L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost));

        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;

            int l3_words = L3_BYTES / 8;
            blocks = (l3_words + threads - 1) / threads;
            copy_l3_to_host<<<blocks, threads>>>(d_l1, i, j, d_l3_stage, l3_words);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(bitmap.data(), d_l3_stage,
                                  L3_BYTES, cudaMemcpyDeviceToHost));

            // skip entirely-zero leaves
            bool any = false;
            for (auto w : bitmap) if (w) { any = true; break; }
            if (!any) continue;

            Leaf leaf;
            leaf.l1   = (uint16_t)i;
            leaf.l2   = (uint16_t)j;
            leaf.data.resize(L3_BYTES);
            memcpy(leaf.data.data(), bitmap.data(), L3_BYTES);
            leaves.push_back(std::move(leaf));
        }
    }

    cudaFree(d_l2_stage);
    cudaFree(d_l3_stage);

    LOG("[export_binary] %zu populated leaves\n", leaves.size());

    // ── write file ────────────────────────────────────────────────────────────
    std::ofstream f(filename, std::ios::binary);
    if (!f) { fprintf(stderr, "[export_binary] failed to open %s\n", filename); return; }

    // header
    PageLogHeader hdr;
    hdr.magic      = PAGELOG_MAGIC;
    hdr.version    = PAGELOG_VERSION;
    hdr.l1_entries = (uint16_t)L1_ENTRIES;
    hdr.l2_entries = (uint16_t)L2_ENTRIES;
    hdr.l3_bytes   = (uint16_t)L3_BYTES;
    hdr.l1_shift   = L1_SHIFT;
    hdr.l2_shift   = L2_SHIFT;
    hdr.l3_shift   = L3_SHIFT;
    hdr.num_leaves = (uint64_t)leaves.size();
    f.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    // index — compute offsets
    uint64_t data_start = sizeof(PageLogHeader)
                        + leaves.size() * sizeof(PageLogIndexEntry);
    std::vector<PageLogIndexEntry> index(leaves.size());
    uint64_t off = data_start;
    for (size_t k = 0; k < leaves.size(); k++) {
        index[k].l1_idx = leaves[k].l1;
        index[k].l2_idx = leaves[k].l2;
        index[k].offset = off;
        off += L3_BYTES;
    }
    f.write(reinterpret_cast<char*>(index.data()),
            leaves.size() * sizeof(PageLogIndexEntry));

    // bitmap data
    for (auto& leaf : leaves)
        f.write(reinterpret_cast<char*>(leaf.data.data()), L3_BYTES);

    LOG("[export_binary] done — %zu bytes written\n", (size_t)off);
}