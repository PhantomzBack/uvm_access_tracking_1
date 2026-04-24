#include "common.h"
#include "tracking.h"
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// COMPILE-TIME MODE FLAGS
// ═══════════════════════════════════════════════════════════════════════════════
#if !defined(TRACKING_MODE_SKIP) && !defined(TRACKING_MODE_ALLOC) && !defined(TRACKING_MODE_NONE)
#define TRACKING_MODE_ALLOC
#endif

#ifdef TRACKING_MODE_SKIP
#define TRACKING_ALLOC_ON_MISS 0
#define USE_PRELOAD 1
#define USE_MANAGED_L2 1
#elif defined(TRACKING_MODE_ALLOC)
#define TRACKING_ALLOC_ON_MISS 1
#define USE_PRELOAD 1
#define USE_MANAGED_L2 0
#else // TRACKING_MODE_NONE
#define TRACKING_ALLOC_ON_MISS 1
#define USE_PRELOAD 0
#define USE_MANAGED_L2 0
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// FAST-PATH DEVICE VARIABLES
//   __device__ (not __constant__) to prevent clang++ from constant-folding the
//   initial zero and dead-code eliminating the tracking body.
// ═══════════════════════════════════════════════════════════════════════════════
__device__ int g_tracking_enabled = 0;
#ifdef TRACKING_MODE_ALLOC
__device__ int g_skip_on_miss = 0;
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// HOST-ACCESSIBLE GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" void** g_uvm_shadow_l1 = nullptr;

// ═══════════════════════════════════════════════════════════════════════════════
// DEVICE GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" {
    __device__ void*** shadow_l1 = nullptr;

    // ── MarkAccess ────────────────────────────────────────────────────────────
    __device__ void MarkAccess(uintptr_t addr)
    {
        if (!g_tracking_enabled) return;
        if (!shadow_l1) return;


        #define LEADER_SELECT_MARK \
        unsigned full = __activemask(); \
        uint32_t word_idx = l3_offset / 64; \
        unsigned long long my_mask = 1ULL << (l3_offset % 64); \
        uint32_t key = (l1_idx << 21) | (l2_idx << 9) | word_idx; \
        unsigned long long agg = my_mask; \
        for (int off = 16; off > 0; off >>= 1) { \
            unsigned long long other = __shfl_xor_sync(full, agg, off); \
            uint32_t other_key = __shfl_xor_sync(full, key, off); \
            if (other_key == key) agg |= other; \
        } \
        int lane = threadIdx.x & 31; \
        uint32_t group_mask = __match_any_sync(full, key); \
        int leader = __ffs(group_mask) - 1; \
        if (lane == leader) { \
            unsigned long long* word = &l3_bitmap[word_idx]; \
            unsigned long long prev = *word; \
            unsigned long long delta = agg & ~prev; \
            if (delta) atomicOr(word, delta); \
            unsigned long long after = *word; \
            LOG("[MarkAccess] after atomicOr, bit is now %lld (delta=0x%llx)\\n", \
                ((after >> (l3_offset % 64)) & 1ULL), delta); \
        } \


#define SKIPPED_ACCESS(addr) \
        uint32_t l1_idx = (uint32_t)((addr >> L1_SHIFT) & L1_MASK); \
        void** l2_table = (void**)shadow_l1[l1_idx]; \
        if(!l2_table) return; \
        uint32_t l2_idx = (uint32_t)((addr >> L2_SHIFT) & L2_MASK); \
        void* l3_leaf = l2_table[l2_idx]; \
        if (!l3_leaf) return; \
        uint32_t l3_offset = (uint32_t)((addr >> L3_SHIFT) & L3_MASK); \
        unsigned long long* l3_bitmap = (unsigned long long*)l3_leaf; \
        if (!l3_bitmap) return; \
        if ((l3_bitmap[l3_offset / 64] >> (l3_offset % 64)) & 1ULL) { \
            LOG("[MarkAccess] page already marked accessed, skipping\n"); \
            return; \
        } \
        LEADER_SELECT_MARK \

#define ALLOC_ACCESS(addr) do { \
    /* Calculate L1 and fetch L2 table */ \
    uint32_t _l1_idx = ((addr) >> L1_SHIFT) & L1_MASK; \
    if (shadow_l1[_l1_idx] == nullptr) { \
        void** _new_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*)); \
        if (_new_l2) { \
            memset(_new_l2, 0, L2_ENTRIES * sizeof(void*)); \
            if (atomicCAS((unsigned long long*)&shadow_l1[_l1_idx], 0ULL, (unsigned long long)_new_l2) != 0ULL) { \
                free(_new_l2); \
            } \
        } \
    } \
    void** _l2_table = (void**)shadow_l1[_l1_idx]; \
    if (!_l2_table) break; \
 \
    /* Interleaved: Calculate L2 and fetch L3 leaf */ \
    uint32_t _l2_idx = ((addr) >> L2_SHIFT) & L2_MASK; \
    if (_l2_table[_l2_idx] == nullptr) { \
        void* _new_l3 = malloc(L3_BYTES); \
        if (_new_l3) { \
            memset(_new_l3, 0, L3_BYTES); \
            if (atomicCAS((unsigned long long*)&_l2_table[_l2_idx], 0ULL, (unsigned long long)_new_l3) != 0ULL) { \
                free(_new_l3); \
            } \
        } \
    } \
    unsigned long long* _l3_bitmap = (unsigned long long*)_l2_table[_l2_idx]; \
    if (!_l3_bitmap) break; \
 \
    /* Interleaved: Calculate bit offset and perform warp-aggregated atomicOr */ \
    uint32_t _l3_offset = ((addr) >> L3_SHIFT) & L3_MASK; \
    uint32_t _word_idx = _l3_offset / 64; \
    unsigned long long _my_mask = 1ULL << (_l3_offset % 64); \
    uint32_t _key = (_l1_idx << 21) | (_l2_idx << 9) | _word_idx; \
 \
    unsigned _full = __activemask(); \
    unsigned long long _agg = _my_mask; \
    for (int _off = 16; _off > 0; _off >>= 1) { \
        unsigned long long _other = __shfl_xor_sync(_full, _agg, _off); \
        uint32_t _other_key = __shfl_xor_sync(_full, _key, _off); \
        if (_other_key == _key) _agg |= _other; \
    } \
 \
    uint32_t _group_mask = __match_any_sync(_full, _key); \
    int _leader = __ffs(_group_mask) - 1; \
    if ((threadIdx.x & 31) == _leader) { \
        unsigned long long* _word_ptr = &_l3_bitmap[_word_idx]; \
        /* Snooping: only fire atomic if bits aren't already set */ \
        if ((*_word_ptr & _agg) != _agg) { \
            atomicOr(_word_ptr, _agg); \
        } \
    } \
} while (0)
        
        // // Decompose address
        // uint32_t l1_idx    = (addr >> L1_SHIFT) & L1_MASK;
        // uint32_t l2_idx    = (addr >> L2_SHIFT) & L2_MASK;
        // uint32_t l3_offset = (addr >> L3_SHIFT) & L3_MASK; // bit index within L3 leaf


#ifdef TRACKING_MODE_SKIP
        SKIPPED_ACCESS(addr);
#elif defined(TRACKING_MODE_ALLOC)
        if (g_skip_on_miss) {
            SKIPPED_ACCESS(addr);
        }
        else {
            ALLOC_ACCESS(addr);
        }
#else
        ALLOC_ACCESS(addr);
#endif
    }

    // ── BatchMarkAccess ───────────────────────────────────────────────────────
    __device__ void BatchMarkAccess(uintptr_t base_addr, int64_t stride, uint64_t count)
    {
        if (!g_tracking_enabled || !shadow_l1 || count == 0) return;

#if defined(TRACKING_MODE_SKIP)
#define MARK_PAGE(addr)                                                          \
        do {                                                                     \
            uint32_t _l1 = (uint32_t)(((addr) >> L1_SHIFT) & L1_MASK);          \
            uint32_t _l2 = (uint32_t)(((addr) >> L2_SHIFT) & L2_MASK);          \
            uint32_t _l3 = (uint32_t)(((addr) >> L3_SHIFT) & L3_MASK);          \
            if (_l1 != cached_l1) {                                              \
                if (!shadow_l1[_l1]) break;                                      \
                l2_table  = (void**)shadow_l1[_l1];                             \
                cached_l1 = _l1;  cached_l2 = (uint32_t)-1;  l3_bitmap = nullptr; \
            }                                                                    \
            if (_l2 != cached_l2) {                                              \
                if (!l2_table[_l2]) break;                                       \
                l3_bitmap = (unsigned long long*)l2_table[_l2];                 \
                cached_l2 = _l2;                                                 \
            }                                                                    \
            if (l3_bitmap) {                                                     \
                unsigned long long* _w = &l3_bitmap[_l3 / 64];                  \
                unsigned long long  _m = 1ULL << (_l3 % 64);                    \
                if (!(*_w & _m)) atomicOr(_w, _m);                              \
            }                                                                    \
        } while (0)

#elif defined(TRACKING_MODE_ALLOC)
#define MARK_PAGE(addr)                                                          \
        do {                                                                     \
            uint32_t _l1 = (uint32_t)(((addr) >> L1_SHIFT) & L1_MASK);          \
            uint32_t _l2 = (uint32_t)(((addr) >> L2_SHIFT) & L2_MASK);          \
            uint32_t _l3 = (uint32_t)(((addr) >> L3_SHIFT) & L3_MASK);          \
            if (_l1 != cached_l1) {                                              \
                if (!shadow_l1[_l1] && !g_skip_on_miss) {                        \
                    void** _nl2 = (void**)malloc(L2_ENTRIES * sizeof(void*));    \
                    if (!_nl2) break;                                             \
                    unsigned long long _old = atomicCAS(                         \
                        (unsigned long long*)&shadow_l1[_l1],                   \
                        0ULL, (unsigned long long)_nl2);                         \
                    if (_old != 0ULL) free(_nl2);                                \
                    else              memset(_nl2, 0, L2_ENTRIES * sizeof(void*)); \
                }                                                                \
                if (!shadow_l1[_l1]) break;                                      \
                l2_table  = (void**)shadow_l1[_l1];                             \
                cached_l1 = _l1;  cached_l2 = (uint32_t)-1;  l3_bitmap = nullptr; \
            }                                                                    \
            if (_l2 != cached_l2) {                                              \
                if (!l2_table[_l2] && !g_skip_on_miss) {                         \
                    void* _nl3 = malloc(L3_BYTES);                               \
                    if (!_nl3) break;                                             \
                    unsigned long long _old = atomicCAS(                         \
                        (unsigned long long*)&l2_table[_l2],                    \
                        0ULL, (unsigned long long)_nl3);                         \
                    if (_old != 0ULL) free(_nl3);                                \
                    else              memset(_nl3, 0, L3_BYTES);                 \
                }                                                                \
                if (!l2_table[_l2]) break;                                       \
                l3_bitmap = (unsigned long long*)l2_table[_l2];                 \
                cached_l2 = _l2;                                                 \
            }                                                                    \
            if (l3_bitmap) {                                                     \
                unsigned long long* _w = &l3_bitmap[_l3 / 64];                  \
                unsigned long long  _m = 1ULL << (_l3 % 64);                    \
                if (!(*_w & _m)) atomicOr(_w, _m);                              \
            }                                                                    \
        } while (0)

#else
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
#endif

        uint32_t cached_l1 = (uint32_t)-1;
        uint32_t cached_l2 = (uint32_t)-1;
        void**   l2_table  = nullptr;
        unsigned long long* l3_bitmap = nullptr;

        if (stride > 0 && (uint64_t)stride < (1ULL << L3_SHIFT)) {
            uintptr_t start_page = base_addr >> 12;
            uintptr_t end_page   = (base_addr + (uint64_t)(count - 1) * (uint64_t)stride) >> 12;

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

    // ── Staging kernels (used by legacy export path when L2 is device memory) ──
    __global__ void copy_l2_to_host(void*** l1, int l1_idx, void** out, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n)
            out[tid] = ((void**)l1[l1_idx])[tid];
    }

    __global__ void copy_l3_to_host(void*** l1, int l1_idx, int l2_idx,
                                    unsigned long long* out, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n)
            out[tid] = ((unsigned long long*)((void**)l1[l1_idx])[l2_idx])[tid];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: INITIALISATION
// ═══════════════════════════════════════════════════════════════════════════════
void init_tracking(void**** d_l1_ptr)
{
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024 * 1024 * 2));

#if USE_MANAGED_L2
    CUDA_CHECK(cudaMallocManaged(d_l1_ptr, L1_ENTRIES * sizeof(void**)));
#else
    CUDA_CHECK(cudaMalloc(d_l1_ptr, L1_ENTRIES * sizeof(void**)));
#endif
    CUDA_CHECK(cudaMemset(*d_l1_ptr, 0, L1_ENTRIES * sizeof(void**)));

    void*** temp = *d_l1_ptr;
    CUDA_CHECK(cudaMemcpyToSymbol(shadow_l1, &temp, sizeof(void***)));
    LOG("[init_tracking] shadow_l1 set to %p\n", temp);

    void*** readback = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_l1, sizeof(void***)));
    if (temp != readback) {
        fprintf(stderr, "[init_tracking] error: shadow_l1 readback mismatch!\n");
        exit(1);
    }

    g_uvm_shadow_l1 = (void**)temp;

#if USE_MANAGED_L2
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(temp, L1_ENTRIES * sizeof(void**), device, 0);
#endif

    int enabled = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(g_tracking_enabled, &enabled, sizeof(int)));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: PRE-POPULATE SHADOW PAGE TABLE (called by LD_PRELOAD wrapper)
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" void uvm_tracking_preload_range(uintptr_t start, size_t size)
{
    if (!g_uvm_shadow_l1 || !start || size == 0)
        return;

    static std::mutex preload_mutex;
    std::lock_guard<std::mutex> lock(preload_mutex);

    uintptr_t end = start + size - 1;
    uint32_t l1_start = (uint32_t)((start >> L1_SHIFT) & L1_MASK);
    uint32_t l1_end   = (uint32_t)((end   >> L1_SHIFT) & L1_MASK);

    for (uint32_t li = l1_start; li <= l1_end; ++li) {
        if (!g_uvm_shadow_l1[li]) {
            void** new_l2 = nullptr;
#if USE_MANAGED_L2
            cudaError_t err = cudaMallocManaged(&new_l2, L2_ENTRIES * sizeof(void*));
            if (err != cudaSuccess) {
                fprintf(stderr, "[preload] cudaMallocManaged L2 failed: %s\n",
                        cudaGetErrorString(err));
                return;
            }
            memset(new_l2, 0, L2_ENTRIES * sizeof(void*));
            g_uvm_shadow_l1[li] = new_l2;
#else
            cudaError_t err = cudaMalloc(&new_l2, L2_ENTRIES * sizeof(void*));
            if (err != cudaSuccess) {
                fprintf(stderr, "[preload] cudaMalloc L2 failed: %s\n",
                        cudaGetErrorString(err));
                return;
            }
            cudaMemset(new_l2, 0, L2_ENTRIES * sizeof(void*));
            cudaMemcpy(&g_uvm_shadow_l1[li], &new_l2, sizeof(void**),
                       cudaMemcpyHostToDevice);
#endif
        }

        uint32_t l2_start = (li == l1_start)
                                ? (uint32_t)((start >> L2_SHIFT) & L2_MASK)
                                : 0;
        uint32_t l2_end   = (li == l1_end)
                                ? (uint32_t)((end >> L2_SHIFT) & L2_MASK)
                                : (L2_ENTRIES - 1);

#if USE_MANAGED_L2
        void** l2_table = (void**)g_uvm_shadow_l1[li];
        for (uint32_t lj = l2_start; lj <= l2_end; ++lj) {
            if (!l2_table[lj]) {
                void* new_l3 = nullptr;
                cudaError_t err = cudaMalloc(&new_l3, L3_BYTES);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[preload] cudaMalloc L3 failed: %s\n",
                            cudaGetErrorString(err));
                    return;
                }
                cudaMemset(new_l3, 0, L3_BYTES);
                l2_table[lj] = new_l3;
            }
        }
#else
        void* h_l2[L2_ENTRIES];
        cudaMemcpy(h_l2, g_uvm_shadow_l1[li], L2_ENTRIES * sizeof(void*),
                   cudaMemcpyDeviceToHost);
        for (uint32_t lj = l2_start; lj <= l2_end; ++lj) {
            if (!h_l2[lj]) {
                void* new_l3 = nullptr;
                cudaError_t err = cudaMalloc(&new_l3, L3_BYTES);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[preload] cudaMalloc L3 failed: %s\n",
                            cudaGetErrorString(err));
                    return;
                }
                cudaMemset(new_l3, 0, L3_BYTES);
                cudaMemcpy(&((void**)g_uvm_shadow_l1[li])[lj], &new_l3, sizeof(void*),
                           cudaMemcpyHostToDevice);
            }
        }
#endif
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: PAGEDUMP EXPORT (text format – legacy, for debugging)
// ═══════════════════════════════════════════════════════════════════════════════
void export_log(void*** d_l1, const char* filename)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    void** d_l2 = nullptr;
    unsigned long long* d_l3 = nullptr;
#if !USE_MANAGED_L2
    cudaMalloc(&d_l2, L2_ENTRIES * sizeof(void*));
    cudaMalloc(&d_l3, L3_BYTES);
#endif

    for (int i = 0; i < L1_ENTRIES; i++) {
        void** l2_table = nullptr;
#if USE_MANAGED_L2
        l2_table = (void**)d_l1[i];
#else
        cudaMemcpy(&l2_table, &d_l1[i], sizeof(void*), cudaMemcpyDeviceToHost);
#endif
        if (!l2_table) continue;

        void** h_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*));
        if (!h_l2) continue;

#if USE_MANAGED_L2
        memcpy(h_l2, l2_table, L2_ENTRIES * sizeof(void*));
#else
        copy_l2_to_host<<< (L2_ENTRIES + 255) / 256, 256 >>>(d_l1, i, d_l2, L2_ENTRIES);
        cudaDeviceSynchronize();
        cudaMemcpy(h_l2, d_l2, L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);
#endif

        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;
            unsigned long long* h_l3 = (unsigned long long*)malloc(L3_BYTES);
            if (!h_l3) continue;
#if USE_MANAGED_L2
            copy_l3_to_host<<< 1, 512 >>>(d_l1, i, j, h_l3, L3_BYTES / 8);
            cudaDeviceSynchronize();
#else
            copy_l3_to_host<<< 1, 512 >>>(d_l1, i, j, d_l3, L3_BYTES / 8);
            cudaDeviceSynchronize();
            cudaMemcpy(h_l3, d_l3, L3_BYTES, cudaMemcpyDeviceToHost);
#endif

            for (int w = 0; w < L3_BYTES / 8; w++) {
                if (h_l3[w]) {
                    for (int b = 0; b < 64; b++) {
                        if (h_l3[w] & (1ULL << b)) {
                            uintptr_t addr = (((uintptr_t)i) << L1_SHIFT)
                                           | (((uintptr_t)j) << L2_SHIFT)
                                           | (((uintptr_t)(w * 64 + b)) << L3_SHIFT);
                            fprintf(fp, "0x%016lx\n", addr);
                        }
                    }
                }
            }
            free(h_l3);
        }
        free(h_l2);
    }
#if !USE_MANAGED_L2
    cudaFree(d_l2);
    cudaFree(d_l3);
#endif
    fclose(fp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: BINARY EXPORT (compact pagelog format)
// ═══════════════════════════════════════════════════════════════════════════════
void export_binary(void*** d_l1, const char* filename)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    struct Leaf { uint16_t l1, l2; std::vector<uint8_t> data; };
    std::vector<Leaf> leaves;

    void** d_l2 = nullptr;
    unsigned long long* d_l3 = nullptr;
#if !USE_MANAGED_L2
    cudaMalloc(&d_l2, L2_ENTRIES * sizeof(void*));
    cudaMalloc(&d_l3, L3_BYTES);
#endif

    for (int i = 0; i < L1_ENTRIES; i++) {
        void** l2_table = nullptr;
#if USE_MANAGED_L2
        l2_table = (void**)d_l1[i];
#else
        cudaMemcpy(&l2_table, &d_l1[i], sizeof(void*), cudaMemcpyDeviceToHost);
#endif
        if (!l2_table) continue;

        void** h_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*));
        if (!h_l2) continue;

#if USE_MANAGED_L2
        memcpy(h_l2, l2_table, L2_ENTRIES * sizeof(void*));
#else
        copy_l2_to_host<<< (L2_ENTRIES + 255) / 256, 256 >>>(d_l1, i, d_l2, L2_ENTRIES);
        cudaDeviceSynchronize();
        cudaMemcpy(h_l2, d_l2, L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);
#endif

        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;
            unsigned long long* h_l3 = (unsigned long long*)malloc(L3_BYTES);
            if (!h_l3) continue;
#if USE_MANAGED_L2
            copy_l3_to_host<<< 1, 512 >>>(d_l1, i, j, h_l3, L3_BYTES / 8);
            cudaDeviceSynchronize();
#else
            copy_l3_to_host<<< 1, 512 >>>(d_l1, i, j, d_l3, L3_BYTES / 8);
            cudaDeviceSynchronize();
            cudaMemcpy(h_l3, d_l3, L3_BYTES, cudaMemcpyDeviceToHost);
#endif

            bool any = false;
            for (int w = 0; w < L3_BYTES / 8; w++) if (h_l3[w]) { any = true; break; }
            if (!any) { free(h_l3); continue; }

            Leaf leaf;
            leaf.l1 = (uint16_t)i;
            leaf.l2 = (uint16_t)j;
            leaf.data.resize(L3_BYTES);
            memcpy(leaf.data.data(), h_l3, L3_BYTES);
            leaves.push_back(std::move(leaf));
            free(h_l3);
        }
        free(h_l2);
    }
#if !USE_MANAGED_L2
    cudaFree(d_l2);
    cudaFree(d_l3);
#endif

    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

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
    fwrite(&hdr, sizeof(hdr), 1, fp);

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
    fwrite(index.data(), leaves.size() * sizeof(PageLogIndexEntry), 1, fp);

    for (auto& leaf : leaves)
        fwrite(leaf.data.data(), L3_BYTES, 1, fp);

    fclose(fp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: LIVE SNAPSHOT (async, for control thread)
// ═══════════════════════════════════════════════════════════════════════════════

struct OrphanedL3 {
    void*   ptr;
    uint16_t l1_idx;
    uint16_t l2_idx;
};

static cudaStream_t g_snapshot_stream = nullptr;

extern "C" void export_binary_live(const char* filename, bool reset,
                                   std::vector<OrphanedL3>* out_orphans)
{
    if (!g_snapshot_stream) {
        cudaStreamCreateWithFlags(&g_snapshot_stream, cudaStreamNonBlocking);
    }

    struct LeafInfo { uint16_t l1, l2; void* l3; };
    std::vector<LeafInfo> leaves;
    leaves.reserve(4096);

    for (int i = 0; i < L1_ENTRIES; i++) {
        void** l2_table = nullptr;
#if USE_MANAGED_L2
        l2_table = (void**)g_uvm_shadow_l1[i];
#else
        cudaMemcpy(&l2_table, &g_uvm_shadow_l1[i], sizeof(void*), cudaMemcpyDeviceToHost);
#endif
        if (!l2_table) continue;

        void** h_l2 = nullptr;
#if USE_MANAGED_L2
        h_l2 = l2_table;
#else
        h_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*));
        copy_l2_to_host<<< (L2_ENTRIES + 255) / 256, 256 >>>((void***)g_uvm_shadow_l1, i, h_l2, L2_ENTRIES);
        cudaDeviceSynchronize();
#endif

        for (int j = 0; j < L2_ENTRIES; j++) {
            if (h_l2[j]) leaves.push_back({(uint16_t)i, (uint16_t)j, h_l2[j]});
        }
#if !USE_MANAGED_L2
        free(h_l2);
#endif
    }

    std::vector<unsigned char*> staging;
    staging.reserve(leaves.size());
    for (size_t k = 0; k < leaves.size(); k++) {
        unsigned char* buf = nullptr;
        cudaMallocHost(&buf, L3_BYTES);
        staging.push_back(buf);
        cudaMemcpyAsync(buf, leaves[k].l3, L3_BYTES,
                        cudaMemcpyDeviceToHost, g_snapshot_stream);
    }
    cudaStreamSynchronize(g_snapshot_stream);

    if (reset) {
        for (const auto& leaf : leaves) {
#if USE_MANAGED_L2
            void** l2 = (void**)g_uvm_shadow_l1[leaf.l1];
            if (l2 && l2[leaf.l2] == leaf.l3) {
                l2[leaf.l2] = nullptr;
                if (out_orphans)
                    out_orphans->push_back({leaf.l3, leaf.l1, leaf.l2});
            }
#else
            // Cannot safely invalidate device L2 from host
#endif
        }
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        for (auto* buf : staging) cudaFreeHost(buf);
        return;
    }

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
    fwrite(&hdr, sizeof(hdr), 1, fp);

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
    fwrite(index.data(), leaves.size() * sizeof(PageLogIndexEntry), 1, fp);

    for (size_t k = 0; k < leaves.size(); k++)
        fwrite(staging[k], L3_BYTES, 1, fp);

    for (auto* buf : staging) cudaFreeHost(buf);
    fclose(fp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: LIVE CONTROL & SNAPSHOT THREAD
// ═══════════════════════════════════════════════════════════════════════════════

static std::string g_control_sock_path;
static std::atomic<bool> g_control_running{false};
static std::thread g_control_thread;
static std::mutex g_control_mutex;

// Forward declaration
static void start_control_thread();

// ── Internal helpers ──────────────────────────────────────────────────────────
static void send_response(int client_fd, const std::string& msg)
{
    write(client_fd, msg.c_str(), msg.size());
    write(client_fd, "\n", 1);
}

static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, delim))
        if (!part.empty()) parts.push_back(part);
    return parts;
}

// ── Host-side range invalidation (managed L2 only) ────────────────────────────
static bool drop_range_host(uintptr_t start, size_t size)
{
#if !USE_MANAGED_L2
    return false;
#endif
    if (!g_uvm_shadow_l1 || !start || size == 0)
        return false;

    uintptr_t end = start + size - 1;
    uint32_t l1_start = (uint32_t)((start >> L1_SHIFT) & L1_MASK);
    uint32_t l1_end   = (uint32_t)((end   >> L1_SHIFT) & L1_MASK);

    for (uint32_t li = l1_start; li <= l1_end; ++li) {
        void** l2 = (void**)g_uvm_shadow_l1[li];
        if (!l2) continue;

        uint32_t l2_start = (li == l1_start)
                              ? (uint32_t)((start >> L2_SHIFT) & L2_MASK)
                              : 0;
        uint32_t l2_end   = (li == l1_end)
                              ? (uint32_t)((end >> L2_SHIFT) & L2_MASK)
                              : (L2_ENTRIES - 1);

        for (uint32_t lj = l2_start; lj <= l2_end; ++lj) {
            void* l3 = l2[lj];
            if (l3) {
                l2[lj] = nullptr;
                cudaFree(l3);
            }
        }
    }
    return true;
}

// ── Control thread main loop ──────────────────────────────────────────────────
static void control_thread_fn()
{
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        fprintf(stderr, "[uvm-ctl] socket() failed: %s\n", strerror(errno));
        return;
    }

    int reuse = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_control_sock_path.c_str(), sizeof(addr.sun_path) - 1);

    unlink(g_control_sock_path.c_str());
    if (bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[uvm-ctl] bind(%s) failed: %s\n",
                g_control_sock_path.c_str(), strerror(errno));
        close(sock_fd);
        return;
    }

    if (listen(sock_fd, 4) < 0) {
        fprintf(stderr, "[uvm-ctl] listen() failed: %s\n", strerror(errno));
        close(sock_fd);
        return;
    }

    fprintf(stderr, "[uvm-ctl] listening on %s\n", g_control_sock_path.c_str());

    while (g_control_running.load()) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock_fd, &fds);
        struct timeval tv{ .tv_sec = 0, .tv_usec = 100000 };

        int rc = select(sock_fd + 1, &fds, nullptr, nullptr, &tv);
        if (rc <= 0) continue;

        int client_fd = accept(sock_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        char buf[256];
        ssize_t n = read(client_fd, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string line = trim(buf);
            std::vector<std::string> parts = split(line, ' ');

            if (parts.empty()) {
                send_response(client_fd, "ERR empty command");
            } else {
                std::string cmd = parts[0];
                std::lock_guard<std::mutex> lock(g_control_mutex);

                if (cmd == "enable") {
                    int val = 1;
                    cudaMemcpyToSymbol(g_tracking_enabled, &val, sizeof(int));
                    send_response(client_fd, "OK tracking enabled");
                }
                else if (cmd == "disable") {
                    int val = 0;
                    cudaMemcpyToSymbol(g_tracking_enabled, &val, sizeof(int));
                    send_response(client_fd, "OK tracking disabled");
                }
                else if (cmd == "mode") {
#ifdef TRACKING_MODE_ALLOC
                    if (parts.size() < 2) {
                        send_response(client_fd, "ERR usage: mode {skip|alloc}");
                    } else {
                        int val = (parts[1] == "skip") ? 1 : 0;
                        cudaMemcpyToSymbol(g_skip_on_miss, &val, sizeof(int));
                        send_response(client_fd, std::string("OK mode=") + parts[1]);
                    }
#else
                    send_response(client_fd, "ERR mode toggle not supported in this build");
#endif
                }
                else if (cmd == "snapshot") {
                    std::string out_path = "/tmp/uvm-snapshot-" + std::to_string(getpid()) + ".pagelog";
                    if (parts.size() >= 2) out_path = parts[1];
                    bool reset = false;
                    if (parts.size() >= 3 && parts[2] == "reset") reset = true;
                    export_binary_live(out_path.c_str(), reset, nullptr);
                    send_response(client_fd, std::string("OK snapshot=") + out_path);
                }
                else if (cmd == "drop") {
                    if (parts.size() < 3) {
                        send_response(client_fd, "ERR usage: drop <hex_start> <hex_size>");
                    } else {
                        uintptr_t start = strtoull(parts[1].c_str(), nullptr, 16);
                        size_t    size  = strtoull(parts[2].c_str(), nullptr, 16);
                        bool ok = drop_range_host(start, size);
                        if (ok)
                            send_response(client_fd, "OK dropped");
                        else
                            send_response(client_fd, "ERR drop not supported in this mode");
                    }
                }
                else if (cmd == "status") {
                    int enabled = 0;
                    cudaMemcpyFromSymbol(&enabled, g_tracking_enabled, sizeof(int));
                    std::ostringstream oss;
                    oss << "OK enabled=" << enabled;
#ifdef TRACKING_MODE_ALLOC
                    int skip = 0;
                    cudaMemcpyFromSymbol(&skip, g_skip_on_miss, sizeof(int));
                    oss << " skip=" << skip;
#else
                    oss << " skip=N/A";
#endif
                    oss << " l1=" << (void*)g_uvm_shadow_l1
                        << " mode=";
#ifdef TRACKING_MODE_SKIP
                    oss << "skip";
#elif defined(TRACKING_MODE_ALLOC)
                    oss << "alloc";
#else
                    oss << "none";
#endif
                    send_response(client_fd, oss.str());
                }
                else {
                    send_response(client_fd, "ERR unknown command: " + cmd);
                }
            }
        }
        close(client_fd);
    }

    close(sock_fd);
    unlink(g_control_sock_path.c_str());
    fprintf(stderr, "[uvm-ctl] thread exiting\n");
}

// ── Thread launcher (idempotent) ──────────────────────────────────────────────
static void start_control_thread()
{
    static std::atomic<bool> started{false};
    bool expected = false;
    if (!started.compare_exchange_strong(expected, true))
        return;

    std::ostringstream oss;
    oss << "/tmp/uvm-ctl." << getpid();
    g_control_sock_path = oss.str();

    g_control_running.store(true);
    g_control_thread = std::thread(control_thread_fn);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST: PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" void uvm_tracking_enable(void)
{
    int val = 1;
    cudaMemcpyToSymbol(g_tracking_enabled, &val, sizeof(int));
    start_control_thread();
}

extern "C" void uvm_tracking_disable(void)
{
    int val = 0;
    cudaMemcpyToSymbol(g_tracking_enabled, &val, sizeof(int));
}

extern "C" void uvm_tracking_set_mode(int skip_on_miss)
{
#ifdef TRACKING_MODE_ALLOC
    int val = skip_on_miss ? 1 : 0;
    cudaMemcpyToSymbol(g_skip_on_miss, &val, sizeof(int));
#else
    (void)skip_on_miss;
#endif
}

extern "C" const char* uvm_tracking_get_socket_path(void)
{
    return g_control_sock_path.empty() ? nullptr : g_control_sock_path.c_str();
}
