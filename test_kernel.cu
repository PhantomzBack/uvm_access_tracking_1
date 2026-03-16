#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

#define NVCC_TESTING
#define MA_TESTING
#define DBG printf("At line: %d\n", __LINE__);
#define DBG_RET(ret) printf("At line: %d, returning %d\n", __LINE__, ret); return ret;
#define LOG printf

// ── Address decomposition (48-bit VA, 2-level + bitmap) ──────────────────────
//
//  [ 47 ......... 39 | 38 ......... 30 | 29 ......... 12 | 11 ......... 0 ]
//      L1 index (9b)     L2 index (9b)    L3 offset (18b)   page offset (12b)
//
//  Wait: 9 + 9 + 18 = 36 bits of VPN ✓  (2^36 pages in 48-bit space)
//
//  L1 table : 512  void**  pointers  (4KB, one page, static)
//  L2 table : 512  void*   pointers  (4KB, one page, on demand)
//  L3 leaf  : 2^18 bits = 32768 bytes = 32KB bitmap (on demand)
//             covers 2^18 pages = 2^18 * 2^12 = 1GB of VA per leaf

#define L1_ENTRIES   512          // 2^9  — one page
#define L2_ENTRIES   4096         // 2^12 — eight pages
#define L3_BYTES     4096         // 2^15 bits = 4096 bytes — one page

#define L3_SHIFT     12           // bits 12–26  (15 bits)
#define L2_SHIFT     27           // bits 27–38  (12 bits)
#define L1_SHIFT     39           // bits 39–47  (9 bits)
#define L1_MASK      0x1FFULL     // 9-bit mask
#define L2_MASK      0xFFFULL     // 12-bit mask
#define L3_MASK      0x7FFFULL    // 15-bit mask

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Device globals ────────────────────────────────────────────────────────────
extern "C" {
    // shadow_l1 points to the L1 table (array of 512 void** pointers)
    __device__ void*** shadow_l1 = nullptr;

    __device__ int mark_access_invoked = 0;

    // ── MarkAccess ────────────────────────────────────────────────────────────
    __device__ void MarkAccess(uintptr_t addr)
    {
        mark_access_invoked = 1;

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

            unsigned long long* init = (unsigned long long*)new_l3;
            // for (int i = 0; i < (L3_BYTES / 8); i++)
            //     init[i] = 0ULL;

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


__global__ void copy_l2_to_host(void*** l1, int l1_idx, void** out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
    if (tid < n)
        out[tid] = ((void**)l1[l1_idx])[tid];
    if (tid == n-1)
        printf("[copy_l2_to_host] successful\n");
}

__global__ void copy_l3_to_host(void*** l1, int l1_idx, int l2_idx,
                                 unsigned long long* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
    if (tid < n)
        out[tid] = ((unsigned long long*)((void**)l1[l1_idx])[l2_idx])[tid];
    if (tid == n-1)
        printf("[copy_l2_to_host] successful\n");

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

    void*** readback = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_l1, sizeof(void***)));
    printf("[init_tracking] shadow_l1 → %p (readback %p)\n", temp, readback);
}

// ── Host: walk the page table and dump accessed addresses ────────────────────
void export_log(void*** d_l1, const char* filename)
{
    std::ofstream f(filename);
    printf("[export_log] writing to %s\n", filename);

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

// ── Test kernels ──────────────────────────────────────────────────────────────
__global__ void myKernel()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[myKernel] Hello from GPU\n");
}

__global__ void stride_access(int* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid * 1024] = tid;
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

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("[main] sync error: %s\n", cudaGetErrorString(err));
    else
        printf("[main] kernels completed successfully\n");

    export_log(d_l1, "access_log.txt");

    cudaFree(d_data);
    cudaFree(d_l1);
    return 0;
}
