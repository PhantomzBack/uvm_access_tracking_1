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

// ── Address decomposition (48-bit VA, 3-level, 12/12/12/12 split) ────────────
//
//  [ 47 ........... 36 | 35 ......... 24 | 23 ......... 12 | 11 ......... 0 ]
//      L1 index (12b)      L2 index (12b)    L3 offset (12b)  page offset (12b)
//
//  L1 table : 4096 void**  pointers  (32KB, statically allocated on device)
//  L2 table : 4096 void*   pointers  (32KB, allocated on demand per L1 entry)
//  L3 leaf  : 512B bitmap            (4096 bits, one per page, on demand)

#define L1_ENTRIES  4096
#define L2_ENTRIES  4096
#define L3_BYTES    512      // 4096 bits → one bit per page in this 16MB range

#define PAGE_SHIFT  12
#define L3_SHIFT    12       // bits  12–23
#define L2_SHIFT    24       // bits  24–35
#define L1_SHIFT    36       // bits  36–47
#define IDX_MASK    0xFFFULL // 12-bit mask

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
    // shadow_l1 points to the L1 table (array of 4096 void** pointers)
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
        uint32_t l1_idx      = (addr >> L1_SHIFT) & IDX_MASK;
        uint32_t l2_idx      = (addr >> L2_SHIFT) & IDX_MASK;
        uint32_t l3_offset   = (addr >> L3_SHIFT) & IDX_MASK; // bit index within L3 leaf

        if (addr % (1 << 20) == 0) // Throttle printf spam
            printf("[MarkAccess] addr=%p  l1=%u l2=%u l3=%u\n",
                   (void*)addr, l1_idx, l2_idx, l3_offset);
        
        LOG("[MarkAccess] addr=%p  l1=%u l2=%u l3=%u\n",
            (void*)addr, l1_idx, l2_idx, l3_offset);
        LOG("[MarkAccess] shadow_l1[l1_idx]=%p\n", shadow_l1[l1_idx]);
        // ── Level 1 → Level 2 (demand allocate L2 table) ─────────────────────
        if (shadow_l1[l1_idx] == nullptr) {
            // LOG("[MarkAccess] trying to allocate L2 table for L1 index %u\n", l1_idx);
            //return;
            void** new_l2 = (void**)malloc(L2_ENTRIES * sizeof(void*));
            LOG("[MarkAccess] malloc for L2 table returned %p\n", new_l2);
            if (!new_l2) return;
            // Zero-init
            // LOG("[MarkAccess] zero-initialising new L2 table at %p\n", new_l2);
            // Atomic CAS: only one thread wins, losers free their allocation
            unsigned long long old = atomicCAS(
                (unsigned long long*)&shadow_l1[l1_idx],
                0ULL,
                (unsigned long long)new_l2
            );
            // Clearing in case of race condition
            if (old != 0ULL){
                free(new_l2);  // another thread already inserted
            }
            else{
                LOG("[MarkAccess] allocated L2 table for L1 index %u at %p\n", l1_idx, new_l2);
                memset(new_l2, 0, L2_ENTRIES * sizeof(void*)); // zero-init the new L2 table
                int flag = 0;
                for(int i = 0; i != 512; i++)
                {
                    if (((unsigned long long*)new_l2)[i] != 0ULL){
                        flag = 1;
                        LOG("[MarkAccess] error: new L2 table at %p not zero-initialised! Entry %d is %p\n",
                            new_l2, i, ((void**)new_l2)[i]);
                        break;
                    }
                }
                if (!flag){
                    LOG("[MarkAccess] new L2 table at %p successfully zero-initialised\n", new_l2);
                }
            }
        }

        void** l2_table = (void**)shadow_l1[l1_idx];

        // ── Level 2 → Level 3 (demand allocate L3 bitmap leaf) ───────────────
        if (l2_table[l2_idx] == nullptr) {
            void* new_l3 = malloc(L3_BYTES);
            if (!new_l3) return;

            // Zero-init
            unsigned long long* init = (unsigned long long*)new_l3;
            for (int i = 0; i < (L3_BYTES / 8); i++)
                init[i] = 0ULL;

            unsigned long long old = atomicCAS(
                (unsigned long long*)&l2_table[l2_idx],
                0ULL,
                (unsigned long long)new_l3
            );
            // Clearing in case of race condition
            if (old != 0ULL)
                free(new_l3);
            else
                LOG("[MarkAccess] allocated L3 leaf for L1 index %u, L2 index %u\n",
                    l1_idx, l2_idx);
        }

        // ── Level 3: flip the bit for this page ──────────────────────────────
        unsigned long long* l3_bitmap = (unsigned long long*)l2_table[l2_idx];
        if (l3_bitmap) {
            LOG("[MarkAccess] marking page accessed in L3 bitmap for L1 index %u, L2 index %u, bit %u\n",
                l1_idx, l2_idx, l3_offset);
            unsigned long long* word = &l3_bitmap[l3_offset / 64];
            unsigned long long  mask = 1ULL << (l3_offset % 64);
            atomicOr(word, mask);
        }
        else{
            printf("[MarkAccess] error: L3 bitmap not allocated for L1 index %u, L2 index %u\n",
                   l1_idx, l2_idx);
        }
    }
}


__global__ void copy_l2_to_host(void*** l1, int l1_idx, void** out, int n)
{
    int tid = threadIdx.x;
    LOG("[copy_l2_to_host] thread %d copying L2 entry %d from L1 index %d\n",
        tid, tid, l1_idx);
    if (tid < n)
        out[tid] = ((void**)l1[l1_idx])[tid];
    LOG("[copy_l2_to_host] thread %d done\n", tid);
}

__global__ void copy_l3_to_host(void*** l1, int l1_idx, int l2_idx,
                                 unsigned long long* out, int n)
{
    int tid = threadIdx.x;
    if (tid < n)
        out[tid] = ((unsigned long long*)((void**)l1[l1_idx])[l2_idx])[tid];
}

// ── Host: initialise the L1 table ─────────────────────────────────────────────
void init_tracking(void**** d_l1_ptr)
{
    // Give the device heap enough room for on-demand L2/L3 allocations
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024 * 1024 * 2));

    // Allocate the L1 table
    CUDA_CHECK(cudaMalloc(d_l1_ptr, L1_ENTRIES * sizeof(void**)));
    CUDA_CHECK(cudaMemset(*d_l1_ptr, 0, L1_ENTRIES * sizeof(void**)));

    // Point shadow_l1 at it
    void*** temp = *d_l1_ptr;
    CUDA_CHECK(cudaMemcpyToSymbol(shadow_l1, &temp, sizeof(void***)));

    // Verify
    void*** readback = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, shadow_l1, sizeof(void***)));
    printf("[init_tracking] shadow_l1 → %p (readback %p)\n", temp, readback);
}

// ── Host: walk the page table and dump accessed addresses ────────────────────

void export_log(void*** d_l1, const char* filename)
{
    std::ofstream f(filename);

    printf("[export_log] writing to %s\n", filename);

    // L1 is cudaMalloc'd so this is fine
    std::vector<void**> h_l1(L1_ENTRIES);
    CUDA_CHECK(cudaMemcpy(h_l1.data(), d_l1,
                          L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost));

    // Staging buffers in cudaMalloc'd memory
    void**               d_l2_stage;
    unsigned long long*  d_l3_stage;
    CUDA_CHECK(cudaMalloc(&d_l2_stage, L2_ENTRIES * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_l3_stage, L3_BYTES));

    std::vector<void*>          h_l2(L2_ENTRIES);
    std::vector<unsigned long long> bitmap(L3_BYTES / 8);

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (!h_l1[i]) continue;

        // Copy L2 table out via kernel into staging buffer
        LOG("[export_log] L1[%d] → %p\n", i, h_l1[i]);
        DBG;
        copy_l2_to_host<<<1, L2_ENTRIES>>>(d_l1, i, d_l2_stage, L2_ENTRIES);
        DBG;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_l2.data(), d_l2_stage,
                              L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost));
        LOG("[export_log]   L2 table copied to host\n");
        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;

            // Copy L3 bitmap out via kernel into staging buffer
            LOG("[export_log]   L2[%d] → %p\n", j, h_l2[j]);
            copy_l3_to_host<<<1, L3_BYTES/8>>>(d_l1, i, j, d_l3_stage, L3_BYTES/8);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(bitmap.data(), d_l3_stage,
                                  L3_BYTES, cudaMemcpyDeviceToHost));

            for (int w = 0; w < (int)bitmap.size(); w++) {
                if (!bitmap[w]) continue;
                for (int b = 0; b < 64; b++) {
                    if (bitmap[w] & (1ULL << b)) {
                        uint64_t l3_offset = (uint64_t)w * 64 + b;
                        uint64_t vaddr = ((uint64_t)i << L1_SHIFT)
                                       | ((uint64_t)j << L2_SHIFT)
                                       | (l3_offset   << L3_SHIFT);
                        f << std::hex << "0x" << vaddr << "\n";
                    }
                }
            }
        }
    }

    cudaFree(d_l2_stage);
    cudaFree(d_l3_stage);
}
/* Alternative version without staging buffers and extra kernels, but with more host-device memcpy calls:
void export_log(void*** d_l1, const char* filename)
{
    std::cout << "[export_log] writing to " << filename << std::endl;
    std::ofstream f(filename);

    // Copy L1 to host
    std::vector<void**> h_l1(L1_ENTRIES);
    CUDA_CHECK(cudaMemcpy(h_l1.data(), d_l1,
                          L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost));

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (!h_l1[i]) continue;

        // Copy this L2 table to host
        printf("[export_log] L1[%d] → %p\n", i, h_l1[i]);
        std::vector<void*> h_l2(L2_ENTRIES);
        CUDA_CHECK(cudaMemcpy(h_l2.data(), h_l1[i],
                              L2_ENTRIES * sizeof(void*), cudaMemcpyDefault));

        for (int j = 0; j < L2_ENTRIES; j++) {
            if (!h_l2[j]) continue;
            // Copy this L3 bitmap leaf to host
            printf("[export_log]   L2[%d] → %p\n", j, h_l2[j]);
            std::vector<uint64_t> bitmap(L3_BYTES / 8);
            CUDA_CHECK(cudaMemcpy(bitmap.data(), h_l2[j],
                                  L3_BYTES, cudaMemcpyDefault));

            for (int w = 0; w < (int)bitmap.size(); w++) {
                if (!bitmap[w]) continue;
                for (int b = 0; b < 64; b++) {
                    if (bitmap[w] & (1ULL << b)) {
                        uint64_t l3_offset = (uint64_t)w * 64 + b;
                        // Reconstruct VA from indices
                        uint64_t vaddr = ((uint64_t)i << L1_SHIFT)
                                       | ((uint64_t)j << L2_SHIFT)
                                       | (l3_offset   << L3_SHIFT);
                        f << std::hex << "0x" << vaddr << "\n";
                    }
                }
            }
        }
    }
}
*/

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

//    myKernel<<<1, 5>>>();
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
