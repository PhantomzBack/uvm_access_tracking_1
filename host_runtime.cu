#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "common.h"
#include "tracking.h"

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

void export_binary(void*** d_l1, const char* filename)
{
    printf("[export_binary] writing to %s\n", filename);

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

    printf("[export_binary] %zu populated leaves\n", leaves.size());

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

    printf("[export_binary] done — %zu bytes written\n", (size_t)off);
}

