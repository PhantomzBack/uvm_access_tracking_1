#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "common.h"
#include "tracking.h"

// Standard page size is 4KB. 
// 4096 bytes / sizeof(int) = 1024 integers per page.
#define PAGE_SIZE_BYTES 4096
#define INTS_PER_PAGE (PAGE_SIZE_BYTES / sizeof(int))

// ── New Kernel: Sequential Page Access ──────────────────────────────────────
/**
 * Each block processes exactly one page.
 * threadIdx.x == 0 performs the sequential write for the entire page.
 */
__global__ void page_sequential_write(int* data, int num_pages)
{
    int page_idx = blockIdx.x;

    if (page_idx < num_pages) {
        // Calculate the starting pointer for this specific page
        int* page_start = data + (page_idx * INTS_PER_PAGE);

        // Perform the sequential operation integer by integer
        for (int i = 0; i < INTS_PER_PAGE; ++i) {
            // Writing the value of the integer relative to the start of the page
            page_start[i] = i;
        }
        
        // Optional: Print confirmation for the first few pages
        if (page_idx < 2) {
            printf("[Kernel] Processed Page %d sequentially (%d integers)\n", page_idx, INTS_PER_PAGE);
        }
    }
}

// ── Existing Test kernels ───────────────────────────────────────────────────
__global__ void myKernel()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[myKernel] Hello from GPU\n");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    // Let's process 10 pages for this test
    int num_pages = 10;
    size_t total_elements = num_pages * INTS_PER_PAGE;
    size_t total_bytes = total_elements * sizeof(int);

    int* d_data;
    // Allocating Managed Memory
    cudaError_t malloc_err = cudaMallocManaged(&d_data, total_bytes);
    if (malloc_err != cudaSuccess) {
        printf("Malloc failed: %s\n", cudaGetErrorString(malloc_err));
        return -1;
    }

    /* Note: I've commented out the tracking functions since they require 
       your specific "tracking.h" and "common.h" implementation logic.
    */
    void*** d_l1;
    init_tracking(&d_l1);

    printf("Launching sequential page writer for %d pages...\n", num_pages);

    // Launch configuration: 1 block per page, 1 thread per block 
    // This ensures that the "sequential" nature is preserved within the block.
    page_sequential_write<<<num_pages, 1>>>(d_data, num_pages);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("[main] sync error: %s\n", cudaGetErrorString(err));
    else
        printf("[main] kernels completed successfully\n");

    // Verification: Check a few values from the first page
    printf("Verification - Page 0, Index 500: %d (Expected: 500)\n", d_data[500]);
    printf("Verification - Page 1, Index 10: %d (Expected: 10)\n", d_data[INTS_PER_PAGE + 10]);

    export_log(d_l1, "access_log.txt");
    export_binary(d_l1, "access_log.bin");

    cudaFree(d_data);
    // cudaFree(d_l1);
    return 0;
}