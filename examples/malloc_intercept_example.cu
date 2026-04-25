// ──────────────────────────────────────────────────────────────────────────────
// Example: Using UVM Tracking with Malloc Intercept
//
// Demonstrates the refactored tracking system with LD_PRELOAD pre-allocation.
// This example shows:
//   1. Initialize shadow page table
//   2. Link with malloc_intercept wrapper
//   3. Allocate UVM memory (pre-allocation happens automatically)
//   4. Run tracking kernels
//   5. Export results
//
// Compilation:
//   nvcc -I./include examples/malloc_intercept_example.cu libMarkAccess.so \
//        -o malloc_intercept_example -lstdc++
//
// Execution (without intercept, old behavior - device malloc):
//   ./malloc_intercept_example
//
// Execution (with intercept, new behavior - host pre-allocation):
//   LD_PRELOAD=./libMallocIntercept.so ./malloc_intercept_example
//
// Expected output:
//   Both should produce identical pagelog outputs (pagelog.bin)
// ──────────────────────────────────────────────────────────────────────────────

#include <stdio.h>
#include <cuda_runtime.h>
#include "tracking.h"

// Simple test kernel: access memory in a pattern that exercises the page table
__global__ void test_access_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Sequential access
        float val = data[idx];
        data[idx] = val + 1.0f;
        
        // Call tracking function (injected by LLVM pass)
        // In a real scenario, this would be automatically inserted
        MarkAccess((uintptr_t)&data[idx]);
    }
}

// Simpler kernel that does explicit tracking
__global__ void test_stride_kernel(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int offset = idx * stride;
        if (offset < n) {
            data[offset] = data[offset] + 1.0f;
        }
    }
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("UVM Tracking with Malloc Intercept\n");
    printf("========================================\n\n");
    
    // ── Step 1: Initialize shadow page table ──────────────────────────────────
    printf("[1/5] Initializing shadow page table...\n");
    void*** d_l1;
    init_tracking(&d_l1);
    printf("      ✓ Shadow L1 table allocated at %p\n\n", d_l1);
    
    // ── Step 2: Initialize malloc intercept wrapper ────────────────────────────
    printf("[2/5] Initializing malloc_intercept wrapper...\n");
    // Since we used cudaMalloc for d_l1, it's on device memory
    init_tracking_with_malloc_intercept(d_l1, true);  // is_device_memory=true
    printf("      ✓ Wrapper initialized\n\n");
    
    // ── Step 3: Allocate UVM memory ──────────────────────────────────────────
    printf("[3/5] Allocating UVM memory (pre-allocation happens automatically)...\n");
    int n = 1024 * 1024;  // 1M floats = 4 MB
    float* d_data = nullptr;
    
    printf("      Allocating %.1f MB...\n", n * sizeof(float) / 1e6);
    cudaError_t err = cudaMallocManaged(&d_data, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "      ERROR: cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("      ✓ Data allocated at %p\n", d_data);
    printf("      ✓ Pre-allocated L1/L2/L3 entries for address range\n\n");
    
    // ── Step 4: Run test kernels ────────────────────────────────────────────
    printf("[4/5] Running test kernels...\n");
    
    // Initialize data
    printf("      Initializing data on host...\n");
    for (int i = 0; i < n; i++) {
        d_data[i] = (float)i;
    }
    
    // Simple sequential access
    printf("      Running sequential access kernel...\n");
    test_access_kernel<<<(n + 255) / 256, 256>>>(d_data, n / 2);
    cudaDeviceSynchronize();
    
    // Stride access (tests different pages)
    printf("      Running stride access kernel...\n");
    int stride = 4096 / sizeof(float);  // ~1 page per stride
    test_stride_kernel<<<(n + 255) / 256, 256>>>(d_data, n / 4, stride);
    cudaDeviceSynchronize();
    
    printf("      ✓ Kernels completed\n\n");
    
    // ── Step 5: Export results ──────────────────────────────────────────────
    printf("[5/5] Exporting page access log...\n");
    export_binary(d_l1, "pagelog.bin");
    export_log(d_l1, "pagelog.txt");
    printf("      ✓ Binary log: pagelog.bin\n");
    printf("      ✓ Text log: pagelog.txt\n\n");
    
    // ── Cleanup ──────────────────────────────────────────────────────────────
    printf("Cleaning up...\n");
    cudaFree(d_data);
    cudaFree(d_l1);
    printf("✓ Done!\n\n");
    
    // ── Stats ────────────────────────────────────────────────────────────────
    printf("========================================\n");
    printf("Summary:\n");
    printf("  Data size: %.1f MB\n", n * sizeof(float) / 1e6);
    printf("  Allocation: %p\n", d_data);
    printf("  Output files:\n");
    printf("    - pagelog.bin (binary format)\n");
    printf("    - pagelog.txt (human-readable)\n");
    printf("\n");
    printf("To compare with old behavior:\n");
    printf("  1. Run WITHOUT LD_PRELOAD: ./malloc_intercept_example\n");
    printf("  2. Run WITH LD_PRELOAD:    LD_PRELOAD=./libMallocIntercept.so ./malloc_intercept_example\n");
    printf("  3. Compare outputs: diff pagelog.txt (should be identical)\n");
    printf("========================================\n");
    
    return 0;
}
