#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include "tracking.h" // UVM tracking API

#define PAGE_SIZE 4096
#define NUM_PAGES_MANAGED 1000
#define NUM_PAGES_DEVICE 1000

// Kernel touches exactly one byte per page to trigger the tracking instrumentation
__global__ void touch_pages(char* managed_ptr, char* device_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < NUM_PAGES_MANAGED) {
        managed_ptr[tid * PAGE_SIZE] = 1;
    }
    if (tid < NUM_PAGES_DEVICE) {
        device_ptr[tid * PAGE_SIZE] = 1;
    }
}

int main(int argc, char** argv) {
    // Explicitly initialize tracking (spawns control thread)
#ifdef TRACKING_ENABLED
    void*** d_l1;
    init_tracking(&d_l1);
#endif


    char *d_managed = nullptr;
    char *d_device = nullptr;

    cudaMallocManaged(&d_managed, NUM_PAGES_MANAGED * PAGE_SIZE);
    cudaMalloc(&d_device, NUM_PAGES_DEVICE * PAGE_SIZE);

    // Signal to the Python script that allocations are done and the socket is ready
    printf("READY\n");
    fflush(stdout);

    // Wait for the Python script to send a newline after configuring the socket
    char buf[10];
    fgets(buf, sizeof(buf), stdin);

    // Launch the kernel
    int threads = 256;
    int max_pages = (NUM_PAGES_MANAGED > NUM_PAGES_DEVICE) ? NUM_PAGES_MANAGED : NUM_PAGES_DEVICE;
    int blocks = (max_pages + threads - 1) / threads;

    touch_pages<<<blocks, threads>>>(d_managed, d_device);
    cudaDeviceSynchronize();

    // Export the pagelog so the python script can count the pages
#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_l1);
#endif

    cudaFree(d_managed);
    cudaFree(d_device);

    return 0;
}