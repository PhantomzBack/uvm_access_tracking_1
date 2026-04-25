// ═══════════════════════════════════════════════════════════════════════════════
// Long-running page-touch test for control-thread verification.
//
// Allocates a 100 MiB managed array and touches every page in a tight loop
// for ~15 s (or until killed).  Between kernel launches it sleeps 100 ms so
// the control thread has time to toggle tracking, snapshot, or clear state.
//
// Build:
//   clang++-20 -x cuda --cuda-gpu-arch=sm_89 -fgpu-rdc -O2 -I./include \
//       -DTRACKING_ENABLED -fpass-plugin=./build/UvmTrackingPass.so \
//       examples/long_running_test.cu libMarkAccess.cu uvm_control_thread.cu \
//       --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64 -lcudart \
//       -o build/long_running_test
//
// Run:
//   LD_PRELOAD=./libMallocIntercept.so ./build/long_running_test
// ═══════════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <thread>
#include "tracking.h"

// Touch every float element → one atomic write per 4 KB page (coalesced).
__global__ void touch_all_pages(float* data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride)
        data[i] = data[i] + 1.0f;
}

int main(int argc, char** argv)
{
    const size_t BYTES       = 100ULL * 1024 * 1024;   // 100 MiB
    const size_t N_FLOATS    = BYTES / sizeof(float);   // 25 600 000
    const int    DURATION_S  = (argc > 1) ? atoi(argv[1]) : 15;
    const int    SLEEP_MS    = 100;

    void*** d_l1 = nullptr;
    init_tracking(&d_l1);

    printf("[long_running_test] allocating %.1f MiB managed memory\n",
           BYTES / (1024.0 * 1024.0));

    float* d_data = nullptr;
    cudaMallocManaged(&d_data, BYTES, cudaMemAttachGlobal);
    cudaMemset(d_data, 0, BYTES);

    printf("[long_running_test] running for %d seconds (sleep %d ms between iters)\n",
           DURATION_S, SLEEP_MS);
    printf("[long_running_test] PID = %d\n", getpid());
    fflush(stdout);

    int threads = 256;
    int blocks  = (N_FLOATS + threads - 1) / threads;
    blocks      = (blocks > 65535) ? 65535 : blocks;

    auto t0 = std::chrono::steady_clock::now();
    int iter = 0;

    while (true) {
        touch_all_pages<<<blocks, threads>>>(d_data, N_FLOATS);
        cudaDeviceSynchronize();

        ++iter;
        auto elapsed = std::chrono::steady_clock::now() - t0;
        int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

        if (elapsed_s >= DURATION_S)
            break;

        // Print progress every second
        if (iter % 10 == 0) {
            printf("[long_running_test] iter=%d  elapsed=%ds\n", iter, elapsed_s);
            fflush(stdout);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS));
    }

    printf("[long_running_test] finished %d iterations, exporting pagelog\n", iter);
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_data);
    return 0;
}
