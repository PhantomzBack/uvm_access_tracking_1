#pragma once
#ifndef UVM_POLICY_H
#define UVM_POLICY_H

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <utility>
#include <string>

// ── Tracking Strategies per Kernel ──────────────────────────────────────────
enum class TrackingStrategy : uint8_t {
    SKIP_TRACKING  = 0,  // Do not track at all (always run clean, 0% overhead)
    PREDICT_HOST   = 1,  // Analytically compute footprint on host from args, run clean kernel
    INSTRUMENT_GPU = 2,  // Data-dependent / non-affine: run instrumented GPU kernel
    AUTO_DETECT    = 3   // Default: use PREDICT_HOST if predictor exists, else INSTRUMENT_GPU
};

// ── Predictor Callback ──────────────────────────────────────────────────────
// Computes list of [start_addr, size_bytes] pairs from kernelParams / args
typedef void (*FootprintPredictorFunc)(
    void **args,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    std::vector<std::pair<uintptr_t, size_t>>& out_ranges
);

// ── Kernel Policy Descriptor ────────────────────────────────────────────────
struct KernelPolicyDescriptor {
    std::string kernel_name;
    const void* clean_func = nullptr;
    const void* instrumented_func = nullptr;
    TrackingStrategy strategy = TrackingStrategy::AUTO_DETECT;
    bool enabled = true;
    FootprintPredictorFunc predictor_fn = nullptr;
    uint64_t launch_count = 0;
};

// ── External Runtime APIs ───────────────────────────────────────────────────
#ifdef __cplusplus
extern "C" {
#endif

// Mark a predicted address range in the shadow page table
void uvm_mark_predicted_range(uintptr_t start, size_t size, void* stream);

// Set policy for a kernel symbol by name
void uvm_set_kernel_strategy(const char* kernel_name, TrackingStrategy strategy);

#ifdef __cplusplus
}
#endif

#endif // UVM_POLICY_H
