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
    AUTO_DETECT    = 3   // Default: use PREDICT_HOST if metadata exists, else INSTRUMENT_GPU
};

// ── Bytecode Instruction Set for Memory Footprint Expressions ───────────────
enum class UvmOpcode : uint8_t {
    OP_END                = 0x00,
    OP_CONST_U64          = 0x01, // [uint64_t val]
    OP_CONST_U32          = 0x02, // [uint32_t val]

    // Push kernel arguments
    OP_ARG_U8             = 0x10, // [uint8_t idx]
    OP_ARG_I8             = 0x11, // [uint8_t idx]
    OP_ARG_U16            = 0x12, // [uint8_t idx]
    OP_ARG_I16            = 0x13, // [uint8_t idx]
    OP_ARG_U32            = 0x14, // [uint8_t idx]
    OP_ARG_I32            = 0x15, // [uint8_t idx]
    OP_ARG_U64            = 0x16, // [uint8_t idx]
    OP_ARG_I64            = 0x17, // [uint8_t idx]
    OP_ARG_PTR            = 0x18, // [uint8_t idx]

    // Push Grid / Block dimensions
    OP_GRID_DIM_X         = 0x20,
    OP_GRID_DIM_Y         = 0x21,
    OP_GRID_DIM_Z         = 0x22,
    OP_BLOCK_DIM_X        = 0x23,
    OP_BLOCK_DIM_Y        = 0x24,
    OP_BLOCK_DIM_Z        = 0x25,

    // Arithmetic
    OP_ADD                = 0x30,
    OP_SUB                = 0x31,
    OP_MUL                = 0x32,
    OP_DIV                = 0x33,
    OP_SHL                = 0x34,
    OP_MAX                = 0x35,
    OP_MIN                = 0x36,

    // Emission
    OP_EMIT_RANGE            = 0x40, // pop size, pop base
    OP_EMIT_ARG_RANGE        = 0x41, // [uint8_t ptr_arg_idx], pop size
    OP_EMIT_STRUCT_ARG_RANGE = 0x42, // [uint8_t struct_arg_idx, uint8_t byte_offset], pop size

    // Active allocation scan fallback (for templated/aggregate kernel args)
    OP_SCAN_ACTIVE_ALLOCS    = 0x50  // [uint8_t ptr_arg_idx] (0xFF = scan args[0])
};

// ── Binary Metadata Format for ELF / Module ─────────────────────────────────
#define UVM_METADATA_MAGIC 0x55564D31 // "UVM1"

#pragma pack(push, 1)
struct UvmKernelMetadataHeader {
    uint32_t magic;         // UVM_METADATA_MAGIC
    uint32_t version;       // 1
    uint32_t num_kernels;   // number of kernels described
    uint32_t table_size;    // total bytes of entire table
};

struct UvmKernelEntryHeader {
    uint32_t name_len;      // length of mangled kernel name (excluding null terminator)
    uint32_t bytecode_len;  // length of bytecode stream
    uint16_t num_ranges;    // expected number of ranges
    uint16_t flags;         // 1 = affine, 2 = requires scan fallback
    // Followed by: char name[name_len + 1], then uint8_t bytecode[bytecode_len]
};
#pragma pack(pop)

// ── Predictor Callback (Optional programmatic fallback) ─────────────────────
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
    std::vector<uint8_t> bytecode;
    uint64_t launch_count = 0;
    int num_args = -1;
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
