// #include "common.h" // TODO: Do we need? 
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef TRACKING_ENABLED
extern void init_tracking(void**** d_l1_ptr); // function to initialize the tracking data structure on the device
void export_log(void*** d_l1, const char* filename); // function to export the tracking log from the device to a human-readable text file
void export_binary(void*** d_l1, const char* filename); // function to export the tracking log from the device to a binary file in the specified format

extern "C" void uvm_tracking_preload_range(uintptr_t start, size_t size);
extern "C" void uvm_tracking_drop_range(uintptr_t start, size_t size);
extern "C" void** g_uvm_shadow_l1;

// Runtime toggles (used by control thread)
extern "C" void uvm_tracking_set_enabled(int v);
extern "C" void uvm_tracking_set_skip_on_miss(int v);

// Background control thread (Unix domain socket server)
extern "C" void uvm_start_control_thread(void);
extern "C" void uvm_stop_control_thread(void);

extern "C" __device__ void*** shadow_l1;
extern "C" __device__ __constant__ int g_tracking_enabled;
extern "C" __device__ __constant__ int g_skip_on_miss;
extern "C" __device__ void MarkAccess(uintptr_t addr);
extern "C" __device__ void BatchMarkAccess(uintptr_t base_addr, int64_t stride, uint64_t count);
#endif

// ── Temporal Tracking (Approach A: In-Kernel Hardware Timestamps) ─────────────
#define MAX_TEMPORAL_ENTRIES 65536

#define UVM_TEMPORAL_FLAG_LOOP_ITER    0x00
#define UVM_TEMPORAL_FLAG_KERNEL_ENTRY 0x01
#define UVM_TEMPORAL_FLAG_KERNEL_EXIT  0x02

struct UvmTripwireMailbox {
    volatile uint32_t seq_num;       // Increments on each tripwire fire
    volatile uint32_t kernel_id;     // Kernel hash
    volatile uint32_t milestone_id;  // Outer loop iteration / milestone
    volatile uint32_t flags;         // UVM_TEMPORAL_FLAG_*
    volatile uint64_t gpu_ts_ns;     // %globaltimer timestamp
    volatile uint64_t base_addr;     // Base address touched in loop
    volatile uint32_t size_bytes;    // Step/tile bytes
};

struct TemporalMilestoneEntry {
    uint32_t kernel_id;      // Unique kernel hash/ID
    uint32_t milestone_id;   // Milestone index or outer loop iteration
    uint64_t timestamp_ns;   // Hardware clock (%globaltimer) in ns
    uint64_t base_addr;      // Pointer address touched (or 0)
    uint32_t size_bytes;     // Tile/footprint size in bytes (or 0)
    uint16_t block_id;       // Thread/block ID that logged
    uint16_t flags;          // UVM_TEMPORAL_FLAG_*
};

#ifdef __CUDACC__
extern "C" __device__ void uvm_record_temporal_milestone(
    uint32_t kernel_id,
    uint32_t milestone_id,
    uintptr_t base_addr,
    uint32_t size_bytes,
    uint32_t flags
);
#endif

#ifdef __cplusplus
extern "C" {
#endif
void uvm_temporal_reset(void);
uint32_t uvm_temporal_get_count(void);
void uvm_temporal_set_enabled(int enabled);
void uvm_temporal_export_json(const char *filename);
void uvm_temporal_export_csv(const char *filename);
UvmTripwireMailbox* uvm_get_tripwire_mailbox(void);
uint32_t uvm_temporal_copy_entries(TemporalMilestoneEntry *out_entries, uint32_t max_entries);
#ifdef __cplusplus
}
#endif

