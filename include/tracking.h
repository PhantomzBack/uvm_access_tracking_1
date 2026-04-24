#pragma once

#include <stdint.h>
#include <stddef.h>

// ── Core tracking lifecycle ───────────────────────────────────────────────────
extern void init_tracking(void**** d_l1_ptr);
void export_log(void*** d_l1, const char* filename);
void export_binary(void*** d_l1, const char* filename);

// ── LD_PRELOAD integration ────────────────────────────────────────────────────
extern "C" void uvm_tracking_preload_range(uintptr_t start, size_t size);

// ── Live control API ──────────────────────────────────────────────────────────
extern "C" void uvm_tracking_enable(void);
extern "C" void uvm_tracking_disable(void);
extern "C" void uvm_tracking_set_mode(int skip_on_miss);
extern "C" const char* uvm_tracking_get_socket_path(void);

// ── Device-side symbols ───────────────────────────────────────────────────────
extern "C" __device__ void*** shadow_l1;
extern "C" __device__ void MarkAccess(uintptr_t addr);
extern "C" __device__ void BatchMarkAccess(uintptr_t base_addr, int64_t stride, uint64_t count);
