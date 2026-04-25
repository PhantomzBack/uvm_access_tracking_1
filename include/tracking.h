// #include "common.h" // TODO: Do we need? 
#pragma once

#include <stdint.h>
#include <stddef.h>

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

//#ifdef __CUDACC__
extern "C" __device__ void*** shadow_l1;
extern "C" __device__ __constant__ int g_tracking_enabled;
extern "C" __device__ __constant__ int g_skip_on_miss;
extern "C" __device__ void MarkAccess(uintptr_t addr);
extern "C" __device__ void BatchMarkAccess(uintptr_t base_addr, int64_t stride, uint64_t count);

//#endif
