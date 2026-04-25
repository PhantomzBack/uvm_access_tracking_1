#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Global toggle: whether wrapper pre-populates cudaMallocManaged allocations
extern int g_uvm_preload_managed;

// Start/stop the background control thread (idempotent)
void uvm_start_control_thread(void);
void uvm_stop_control_thread(void);

#ifdef __cplusplus
}
#endif
