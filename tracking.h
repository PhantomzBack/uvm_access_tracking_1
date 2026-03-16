// #include "common.h" // TODO: Do we need? 
#pragma once

extern void init_tracking(void**** d_l1_ptr); // function to initialize the tracking data structure on the device
void export_log(void*** d_l1, const char* filename); // function to export the tracking log from the device to a human-readable text file
void export_binary(void*** d_l1, const char* filename); // function to export the tracking log from the device to a binary file in the specified format


#ifdef __CUDACC__
extern "C"{
    extern __device__ void*** shadow_l1; // device pointer to the L1 page table (array of pointers to L2 tables)
    extern __device__ void MarkAccess(uintptr_t addr); 
}
#endif
