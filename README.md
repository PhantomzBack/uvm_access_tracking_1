# LLVM Compiler-Pass For Access Tracking in UVM GPUs

This repository implements a lightweight, compiler-based memory tracking system for Unified Virtual Memory (UVM) on NVIDIA GPUs. Instead of relying on slow runtime sampling or heavily contended device-side memory tracking, this project uses an LLVM pass to statically analyze loop memory access patterns and injects highly optimized, warp-aggregated tracking calls.

## Artifact Directory Structure
The artifact encompasses the compiler pass, runtime library, testing framework, and analysis tools. 

- `UvmTrackingPass.cpp`: The core LLVM pass that analyzes PTX functions and instruments memory accesses statically.
- `libMarkAccess.cu`: Device-side runtime kernel implementing the lock-free 3-level shadow page table and warp-level tracking logic.
- `libMallocIntercept.cpp`: An `LD_PRELOAD` interceptor for CUDA allocation functions to eagerly pre-populate shadow tables.
- `uvm_control_thread.cu` / `uvm_control_thread.h`: Host-side UNIX socket server and device kernels for dynamic tracking control.
- `tests/`: Comprehensive test suite containing microbenchmarks, integration tests, and control thread tests.
- `scripts/`: Utility scripts for benchmarking, log parsing, and analysis (`page_analysis.py`).
- `benchmark.py`: Automation script for executing overhead benchmarks.
- `pagelog_drill.py`: Interactive visualization dashboard for binary access logs.

## Setup Instructions

### CPU
- **Requirements**: Multi-core x86_64 CPU.
- **Details**: 4+ cores recommended to handle LLVM compilation and concurrent control thread execution efficiently.

### Memory Required
- **Requirements**: 16 GB+ RAM recommended.
- **Details**: Required for compiling LLVM passes and backing the host-side allocations for the 3-level shadow page tables during large UVM memory tracking.

### Storage Required
- **Requirements**: ~20 GB free space.
- **Details**: Storage is needed for the LLVM toolchain, CUDA toolkit, and the generated binary logs (`access_log.bin` can grow to several megabytes/gigabytes depending on the workload). No specific disk partition structure is required.

### Extra Hardware Required
- **Requirements**: NVIDIA GPU with UVM support.
- **Details**: Compute Capability 6.0 or higher is recommended (e.g., Pascal, Volta, Ampere, Hopper architectures).

### Linux Kernel Compilation Instructions
- **Details**: A standard Linux distribution kernel is sufficient. **Custom Linux kernel compilation is not required.** The UVM memory management is handled entirely by the proprietary NVIDIA driver, and our tracking logic operates at the compiler and CUDA runtime levels.

### OS
- **Requirements**: Linux (Ubuntu 20.04 or Ubuntu 22.04 LTS recommended).

### Software Dependencies
- **Compilers/Toolchains**: 
  - LLVM 20 (`llvm-20`, `clang++-20`)
  - NVIDIA CUDA Toolkit (installed at `/usr/local/cuda`, e.g., 11.x or 12.x)
  - `cmake` and `make`
- **Python**: Python 3.x
- **Python Packages**: `dash`, `numpy` (used for the visualization dashboard).

## Features/functionalities supported by your implementation

### Compiler Instrumentation
The core of our tracking system relies on an LLVM compiler pass. Rather than using slow, unoptimized memory polling or fault-based tracking, the compiler static analysis is utilized to optimize memory tracking calls directly within the application's PTX code, based on static code analysis.
- **What it achieves**: Minimal runtime overhead by tracking memory accesses inline with the operations.
- **Compile time optimizations**: It utilizes LLVM's Scalar Evolution (SCEV) to perform loop hoisting and batching. When it detects predictable, affine loop access patterns, it pulls the tracking calls out of tight inner loops, replacing them with a single, batched bounding-box tracking call.
- **Modes of operation**: It can operate in a per-element tracking mode (for irregular accesses) or a batched mode (for regular, strided, or contiguous access patterns).

### Pre-run Time Configuration
Our implementation heavily relies on configurations that are applied prior to kernel launch, avoiding any need to recompile the user's application for different tracking behaviors.
- **Environment variables and Wrappers**: Through `LD_PRELOAD`, we intercept `cudaMalloc` and `cudaMallocManaged` calls. This allows the host to eagerly pre-populate the 3-level shadow page table, drastically preventing device-side allocation bottlenecks.
- Users can dynamically supply configuration flags and environment variables to orchestrate these memory allocations entirely before the GPU kernels are engaged.

### Runtime Socket Control Plane
A dedicated host-side background thread handles real-time configuration without pausing the primary execution pipeline.
- **Runtime tweaking**: The UNIX domain socket enables you to dynamically `ENABLE` or `DISABLE` memory tracking at runtime, allowing selective tracking of specific execution phases.
- **Controlling parameters**: You can use socket commands to flush memory tracking snapshots to disk (`SNAPSHOT`), clear existing tracking state (`CLEAR`), and configure other run-time behaviours without ever modifying the application's source code or stopping the execution.

### Evaluation of Features & Test Scenarios

| Feature | Test Scenario & Automation Script | Parameters | Objective | Expected Outcome | Findings / Notes |
|---------|-----------------------------------|------------|-----------|------------------|------------------|
| **Compiler Loop Hoisting & Batching** | Microbenchmarks<br>`python3 run_tests.py microbenchmarks` | Arrays up to N=10M, block sizes 128-512 | Verify memory tracking calls are correctly hoisted out of tight loops via Scalar Evolution. | Accurate page access counts with <5% overhead. | Successfully batches tracking in regular loops. Rare assertion failures possible if loop bounds are extremely non-deterministic, though correctness falls back to per-element tracking. |
| **Warp-Aggregated Tracking** | Integration Tests<br>`python3 run_tests.py integration` | Multiple threads/blocks accessing overlapping pages | Ensure warp-level deduplication prevents excessive atomic contention on the shadow page table. | Consistent pagelogs, reduced global memory bottlenecking. | Dramatic reduction in tracking overhead compared to naive atomics. No deadlocks or crashes observed during standard operation. |
| **Dynamic Control Socket** | Control Thread Tests<br>`python3 run_tests.py control_thread` | Socket commands: `ENABLE`, `DISABLE`, `SNAPSHOT`, `CLEAR` | Toggle tracking at runtime and dump memory snapshots without stopping the application. | Uninterrupted application flow; perfectly captures the instantaneous tracking state. | Socket communication is highly reliable. Dumping massive page tables (GBs) takes proportional I/O time but does not crash. |
| **Eager Page Pre-population** | Preload Tests<br>`python3 run_tests.py thread_modes` | `LD_PRELOAD` enabled vs disabled | Prevent device-side `malloc` bottlenecks by eagerly allocating the shadow page table on the host. | Elimination of runtime latency spikes on first memory access by the GPU. | Successfully eliminates device-side stalls. Requires exactly matching the `LD_PRELOAD` shared library path in the environment. |

## Assumptions and unsupported features

**Assumptions:**
- The target GPU supports Unified Memory (UVM) natively via NVIDIA drivers.
- The instrumented applications allocate memory using standard `cudaMallocManaged` APIs.
- The system has sufficient host memory to back the 3-level shadow page table.
- The modes

**Unsupported Features:**
- Tracking standard host memory (`malloc` / `new`) that is not explicitly managed by CUDA UVM.
- Executing the tracking runtime on non-NVIDIA GPUs (e.g., AMD ROCm/HIP is not supported).
- Multi-GPU unified tracking is currently experimental and may not perfectly synchronize unified states across discrete devices without explicit peer-to-peer copies.

## Getting Started

You can verify the basic functionality of the artifact within a few minutes using a "Hello world"-sized example.

### 1. Build the Artifact
First, compile the LLVM tracking pass. From the repository root:
```bash
cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
make
cd ..
```

### 2. Apply to a "Hello world" Example
We will use a basic CUDA program. Create a file named `hello_uvm.cu`:
```cpp
#include "uvm_control_thread.h"
#include <stdio.h>

__global__ void touch_memory(int *arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    arr[idx] = idx * 2;
}

int main() {
    // 1. Initialize tracking
    void* d_l1;
    init_tracking(&d_l1);

    // 2. Allocate UVM memory and launch kernel
    int *managed_arr;
    cudaMallocManaged(&managed_arr, 4096 * 10); // 10 pages
    touch_memory<<<10, 256>>>(managed_arr);
    cudaDeviceSynchronize();

    // 3. Export the tracked accesses
    export_binary(d_l1, "access_log.bin");
    cudaFree(managed_arr);
    printf("Done!\n");
    return 0;
}
```

### 3. Compile and Run
Compile the code using our compiler pass. Ensure the GPU architecture matches your system (the `nvidia-smi` snippet detects this automatically):
```bash
clang++-20 -O2 -x cuda \
  --cuda-gpu-arch=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.') \
  -fgpu-rdc \
  -fpass-plugin=./build/UvmTrackingPass.so \
  ./hello_uvm.cu ./libMarkAccess.cu \
  --cuda-path=/usr/local/cuda \
  -L/usr/local/cuda/lib64 -lcudart \
  -o hello_uvm
```

> Note: Running with -O2 is imperative since otherwise mem2reg runs after the compiler pass, which is very problematic since you would get many many more store instructions.

Execute the binary:
```bash
./hello_uvm
```
This will run the program and generate an `access_log.bin` file in your current directory, confirming the basic functionality is working.

### Supplying Your Own Inputs
To apply this tracking pass to your own CUDA applications:
1. Include `uvm_control_thread.h` in your main source file.
2. Call `init_tracking(&d_l1);` before your kernel launches.
3. Call `export_binary(d_l1, "access_log.bin");` before your application exits.
4. Compile your application exactly as shown in Step 3, passing your `.cu` files alongside `-fpass-plugin=./build/UvmTrackingPass.so` and `libMarkAccess.cu`.

## Detailed Evaluation

### 1. Overhead Benchmarking
- **Purpose**: Measure the runtime performance overhead imposed by the compiler instrumentation pass compared to a native, uninstrumented baseline execution.
- **How to run**: `python3 benchmark.py --rebuild --mode <MODE>`
    `<MODE>` can be:
    - `no-preload`: Standard tracking where the device kernel coordinates tracking structures.
    - `preload-alloc`: Eagerly allocates shadow tables on the host via `LD_PRELOAD`, can be configured on runtime to disable or enable device side allocating of tracking structures.
    - `preload-only`: Pure host-side preloading without dynamic device-side tracking state instantiation.
- **Estimated runtime**: 3-4 minutes.
- **Expected result**: The script compiles and executes a suite of kernels, outputting the execution times for the Baseline vs. Tracked versions.

- **How to access the result**: The results are printed directly to the terminal's standard output. If configured, it also saves a CSV or markdown summary inside the `perf_results/` directory.

### 2. Access Log Correctness & Page Accuracy
- **Purpose**: Verify that the tracking pass logs the exact 4KB pages touched by memory operations without dropping accesses or generating false positives.
- **How to run**: `python3 run_tests.py`
- **Estimated runtime**: < 2 minutes.
- **Expected result**: All sub-suites (microbenchmarks, integration, etc.) will run and exit cleanly with status code `0`. The parsed binary pagelogs will perfectly match the expected analytical access patterns.
- **How to access the result**: The console output from the test script will indicate `Test passed.` for each testing directory.

### 3. Interactive Pagelog Visualization
- **Purpose**: Demonstrate the ability to visually drill down into the tracked memory access patterns over the kernel execution space.
- **How to run**: 
  ```bash
  python3 -m venv env
  source env/bin/activate
  pip install dash numpy
  python pagelog_drill.py access_log.bin
  ```
- **Estimated runtime**: Instantaneous startup.
- **Expected result**: A local web server initializes and hosts the Dash application with the parsed binary data.
- **How to access the result**: Open a web browser and navigate to `http://127.0.0.1:8050`. You will see interactive plots displaying page access hot-spots.

### 4. Advanced Technical Details & Control Plane

For a detailed breakdown of compilation flags, `benchmark.py` parameters, environment variables for configuration, and the runtime UNIX domain socket control plane (`uvm_control_thread.cu`), please refer to the [Advanced Technical Documentation](#advanced-technical-documentation) section aheaf.

# Advanced Technical Documentation

This document contains deep technical details about the UVM Access Tracking framework, including configuration flags, benchmarking parameters, and runtime control mechanisms.

## Benchmark & Compilation Configuration Flags

To deeply evaluate the framework, `benchmark.py` provides numerous command-line arguments to customize the benchmarking runs. Under the hood, it also interacts with several C++ macro flags during compilation.

### `benchmark.py` Execution Flags
- `[output_file]`: Positional argument for the output markdown filename (default: `results.md`).
- `--rebuild`: Forces a clean rebuild of all standard and instrumented binaries.
- `--generate-baseline`: Generates baseline binary pagelogs used for correctness checking.
- `--check`: Runs pagelog correctness checks after benchmarks by comparing against the baselines.
- `--preload`: Utilizes the `LD_PRELOAD` wrapper (`libMallocIntercept.so`) for eager malloc interception.
- `--mode <no-preload|preload-alloc|preload-only>`: Determines the tracking mode for the compiled binaries (default: `no-preload`). This directly sets the `UVM_TRACKING_MODE` compilation macro.
- `--iterations` / `-n <int>`: Number of iterations to run each benchmark for variance smoothing (default: 1).
- `--metric <mean|median|min>`: The statistical metric used to compute overhead across multiple iterations (default: `mean`).
- `--diff <FILE1> <FILE2>`: Compares two result markdown files and prints the differences in tracking overhead.

### Compilation Flags (User Applications & `UvmTrackingPass`)
When compiling your CUDA code alongside the `UvmTrackingPass.so` plugin, the following C++ macro flags are supported:
- `-DTRACKING_ENABLED`: Enables the tracking runtime logic (including the background control socket thread and device kernel initialization).
- `-DUVM_TRACKING_MODE=<0|1|2>`: Dictates how the tracking framework initializes memory:
  - `0` (No Preload): Standard tracking where the device kernel coordinates tracking structures.
  - `1` (Preload-Alloc): Eagerly allocates shadow tables on the host via `LD_PRELOAD`.
  - `2` (Preload-Only): Pure host-side preloading without dynamic device-side tracking state instantiation.

### Pass Build Flags
- `-DUVM_DEBUG`: Passed during the CMake build of the `UvmTrackingPass.so` LLVM plugin itself to enable verbose LLVM `errs()` debugging outputs when running the static analysis over PTX code.

### `build_and_time.py` — Simplified Compilation Harness

For convenience, `build_and_time.py` provides a streamlined Python interface for compiling and benchmarking CUDA kernels with the tracking pass. This tool automates the compilation of both normal (uninstrumented) and instrumented binaries, measures their execution times, and computes the tracking overhead. It also supports batch compilation across multiple tracking modes in a single invocation.

#### Usage

```
usage: build_and_time.py [-h] [--mode MODE] [--run] [--normal-only]
                         [--instrumented-only] [--force] [--timeout SECONDS] [--no-control-thread]
                         [--extra-flags FLAGS] [--dry-run]
                         SOURCE

Compile and optionally benchmark CUDA kernels with UVM tracking.

positional arguments:
  SOURCE                Path to CUDA source file (.cu)

options:
  -h, --help            show this help message and exit
  --mode MODE           Tracking mode (default: no-preload). Can be a single mode 
                        (no-preload, preload-alloc, preload-only) or index (0, 1, 2),
                        or multiple modes in braces (e.g., {no-preload,preload-alloc} or {0,2})
  --run                 Compile both versions and run benchmarks
  --normal-only         Skip instrumented compilation
  --instrumented-only   Skip normal compilation
  --force               Rebuild even if outputs are fresh
  --timeout SECONDS     Timeout for kernel runs (default: 60)
  --no-control-thread   Disable control thread (adds -DUVM_NO_CONTROL_THREAD)
  --extra-flags FLAGS   Additional compiler flags (space-separated)
  --dry-run             Print compilation commands without executing
```

#### Mode Specification

The `--mode` parameter is flexible:

- **Single mode by name**: `--mode no-preload`, `--mode preload-alloc`, `--mode preload-only`
- **Single mode by index**: `--mode 0` (no-preload), `--mode 1` (preload-alloc), `--mode 2` (preload-only)
- **Multiple modes**: `--mode "{0,1,2}"` or `--mode "{no-preload,preload-alloc,preload-only}"`
- **Mixed syntax**: `--mode "{0,preload-alloc,2}"` (indices and names can be mixed)

#### Examples

**Basic compilation in default (no-preload) mode:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu
```

**Compile using mode index:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --mode 1
```

**Compile and run benchmarks with execution timing:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --run
```

**Compile all three modes in one invocation:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --mode "{0,1,2}"
```

**Compile multiple modes with execution and overhead measurement:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --mode "{0,1,2}" --run
```

**Preload-only mode with custom include paths:**
```bash
python3 build_and_time.py examples/sgemm/sgemm_cutlass.cu \
  --mode preload-only \
  --no-control-thread \
  --extra-flags="-I./examples/sgemm -I/YOUR/CUTLASS/INCLUDE/PATH"
```

**Dry run to inspect compilation commands for multiple modes:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --mode "{no-preload,preload-only}" --dry-run
```

**Force rebuild with specific tracking mode:**
```bash
python3 build_and_time.py examples/benchmark_kernel.cu --force --mode preload-alloc
```


## Runtime Control Plane (`uvm_control_thread.cu`)

The `uvm_control_thread.cu` implementation spawns a background thread on the host alongside your instrumented application. This thread operates independently of the GPU kernels and listens on a UNIX domain socket (`/tmp/uvm-ctl.<pid>`) for real-time commands.

### Environment Variables
The control thread reads the following environment variable upon initialization:
- `UVM_PRELOAD_MANAGED`: When set to `0`, disables eager pre-population of managed memory allocations even if the `LD_PRELOAD` wrapper is present. By default, or if unset, it defaults to `1` (enabled).

### Supported Socket Commands
You can interact with the control thread at runtime using tools like `nc` (netcat), `socat`, or custom Python scripts. The following commands are supported (commands should end with a newline):

- `ENABLE`: Turns on memory tracking globally (`uvm_tracking_set_enabled(1)`).
- `DISABLE`: Turns off memory tracking globally (`uvm_tracking_set_enabled(0)`).
- `MODE SKIP`: Sets tracking to skip unmapped memory pages on a cache miss, effectively avoiding device-side allocations (`uvm_tracking_set_skip_on_miss(1)`).
- `MODE ALLOC`: Configures the tracking runtime to dynamically allocate missing L3 page tables on the device (`uvm_tracking_set_skip_on_miss(0)`).
- `PRELOAD_MANAGED <0|1>`: Dynamically sets the `g_uvm_preload_managed` flag at runtime to intercept future `cudaMallocManaged` calls.
- `SNAPSHOT <absolute_path>`: Flushes the current multi-level shadow page table tracking state into a binary pagelog file at the specified path.
- `DROP_RANGE <hex-addr> <len>`: Instructs the runtime to drop the tracked shadow state for a specific memory range, typically used when memory is freed.
- `STATUS`: Returns a detailed textual dump of the current internal configuration (e.g., tracking mode, preload flag status, pointer to the shadow L1 table).
- `CLEAR`: Launches a GPU kernel (`uvm_clear_l3s_kernel`) to quickly zero out all bits in the allocated L3 bitmap shadow pages, essentially resetting the tracked access history without destroying the page table structure.
- `SHUTDOWN`: Safely signals the control thread loop to terminate and exit.

## Malloc Interception Wrapper (`libMallocIntercept.cpp`)

The framework provides an `LD_PRELOAD` shared library that hooks into native CUDA memory management APIs (`cudaMalloc`, `cudaMallocManaged`, `cudaFree`, `cudaDeviceSynchronize`) using `dlsym(RTLD_NEXT)`.

### Purpose & Mechanism
When a GPU accesses a memory page for the first time during tracked execution, the shadow page table structure (L2 and L3 layers) must be dynamically allocated. If thousands of GPU threads attempt this simultaneously, the resulting lock contention on the device-side `malloc()` causes massive latency spikes. 

The wrapper mitigates this by **eagerly pre-populating the host-side shadow page tables** the moment an allocation is made.
- **Immediate Pre-population**: If the tracking structure (`g_uvm_shadow_l1`) is already initialized, intercepted allocations immediately trigger `uvm_tracking_preload_range()` to allocate the underlying table layers from the host.
- **Pending Queue**: If an allocation is made *before* the application initializes the tracking runtime (via `init_tracking()`), the wrapper queues the pointer and size in a `g_pending` list. 
- **Synchronization Trigger**: `cudaDeviceSynchronize` is intercepted to flush any pending allocations. Because `init_tracking()` ends with a synchronization barrier, this guarantees the queue is drained and preloaded right before any kernel launches.
- **Cleanup**: `cudaFree` is intercepted to call `uvm_tracking_drop_range()`, ensuring shadow structures are correctly cleaned up to prevent tracking stale memory addresses.

### Related Environment Variables
The wrapper reads the following environment variables during execution to adjust its behavior:
- `UVM_PRELOAD_MANAGED`: When set to `0`, disables pre-population for memory allocated via `cudaMallocManaged`.
- `UVM_PRELOAD_DEVICE`: When set to `0`, disables pre-population for memory allocated via standard `cudaMalloc`.
- `UVM_NO_CONTROL_THREAD`: If set to `1`, prevents the wrapper from automatically spawning the `uvm_control_thread` in the background upon the first intercepted CUDA call.


## Advanced Use Case: Hybrid Tracking with Driver Page Faults

While this repository focuses on compiler-based static instrumentation, another common approach to memory access tracking is **UVM driver page fault induction**. By artificially unmapping pages at the driver level, the resulting page faults can be intercepted to log accesses. 

Both approaches have distinct trade-offs: instrumentation provides ultra-low overhead tracking of dense inner loops without kernel traps, while driver-level faulting offers a catch-all mechanism without requiring source recompilation. For advanced memory management tasks—such as periodic access sampling to inform smart prefetching and migration heuristics—these two paradigms can be unified into a powerful hybrid tracking system.

By leveraging the `uvm_control_thread` UNIX socket alongside a custom or modified UVM driver, you can create a bidirectional control plane where the driver and the user-space instrumentation coordinate tracking responsibilities dynamically.

### Segmented Tracking Responsibilities
You can partition memory tracking based on the allocation type, playing to the strengths of each method:

* **Driver-Side Tracking for `cudaMallocManaged`**: The UVM driver can natively handle tracking for managed memory regions via page fault induction. Since managed memory inherently relies on driver intervention for migration and page mapping between the host and device, piggybacking access logging onto these existing driver routines is highly effective.
* **Instrumentation for `cudaMalloc`**: Pure device allocations do not trigger UVM driver migrations. Instead of extending fault induction to device-side memory (which can be architecturally complex or unsupported), the compiler pass can exclusively track `cudaMalloc` regions. The `LD_PRELOAD` wrapper (`UVM_PRELOAD_DEVICE=1`) ensures these shadow tables are eagerly populated, yielding a complete access map without driver-level page faults.

### Dynamic Overhead Shedding via Socket Control
A hybrid setup allows the UVM driver (or a coordinating user-space daemon) to actively manage the overhead of the compiler instrumentation at runtime. 

If the tracking framework is used to sample access patterns every few intervals to build a prefetching heuristic, continuous tracking becomes redundant once a region's "hotness" is established. The driver can communicate directly with the instrumentation's socket (`/tmp/uvm-ctl.<pid>`) to prune the tracking tree:

1.  **Targeted Unmapping (`DROP_RANGE`)**: Once the driver has confidently modeled the access pattern of a specific memory range, it can issue a `DROP_RANGE <hex-addr> <len>` command to the socket. The instrumentation runtime will drop the shadow state for that range, instantly eliminating the device-side tracking overhead for those specific addresses while continuing to track unknown regions.
2.  **Interval Sampling (`ENABLE` / `SNAPSHOT` / `DISABLE`)**: The driver can act as a choreographer, pulsing the `ENABLE` and `DISABLE` socket commands to sample compiler-instrumented accesses during specific execution windows. By calling `SNAPSHOT` at the end of a window, the driver merges the instrumentation's warp-aggregated logs with its own page fault logs to generate a comprehensive, global access heatmap. 
3.  **State Resets (`CLEAR`)**: Between sampling intervals, the driver can issue a `CLEAR` command to zero-out the L3 shadow bitmaps. This allows the system to establish fresh access epochs for dynamic workloads without the overhead of tearing down and reallocating the underlying page tables.