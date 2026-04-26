# LLVM Compiler-Pass For Access Tracking in UVM GPUs

This repository implements a lightweight, compiler-based memory tracking system for Unified Virtual Memory (UVM) on NVIDIA GPUs. Instead of relying on slow runtime sampling or heavily contended device-side memory tracking, this project uses an LLVM pass to statically analyze loop memory access patterns (via Scalar Evolution) and injects highly optimized, warp-aggregated tracking calls.

## Key Features

- **Intelligent Compiler Instrumentation**: An LLVM 20 pass (`UvmTrackingPass.cpp`) that hoists and batches memory tracking calls out of tight loops whenever possible, drastically reducing runtime overhead.
- **Warp-Aggregated Tracking**: The device-side tracking kernel (`libMarkAccess.cu`) uses a lock-free 3-level shadow page table and warp-level deduplication to minimize global memory atomics.
- **Runtime Interception**: An `LD_PRELOAD` wrapper (`libMallocIntercept.cpp`) intercepts `cudaMalloc` and `cudaMallocManaged` to eagerly pre-populate the shadow page tables on the host, preventing fatal device-side `malloc` bottlenecks.
- **Control Socket Thread**: A background host thread (`uvm_control_thread.cu`) enables you to dynamically toggle tracking, clear tracking state, and dump snapshots to disk via a UNIX domain socket.
- **Deterministic Logging**: Exports dense binary files (`.pglog`) containing the exact 4KB pages touched by the GPU.

## Project Structure

- `UvmTrackingPass.cpp`: The LLVM pass that analyzes PTX functions and instruments memory accesses.
- `libMarkAccess.cu`: The runtime kernel implementing `MarkAccess` and `BatchMarkAccess` with the 3-level shadow page table.
- `libMallocIntercept.cpp`: The `LD_PRELOAD` interceptor for CUDA allocation functions.
- `uvm_control_thread.cu` / `uvm_control_thread.h`: The UNIX socket server and device kernels for dynamically toggling and clearing page logs.
- `tests/`: A comprehensive test suite using a custom `harness.py` to compile and orchestrate tests across different tracking modes.
- `scripts/`: Assorted python scripts and utilities (e.g., `benchmark_history.py`, `page_analysis.py`).
- `CMakeLists.txt`: Build configuration for the compiler pass and related objects.

## Quick Start

### Prerequisites

- NVIDIA GPU with UVM support.
- CUDA Toolkit 12.x or later.
- LLVM 20 (required for the opaque-pointer IR pass).
- C++17 compatible compiler (GCC/Clang).
- Python 3.6+.

### Building the Pass and Interceptor

You can build the tracking pass and interceptor library using standard CMake:

```bash
mkdir build
cd build
cmake .. -DLLVM_DIR=...
make
```

### Running the Test Suite

The project comes with a robust test suite covering control thread socket interaction, phase tracking, and different tracking modes (No-preload, Preload-Alloc, and Preload-Only).

```bash
python3 run_tests.py
```

## Instrumenting Custom Applications

To apply the UVM tracking pass to your own CUDA kernels, you must compile your `.cu` files using `clang++-20` and load the LLVM pass plugin. 

### 1. Compilation

Compile your kernels by passing the `-fpass-plugin` flag pointing to the built `UvmTrackingPass.so`. You must also link the runtime support files.

```bash
clang++-20 -x cuda --cuda-gpu-arch=sm_80 -fgpu-rdc -O2 -rdynamic \
    -fpass-plugin=./build/UvmTrackingPass.so \
    -DTRACKING_ENABLED \
    your_kernel.cu \
    libMarkAccess.cu \
    uvm_control_thread.cu \
    -lcudart -o your_app
```

### 2. Runtime Execution

To ensure the shadow page table is eagerly allocated, run your compiled application with the `LD_PRELOAD` wrapper:

```bash
LD_PRELOAD=./build/libMallocIntercept.so ./your_app
```

## Controlling Tracking at Runtime

When the instrumented application starts, it spawns a background thread listening on `/tmp/uvm-ctl.<pid>`. You can connect to this socket using `socat`, `nc`, or standard Python sockets to orchestrate tracking dynamically.

Supported Socket Commands (must end with a newline `\n`):
- `ENABLE`: Turn on GPU tracking.
- `DISABLE`: Turn off GPU tracking.
- `STATUS`: Query current tracking mode and configurations.
- `CLEAR`: Launch a device kernel to zero out the existing tracking bitmaps.
- `SNAPSHOT <absolute_path>`: Dump the tracked page table state to the specified file.
- `PRELOAD_MANAGED <0|1>`: Toggle whether the `LD_PRELOAD` interceptor eagerly populates managed memory.
- `SHUTDOWN`: Safely terminate the control thread.

## Analyzing Results

Once you have dumped a snapshot (e.g., `results.pglog`), you can use the provided python scripts to analyze the memory footprint.

```bash
python3 scripts/page_analysis.py /path/to/results.pglog
```

## Documentation

For a comprehensive deep dive into the methodology—including how the compiler pass classifies loop strides via Scalar Evolution, and how the `__shfl_xor_sync` warp-reductions operate—please read the `architecture_summary.md` document (if available) or review the comments directly inside `UvmTrackingPass.cpp` and `libMarkAccess.cu`.
