# LLVM Compiler-Pass For Access Tracking in UVM GPUs

This repository implements a lightweight, compiler-based memory tracking system for Unified Virtual Memory (UVM) on NVIDIA GPUs. Instead of relying on slow runtime sampling or heavily contended device-side memory tracking, this project uses an LLVM pass to statically analyze loop memory access patterns (via Scalar Evolution) and injects highly optimized, warp-aggregated tracking calls.

## Key Features

- **Intelligent Compiler Instrumentation**: An LLVM 20 pass (`UvmTrackingPass.cpp`) that hoists and batches memory tracking calls out of tight loops whenever possible, drastically reducing runtime overhead.
- **Warp-Aggregated Tracking**: The device-side tracking kernel (`libMarkAccess.cu`) uses a lock-free 3-level shadow page table and warp-level deduplication to minimize global memory atomics.
- **Runtime Interception**: An `LD_PRELOAD` wrapper (`libMallocIntercept.cpp`) intercepts `cudaMalloc` and `cudaMallocManaged` to eagerly pre-populate the shadow page tables on the host, preventing fatal device-side `malloc` bottlenecks.
- **Control Socket Thread**: A background host thread (`uvm_control_thread.cu`) enables you to dynamically toggle tracking, clear tracking state, and dump snapshots to disk via a UNIX domain socket.
- **Deterministic Logging**: Exports dense binary files (`.pglog`) containing the exact 4KB pages touched by the GPU.

## Prerequisites

- LLVM 20 (`llvm-20`, `clang++-20`)
- NVIDIA CUDA Toolkit (installed at `/usr/local/cuda`)
- NVIDIA GPU with UVM support
- Python 3.x

## Quick Start

1. **Build the compiler pass:**
   ```bash
   cd build
   cmake .. -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
   make
   ```

2. **Instrument and compile a CUDA program:**
   ```bash
   clang++-20 -x cuda \
     --cuda-gpu-arch=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.') \
     -fgpu-rdc \
     -fpass-plugin=./build/UvmTrackingPass.so \
     ./my_kernel.cu ./libMarkAccess.cu \
     --cuda-path=/usr/local/cuda \
     -L/usr/local/cuda/lib64 -lcudart \
     -o my_program
   ```

3. **Run with tracking enabled:**
   ```bash
   ./my_program
   ```

4. **Visualize the results:**
   ```bash
   python3 -m venv env && source env/bin/activate
   pip install dash numpy
   python pagelog_drill.py build/access_log.bin
   ```
   Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

## Project Structure

- **`UvmTrackingPass.cpp`**: The LLVM pass that analyzes PTX functions and instruments memory accesses.
- **`libMarkAccess.cu`**: Runtime kernel implementing `MarkAccess` and `BatchMarkAccess` with the 3-level shadow page table.
- **`libMallocIntercept.cpp`**: `LD_PRELOAD` interceptor for CUDA allocation functions.
- **`uvm_control_thread.cu` / `uvm_control_thread.h`**: UNIX socket server and device kernels for dynamic control.
- **`tests/`**: Comprehensive test suite with custom `harness.py` for different tracking modes.
- **`scripts/`**: Utility scripts for benchmarking and analysis (e.g., `benchmark_history.py`, `page_analysis.py`).
- **`CMakeLists.txt`**: Build configuration for the pass and runtime objects.

## Getting Started

### 1. Build the Compiler Pass

From the project root:
```bash
cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
make
```

### 2. Add Required API Calls to Your CUDA Program

Before running any tracked kernel, add these three calls to your host code:

```cpp
// 1. Initialize the tracking structure on the device
init_tracking(&d_l1);

// --- your kernel launches here ---

// 2. Export the log as human-readable text (for debugging)
export_log(d_l1, "access_log.txt");

// 3. Export the log as binary (required for visualization)
export_binary(d_l1, "access_log.bin");
```

**Function Documentation:**
- **`init_tracking`**: Allocates and initializes the device-side access log structure before kernels run.
- **`export_log`**: Copies the log to host and writes human-readable text. Useful for quick inspection and debugging.
- **`export_binary`**: Writes the log in compact binary format (consumed by `pagelog_drill.py`). **Required for dashboard visualization.**

### 3. Instrument Your CUDA Program

Compile with the instrumentation pass injected via `-fpass-plugin`:

```bash
clang++-20 -x cuda \
  --cuda-gpu-arch=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.') \
  -fgpu-rdc \
  -fpass-plugin=./build/UvmTrackingPass.so \
  ./my_kernel.cu ./libMarkAccess.cu \
  --cuda-path=/usr/local/cuda \
  -L/usr/local/cuda/lib64 -lcudart \
  -o my_program
```

> **Note:** GPU architecture (`sm_XX`) is detected automatically at compile time.

### 4. Run the Instrumented Binary

```bash
./my_program
```

This produces `build/access_log.bin` — a binary log of all tracked memory accesses.

## Visualization & Analysis

### Set Up the Python Environment

```bash
python3 -m venv testing_env
source testing_env/bin/activate
pip install dash numpy
```

### Launch the Interactive Dashboard

```bash
python pagelog_drill.py build/access_log.bin
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) to explore the access log interactively.

### Analyze Results from Snapshots

Once you have dumped a snapshot (e.g., `results.pglog`), analyze it with:

```bash
python3 scripts/page_analysis.py /path/to/results.pglog
```

## Advanced Usage: Custom Applications with Control Thread

To apply the UVM tracking pass to your own CUDA kernels with full control thread support, compile using `clang++-20` and load the LLVM pass plugin.

### Compilation

Compile your kernels with the `-fpass-plugin` flag and link the runtime support files:

```bash
clang++-20 -x cuda --cuda-gpu-arch=sm_80 -fgpu-rdc -O2 -rdynamic \
    -fpass-plugin=./build/UvmTrackingPass.so \
    -DTRACKING_ENABLED \
    your_kernel.cu \
    libMarkAccess.cu \
    uvm_control_thread.cu \
    -lcudart -o your_app
```

### Runtime Execution

To eagerly allocate the shadow page table, run your application with the `LD_PRELOAD` wrapper:

```bash
LD_PRELOAD=./build/libMallocIntercept.so ./your_app
```

### Controlling Tracking at Runtime

When your instrumented application starts, it spawns a background thread listening on `/tmp/uvm-ctl.<pid>`. Connect using `socat`, `nc`, or Python sockets to orchestrate tracking dynamically.

**Supported Socket Commands** (end with newline `\n`):
- `ENABLE`: Turn on GPU tracking.
- `DISABLE`: Turn off GPU tracking.
- `STATUS`: Query current tracking mode and configurations.
- `CLEAR`: Launch a device kernel to zero out tracking bitmaps.
- `SNAPSHOT <absolute_path>`: Dump the tracked page table state to a file.
- `PRELOAD_MANAGED <0|1>`: Toggle eager population of managed memory by `LD_PRELOAD`.
- `SHUTDOWN`: Safely terminate the control thread.

## How It Works

1. **Compile-Time Instrumentation**: The LLVM pass (`UvmTrackingPass.so`) analyzes memory operations in GPU kernels and injects hoisted, batched tracking calls to minimize loop overhead.

2. **Runtime Tracking**: `libMarkAccess.cu` records accesses into a shared binary log using a lock-free 3-level shadow page table with warp-level deduplication to reduce global memory contention.

3. **Preload Interception**: The `LD_PRELOAD` interceptor eagerly populates the shadow page table on the host, preventing device-side allocation bottlenecks.

4. **Dynamic Control**: The control socket thread enables you to toggle tracking, clear state, and export snapshots without recompilation.

5. **Visualization & Analysis**: `pagelog_drill.py` parses the binary log and renders an interactive dashboard for drill-down analysis of memory access patterns.

## Testing

The project includes a comprehensive test suite covering control thread interaction, phase tracking, and different tracking modes (No-preload, Preload-Alloc, and Preload-Only).

```bash
python3 run_tests.py
```

## Documentation

For a comprehensive deep dive into the methodology—including how the compiler pass classifies loop strides via Scalar Evolution and how `__shfl_xor_sync` warp-reductions operate—see the `architecture_summary.md` document (if available) or review the detailed comments in `UvmTrackingPass.cpp` and `libMarkAccess.cu`.
