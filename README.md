# Low-Overhead Access Tracking in NVIDIA UVM GPUs via Static Binary Instrumentation

A compiler-assisted framework for tracking GPU memory access patterns in NVIDIA Unified Virtual Memory (UVM) systems using static binary instrumentation via an LLVM pass.

---

## Overview

This project instruments CUDA kernels at compile time to log memory accesses with minimal runtime overhead. Access logs are captured in a compact binary format and can be interactively visualized through a browser-based dashboard.

---

## Prerequisites

- LLVM 20 (`llvm-20`, `clang++-20`)
- NVIDIA CUDA Toolkit (installed at `/usr/local/cuda`)
- NVIDIA GPU with UVM support
- Python 3.x

---

## Getting Started

### 1. Build the Compiler Pass

From the project root, configure and build the LLVM pass plugin:
```bash
cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
make
```
## 2. Required API Calls in Your CUDA Program

Before running any tracked kernel, you must add three function calls to your host code:
```cpp
// 1. Initialize the tracking structure on the device
init_tracking(&d_l1);

// --- your kernel launches here ---

// 2. Export the log as human-readable text (for debugging/inspection)
export_log(d_l1, "access_log.txt");

// 3. Export the log as binary (required for the visualization dashboard)
export_binary(d_l1, "access_log.bin");
```

> **`init_tracking`** - Allocates and initializes the device-side access log structure before any kernel runs.
>
> **`export_log`** - Copies the log from device memory to the host and writes it as a human-readable text file. Useful for quick inspection and debugging.
>
> **`export_binary`** - Writes the same log in a compact raw binary format. This is the file consumed by `pagelog_drill.py` - the visualization dashboard **will not work** without this.

### 3. Instrument a CUDA Program

Compile your CUDA source with the instrumentation pass injected via `-fpass-plugin`:
```bash
clang++-20 -x cuda \
  --cuda-gpu-arch=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.') \
  -fgpu-rdc \
  -fpass-plugin=./build/UvmTrackingPass.so \
  ../test_kernel.cu ../libMarkAccess.cu \
  --cuda-path=/usr/local/cuda \
  -L/usr/local/cuda/lib64 -lcudart \
  -o my_program
```

> **Note:** The GPU architecture (`sm_XX`) is detected automatically at compile time using `nvidia-smi`.

### 4. Run the Instrumented Binary
```bash
./my_program
```

This produces `build/access_log.bin` — a binary log of all tracked memory accesses.

---

## Visualization

### Set Up the Python Environment
```bash
python3 -m venv testing_env
source testing_env/bin/activate
pip install dash numpy
```

### Launch the Dashboard
```bash
python pagelog_drill.py build/access_log.bin
```

Open your browser and navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) to explore the access log interactively.

---

## Project Structure
```
.
├── build/                  # Build output directory (pass plugin, binaries, logs)
├── test_kernel.cu          # Example CUDA kernel for testing
├── libMarkAccess.cu        # Runtime support library for access marking
├── pagelog_drill.py        # Interactive visualization dashboard
└── CMakeLists.txt
```

---

## How It Works

1. **LLVM Pass** - `UvmTrackingPass.so` instruments memory operations in GPU kernels at compile time.
2. **Runtime Library** - `libMarkAccess.cu` records accesses into a shared binary log with minimal overhead.
3. **Visualization** - `pagelog_drill.py` parses the binary log and renders an interactive Dash dashboard for drill-down analysis.

