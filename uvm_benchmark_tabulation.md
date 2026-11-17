# Master Tabulation: Coalesced UVM Prefetcher & Access Tracking Experiments

This master tabulation consolidates **all** empirical experiments conducted across **`llama.cpp`**, **`vLLM`**, **`PyTorch TorchVision`**, and the **24-Kernel Benchmark Suite**. It compares:
1. **Access Tracking & Tripwire Performance**: Non-intrusive observation cost, temporal resolution ($\Delta t$), spatial precision, and coverage.
2. **Proactive Prefetcher Oversubscription Performance**: Throughput acceleration, latency reduction, and hardware page-fault elimination across deep learning CV, LLM batch serving, and multi-turn conversational inference.

All measurements were performed on an **NVIDIA A40 GPU (`sm_86`, 48 GB GDDR6, PCIe Gen4 x16 ~25 GB/s)** under CUDA 13.0 and Ubuntu Linux x86_64.

---

## 1. Production Engines: Access Tracking & Tripwire Live Overhead

Evaluates the non-intrusive runtime overhead and observation fidelity of compiler-directed static analysis and runtime tripwire interception with prefetching disabled (measuring purely observation cost vs. native un-instrumented execution).

| Production Engine & Model | Domain / Workload | Metric | Clean Native Baseline | With Tripwire Access Tracking | Tracking Overhead | Total Memory Footprint | Launches / Milestones | Temporal Granularity ($\Delta t$) | Spatial Recall | Unawareness Stride (Precision) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama.cpp`** (Qwen 2.5 Coder 7B Q4_K_M) | LLM Generative Decode | **Decode Throughput** | **103.1 tok/s** | **101.3 tok/s** | **+1.7%** (0.0% on GPU) | 4.4 GB ($1,126,400$ p) | 2,783 launches (~87 trips/tok) | **16.8 µs** | **100.0%** | 73.9% (pool) $\to$ **3.4%** (clamped) |
| | | Prompt Processing | 1,131.2 tok/s | 991.8 tok/s | +12.3% | — | — | — | — | — |
| **`vLLM`** (Qwen 2.5 Coder 7B BF16 Eager) | LLM Batch Serving | **Decode Throughput** | **35.43 tok/s** | **31.01 tok/s** | **+12.4%** (~10.7 µs/kernel) | 14.25 GB weights | 12,052 launches (32 steps) | **~25.0 µs** | **100.0%** | >95% precision |
| | | Decode Latency / tok | 28.22 ms | 32.25 ms | +12.4% (+4.03 ms/tok) | — | ~376 micro-kernels/tok | — | — | (Collapses to <1% with CUDA Graphs) |
| **`PyTorch TorchVision`** (ResNet-50 Batch=32) | Deep Learning CV Inference | **Batch Step Time** | **25.18 ms** | **25.18 ms** | **0.0% (0.0% penalty)** | 148.0 MB ($37,888$ p) | 83 launches / pass | **138.3 µs** | **100.0%** | 95.6% (pool) $\to$ **4.1%** (clamped) |

---

## 2. `llama.cpp` Oversubscription & Prefetcher Performance

Evaluated using `llama.cpp` (`b1-ca89968`) with Unified Memory enabled (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) under constrained physical VRAM budgets enforced via pinned dummy allocations.

### Experiment 2.1: Qwen 2.5 Coder 14B Q4_K_M (8.4 GB Model) under 3.5 GiB Physical VRAM Limit
* **Oversubscription**: ~5.0 GiB of model weights forced into Host CPU DRAM. Generation length = 32 tokens (`--temp 0.0`).

| Metric | 1. Native CPU Offload (`-ngl 10`, 38 on CPU) | 2. Vanilla UVM Demand Paging (`-ngl 999`) | 3. Our Synchronous Prefetcher (`UVM_PREFETCH_SYNC=1`) | Prefetcher vs. Demand Paging |
| :--- | :---: | :---: | :---: | :---: |
| **Prefill Speed** | 16.10 tok/s | 15.60 tok/s | **8.20 tok/s** (27.40 in Phase 2) | — |
| **Decode Speed** | 0.60 tok/s | 0.20 tok/s | **0.40 tok/s** (0.50 in Phase 2) | **2.00x Faster (+100.0% Throughput)** |
| **Total Wall Duration** | 77.66 s | 196.58 s | **116.91 s** (71.75s in Phase 2) | **1.68x Faster (-40.5% Latency / -79.67s)** |
| **Hardware GPU Page Faults**| **0** | 16,879,739 faults | **10,673,569 faults** | **-6,206,170 Faults Slashed (-36.8%)** |
| **Prefetch Calls Issued** | N/A | N/A | **7,724 calls** (1,734 in Phase 2) | — |
| **Prefetched Data Volume** | N/A | N/A | **245.3 GB** (24.5 GB in Phase 2) | — |

---

### Experiment 2.2: Qwen 2.5 Coder 7B Q4_K_M (4.4 GB Model) under 2.0 GiB Physical VRAM Limit
* **Oversubscription**: ~2.4 GiB of model weights forced into Host CPU DRAM. Generation length = 64 tokens (`--temp 0.0`).

| Metric | 1. Native CPU Offload (`-ngl 6`, 22 on CPU) | 2. Vanilla UVM Demand Paging (`-ngl 999`) | 3. Our Synchronous Prefetcher (`UVM_PREFETCH_SYNC=1`) | Prefetcher vs. Demand Paging |
| :--- | :---: | :---: | :---: | :---: |
| **Prefill Speed** | 26.80 tok/s | 30.20 tok/s | **14.70 tok/s** | — |
| **Decode Speed** | 1.00 tok/s | 0.40 tok/s | **0.40 tok/s** | **1.00x (Parity)** |
| **Total Wall Duration** | 66.83 s | 173.54 s | **177.14 s** | 0.98x |
| **Hardware GPU Page Faults**| **0** | 16,161,437 faults | **15,763,174 faults** | **-398,263 Faults Eliminated** |
| **Prefetch Calls Issued** | N/A | N/A | **4,481 calls** | — |
| **Prefetched Data Volume** | N/A | N/A | **142.3 GB** | — |

---

## 3. `PyTorch TorchVision` ResNet-50 Oversubscription Breakthrough

Evaluated on ResNet-50 (batch size = 32). All 53 convolutional layers and model parameters ($97.70$ MB) are forced to Host CPU DRAM via `cudaMemAdviseSetPreferredLocation(cudaCpuDeviceId)`.

| Metric | Native In-VRAM Baseline | Vanilla UVM Demand Paging | Our Proactive Prefetcher | Improvement vs. Demand Paging |
| :--- | :---: | :---: | :---: | :---: |
| **Forward Pass Latency** | **25.18 ms** | **312.94 ms** | **53.63 ms** | **-82.9% (-259.31 ms Latency Saved)** |
| **Inference Throughput** | **1,270.8 img/s** | **102.3 img/s** | **596.7 img/s** | **+483.6% (5.84x Speedup)** |
| **Hardware Page Faults** | **0** | **100,359 faults / pass** | **0 faults / pass** | **-100.0% (Zero Faults / 100% Eliminated)** |
| **Performance Recovery** | 100.0% (In-VRAM) | 8.0% of Native | **46.9% of Native** | **5.84x Recovery of Unconstrained Speed** |

---

## 4. `vLLM` Production LLM Serving Oversubscription Scenarios

Evaluated on `vLLM` (V1 Engine) serving `Qwen/Qwen2.5-Coder-7B-Instruct` under strict physical GPU memory budgets on the NVIDIA A40.

### Scenario 4.1: Severe KV-Cache Oversubscription (22.25 GiB on Host DRAM, 2.0 GiB GPU Budget)
* **Workload**: 20 concurrent requests, 512 prompt tokens, 64 generated tokens per request (11,520 total sequence tokens). 24.25 GiB KV-cache capacity (454,048 tokens), 22.25 GiB forced to CPU DRAM.

#### A. FlashAttention Backend (`FLASH_ATTN`)
| Metric | Native In-VRAM Baseline | Vanilla UVM Demand Paging | Pre-Launch Prefetcher (`UVM_PREFETCH_SYNC=1`) | Impact vs. Vanilla Demand Paging |
| :--- | :---: | :---: | :---: | :---: |
| **Request Throughput** | **6.23 req/s** | **2.34 req/s** | **5.37 req/s** | **+129.5% (+2.30x Speedup)** |
| **Output Token Speed** | **399.02 tok/s** | **149.59 tok/s** | **343.96 tok/s** | **+130.0% (+2.30x Speedup)** |
| **Total Token Speed** | **3,591.21 tok/s** | **1,346.31 tok/s** | **3,095.62 tok/s** | **+130.0% (+2.30x Speedup)** |
| **Total Elapsed Time** | **3.21 s** | **8.54 s** | **3.65 s** | **-57.3% (-4.89s Latency Reduction)** |
| **Hardware GPU Page Faults**| **0** | **10,370 faults** | **0 faults** | **-100.0% (Zero Faults / Complete Elimination)** |
| **Prefetched Data Volume** | 0 MB | 0 MB | **120,064 MB (120.0 GB)** | 3,752 prefetch calls |
| **% of Native Speed** | **100.0%** | **37.6%** | **86.2%** | **Recovers 86.2% of unconstrained VRAM speed** |

#### B. Triton Attention Backend (`--attention-backend TRITON_ATTN`)
| Metric | Vanilla UVM Demand Paging | Pre-Launch Prefetcher (`UVM_PREFETCH_SYNC=1`) | Impact vs. Vanilla Demand Paging |
| :--- | :---: | :---: | :---: |
| **Request Throughput** | **1.27 req/s** | **3.20 req/s** | **+152.0% (+2.52x Speedup)** |
| **Output Token Speed** | **81.34 tok/s** | **204.54 tok/s** | **+151.5% (+2.52x Speedup)** |
| **Total Token Speed** | **732.02 tok/s** | **1,840.88 tok/s** | **+151.5% (+2.52x Speedup)** |
| **Total Elapsed Time** | **15.70 s** | **6.24 s** | **-60.3% (-9.46s Latency Reduction)** |
| **Hardware GPU Page Faults**| **14,495 faults** | **196 faults** | **-98.6% (-14,299 Faults Eliminated)** |
| **Prefetched Data Volume** | 0 MB | **128,536 MB (128.5 GB)** | 18,032 prefetch calls |

---

### Scenario 4.2: Extreme Full Weight + KV-Cache Oversubscription (18.9× Hardware Memory Pressure)
* **Configuration**: 14.25 GiB Model Weights + 24.25 GiB KV-Cache managed by UVM under a strict **2.0 GiB GPU physical limit**.
* **Host DRAM Resident**: **~12.65 GiB of layer weights** + 100% of KV-cache forced onto Host DRAM.

| Metric | Vanilla UVM Demand Paging | Our Synchronous Prefetcher (`UVM_PREFETCH_SYNC=1`) | Impact / Acceleration |
| :--- | :---: | :---: | :---: |
| **Output Token Speed** | **0.36 tok/s** (2.80 s / token) | **13.41 tok/s** | **+37.25x Speedup (+3,625%)** |
| **16-Token Generation Time** | **44.79 s** | **1.19 s** | **-97.3% Latency Drop (-43.60s saved)** |
| **Engine Initialization Time**| **94.55 s** | **5.29 s** | **17.8x Faster Engine Startup** |
| **Warmup/Init GPU Faults** | **620,104 faults** | Continuous DMA prefetch | Massive fault storm eliminated |
| **Generation GPU Faults** | **96 faults** | **15 faults** | **-84.4% Fault Reduction** |
| **Output Correctness** | Exact identical code completion | Exact identical code completion | 100% Bit-accurate verification |

---

### Scenario 4.3: Multi-Turn Context Inflation (3 Sequential Conversational Turns)
* **Configuration**: Progressive context inflation under full weights (12.65 GiB) and KV-cache host offloading. Each turn appends previous context and code completions. Greedy decoding (`temp=0.0`).

| Turn / Context State | Metric | Vanilla Demand Paging | Our Synchronous Prefetcher | Acceleration / Comparison |
| :--- | :--- | :---: | :---: | :---: |
| **Turn 1 (124 $\to$ 284 tokens)** | Elapsed Generation Time | 286.17 s (4m 46s) | **11.11 s** | **25.75x Faster (-275.06s saved)** |
| | Decode Throughput | 0.56 tok/s | **14.40 tok/s** | **+2,471% Throughput** |
| | Turn GPU Page Faults | 184 faults | 81 faults | -56.0% Fault Reduction |
| **Turn 2 (398 $\to$ 558 tokens)** | Elapsed Generation Time | 285.04 s (4m 45s) | **10.43 s** | **27.33x Faster (-274.61s saved)** |
| | Decode Throughput | 0.56 tok/s | **15.34 tok/s** | **+2,639% Throughput** |
| | Turn GPU Page Faults | 0 faults | 0 faults | Clean DMA streaming |
| **Turn 3 (683 $\to$ 843 tokens)** | Elapsed Generation Time | 287.58 s (4m 47s) | **10.47 s** | **27.47x Faster (-277.11s saved)** |
| | Decode Throughput | 0.56 tok/s | **15.28 tok/s** | **+2,628% Throughput** |
| | Turn GPU Page Faults | 0 faults | 0 faults | Clean DMA streaming |
| **Cumulative Total (3 Turns)** | **Total Generation Time** | **858.79 s (14 min 18.8 s)** | **32.01 s** | **26.83x Faster (-13m 46.8s saved)** |
| | **Average Decode Speed** | **0.56 tok/s** | **14.99 tok/s** | **+2,577% Speedup** |
| | **Engine Initialization** | 115.90 s | 32.03 s | 3.6x Faster Startup |
| | **Init Warmup Page Faults**| 617,325 faults | Continuous DMA prefetch | Massive fault storm prevented |
| | **Total Turn Page Faults** | 184 faults | 81 faults | -56.0% Turn Fault Reduction |

---

## 5. Standard 24-Benchmark Suite: Spatial Precision & Tripwire Granularity

Consolidated metrics across Polybench, CUTLASS, and Synthetic Memory Patterns showing compiler pass spatial precision and temporal milestone spacing.

| Category | Benchmark Kernel | Memory Footprint | Spatial Coverage (Recall) | Milestone Granularity ($\Delta t$) | Unawareness Stride (Precision) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stencils & Convolutions** | `2DConvolution` | 190.7 MB | **100.0%** | 14,225.4 µs | 100.0% (single tile) |
| | `3DConvolution` | 189.3 MB | **99.7%** | 74.9 µs | 50.0% (halo boundary) |
| | `fdtd2d` | 476.8 MB | **100.0%** | 209.3 µs | 77.8% |
| **Linear Algebra & BLAS** | `MatrixMul` (1024) | 12.0 MB | **100.0%** | 34.2 µs | **1.0%** |
| | `CUTLASS` (2048) | 48.0 MB | **100.0%** | 254.0 µs | **1.0%** |
| | `CUTLASS` (4096) | 192.0 MB | **100.0%** | 639.1 µs | **1.0%** |
| | `2mm` | 190.7 MB | **100.0%** | 459.3 µs | **0.6%** |
| | `3mm` | 286.0 MB | **100.0%** | 352.0 µs | **0.7%** |
| | `gemm` | 190.7 MB | **100.0%** | 240.2 µs | **0.5%** |
| | `gemver` | 190.8 MB | **100.0%** | 353.4 µs | **1.6%** |
| | `gesummv` | 190.8 MB | **100.0%** | 140.2 µs | **1.6%** |
| | `atax` | 190.8 MB | **100.0%** | 295.9 µs | **1.6%** |
| | `bicg` | 190.8 MB | **100.0%** | 376.5 µs | **1.6%** |
| | `syrk` | 32.0 MB | **100.0%** | 1,723.8 µs | **2.2%** |
| | `syr2k` | 32.0 MB | **100.0%** | 294.5 µs | **1.2%** |
| | `covariance` | 47.7 MB | **100.0%** | 24,057.7 µs | **0.5%** |
| | `correlation` | 47.7 MB | **100.0%** | 28,143.2 µs | **0.5%** |
| | `gramschmidt` | 32.0 MB | **100.0%** | 6.2 µs | **2.0%** |
| | `mvt` | 190.8 MB | **100.0%** | 239.5 µs | **1.6%** |
| | `doitgen` | 190.8 MB | **100.0%** | 1,480.9 µs | **0.8%** |
| | `lu` | 64.0 MB | **100.0%** | 2,870.0 µs | **1.0%** |
| | `ludcmp` | 16.0 MB | **100.0%** | 120.0 µs | **1.0%** |
| **Irregular & Strided** | `needle` (NW) | 1,906.7 MB | **100.0%** | 8.6 µs | **1.9%** |
| | `Transpose` | 32.0 MB | **100.0%** | 2,231.3 µs | 100.0% |
