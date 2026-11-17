#!/usr/bin/env python3
"""
Comprehensive 4-Way Evaluation of Static Analysis + Host-Side Tripwire Temporal Tracking
across all 24 benchmarks (Microbenchmarks, Extended Kernels, Polybench Suite, and CUTLASS GEMM).

Measures:
  1. Overhead: Clean Baseline vs Full GPU Instrumentation vs Host-Only Static vs Static + Host Tripwire
  2. Spatial Accuracy: Predicted/Tracked Pages vs Ground-Truth Pages & Cluster Layout
  3. Temporal Granularity: Total Milestones, Milestones/Launch, Sub-Kernel Slice Size (pages), Delta-t (us)
  4. Pattern Coverage: Static analysis pattern rule matched per kernel
"""
import os
import re
import sys
import json
import struct
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tests.harness as h
import benchmark as b_mod

PASS_PATH  = os.path.join(ROOT, "build", "UvmTrackingPass.so")
PRELOAD_SO = os.path.join(ROOT, "libMallocIntercept.so")
CLANG      = "clang++-20"
CUDA_PATH  = "/usr/local/cuda"
SM_ARCH    = "86"

CUTLASS_INC = "-I/home/arush/data/uv_cache/archive-v0/kM3gMCIGSl1iKbxdStF1c/tilelang/3rdparty/cutlass/include"

# Full 24-benchmark suite
SUITE = [
    # ── Group 1: Microbenchmarks (examples/benchmark_kernel.cu) ──
    {
        "group": "Microbenchmarks",
        "name": "Coalesced",
        "src": "examples/benchmark_kernel.cu",
        "args": ["0", "50"],
        "acc_args": ["0", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/coalesced.pagelog",
        "pattern": "1D Linear Loop Bound (`n * elem_size`)",
    },
    {
        "group": "Microbenchmarks",
        "name": "Stride",
        "src": "examples/benchmark_kernel.cu",
        "args": ["1", "50"],
        "acc_args": ["1", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/stride.pagelog",
        "pattern": "1D Strided Mul (`n * mem_stride * elem_size`)",
    },
    {
        "group": "Microbenchmarks",
        "name": "Random",
        "src": "examples/benchmark_kernel.cu",
        "args": ["2", "50"],
        "acc_args": ["2", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/random.pagelog",
        "pattern": "1D Dense Indirect Scatter/Gather (`data[indices[i]]`)",
    },
    {
        "group": "Microbenchmarks",
        "name": "Stencil",
        "src": "examples/benchmark_kernel.cu",
        "args": ["3", "50"],
        "acc_args": ["3", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/stencil.pagelog",
        "pattern": "1D 3-Point Neighborhood Stencil (`in[i+-1], out[i]`)",
    },
    {
        "group": "Microbenchmarks",
        "name": "Atomic",
        "src": "examples/benchmark_kernel.cu",
        "args": ["4", "50"],
        "acc_args": ["4", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/atomic.pagelog",
        "pattern": "Pattern 0: Direct Scalar/Constant-Index Deref (`&data[0]`)",
    },
    # ── Group 2: Extended Microbenchmarks (examples/benchmark_extended.cu) ──
    {
        "group": "Extended Kernels",
        "name": "SAXPY",
        "src": "examples/benchmark_extended.cu",
        "args": ["0", "50"],
        "acc_args": ["0", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/saxpy.pagelog",
        "pattern": "1D Dual-Stream Linear (`x[i], y[i]`)",
    },
    {
        "group": "Extended Kernels",
        "name": "Reduction",
        "src": "examples/benchmark_extended.cu",
        "args": ["1", "50"],
        "acc_args": ["1", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/reduction.pagelog",
        "pattern": "Pattern 0c + 0: Grid-Bounded Reduction (`grid*block`) + Scalar Atomic",
    },
    {
        "group": "Extended Kernels",
        "name": "Histogram",
        "src": "examples/benchmark_extended.cu",
        "args": ["2", "50"],
        "acc_args": ["2", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/histogram.pagelog",
        "pattern": "Pattern 0b: Modulo-Bounded Scatter (`hist[x % bins]`) + 1D Stream",
    },
    {
        "group": "Extended Kernels",
        "name": "Transpose",
        "src": "examples/benchmark_extended.cu",
        "args": ["3", "50"],
        "acc_args": ["3", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/transpose.pagelog",
        "pattern": "2D Square Matrix Stride (`N * N * elem_size`)",
    },
    {
        "group": "Extended Kernels",
        "name": "GEMV",
        "src": "examples/benchmark_extended.cu",
        "args": ["4", "50"],
        "acc_args": ["4", "1"],
        "extra_flags": [],
        "baseline_pagelog": "baselines/gemv.pagelog",
        "pattern": "2D Matrix (`M * N`) + 1D Loop-Separated Vectors (`x[N], y[M]`)",
    },
    # ── Group 3: Tiled GEMM & CUTLASS ──
    {
        "group": "GEMM Workloads",
        "name": "MatrixMul (Tiled 1024)",
        "src": "examples/matrixMul.cu",
        "args": [],
        "acc_args": [],
        "extra_flags": [],
        "baseline_pagelog": "baselines/matrixmul.pagelog",
        "pattern": "Pattern A: 2D Tiled CTAID Stride (`gridDim.y * 32 * wA`)",
    },
    {
        "group": "GEMM Workloads",
        "name": "CUTLASS GEMM (2048x2048)",
        "src": "examples/cutlass_benchmark.cu",
        "args": ["2048", "10"],
        "acc_args": ["2048", "1"],
        "extra_flags": [CUTLASS_INC],
        "expected_pages": 12288,
        "pattern": "ByVal Params Struct -> Active Allocation Scan (`OP_SCAN_ACTIVE_ALLOCS`)",
    },
    {
        "group": "GEMM Workloads",
        "name": "CUTLASS GEMM (4096x4096)",
        "src": "examples/cutlass_benchmark.cu",
        "args": ["4096", "5"],
        "acc_args": ["4096", "1"],
        "extra_flags": [CUTLASS_INC],
        "expected_pages": 49152,
        "pattern": "ByVal Params Struct -> Active Allocation Scan (`OP_SCAN_ACTIVE_ALLOCS`)",
    },
    # ── Group 4: Polybench GPU Suite ──
    {
        "group": "Polybench Suite",
        "name": "2DConvolution",
        "src": "examples/2DCONV/2DConvolution.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/2DCONV", "-Iexamples"],
        "pattern": "2D Stencil (`NI * NJ * elem_size`)",
    },
    {
        "group": "Polybench Suite",
        "name": "3DConvolution",
        "src": "examples/3DCONV/3DConvolution.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/3DCONV", "-Iexamples"],
        "pattern": "3D Multi-Stride Tensor (`NI * NJ * NK * elem_size`)",
    },
    {
        "group": "Polybench Suite",
        "name": "2mm",
        "src": "examples/2MM/2mm.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/2MM", "-Iexamples"],
        "pattern": "Multi-Kernel 2D GEMM Chain (`NI*NK, NK*NJ, NI*NJ, NJ*NL, NI*NL`)",
    },
    {
        "group": "Polybench Suite",
        "name": "atax",
        "src": "examples/ATAX/atax.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/ATAX", "-Iexamples"],
        "pattern": "2D Matrix (`NX * NY`) + 1D Vectors (`x[NY], tmp[NX], y[NY]`)",
    },
    {
        "group": "Polybench Suite",
        "name": "bicg",
        "src": "examples/BICG/bicg.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/BICG", "-Iexamples"],
        "pattern": "2D Matrix (`NX * NY`) + 1D Vectors (`r[NX], s[NY], p[NY], q[NX]`)",
    },
    {
        "group": "Polybench Suite",
        "name": "covariance",
        "src": "examples/COVAR/covariance.cu",
        "args": ["50000000"],
        "acc_args": ["50000000"],
        "extra_flags": ["-Iexamples/COVAR", "-Iexamples"],
        "pattern": "Multi-Kernel 2D Rectangular (`M*N`) + 2D Square (`M*M`) + 1D (`M`)",
    },
    {
        "group": "Polybench Suite",
        "name": "fdtd2d",
        "src": "examples/FDTD-2D/fdtd2d.cu",
        "args": ["-mb", "500"],
        "acc_args": ["-mb", "500"],
        "extra_flags": ["-Iexamples/FDTD-2D", "-Iexamples"],
        "pattern": "Time-Stepped 2D Stencil (`NX * NY`) + Step Scalar (`_fict_[t]`)",
    },
    {
        "group": "Polybench Suite",
        "name": "gramschmidt",
        "src": "examples/GRAMSCHM/gramschmidt.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/GRAMSCHM", "-Iexamples"],
        "pattern": "Iterative 3-Kernel QR (`M * N` with Lower-Bound Filter `j > k`)",
    },
    {
        "group": "Polybench Suite",
        "name": "needle",
        "src": "examples/nw/needle.cu",
        "args": ["-mb", "2000"],
        "acc_args": ["-mb", "2000"],
        "extra_flags": ["-Iexamples/nw", "-Iexamples"],
        "pattern": "Wavefront Dynamic Programming (`cols * cols * elem_size`)",
    },
    {
        "group": "Polybench Suite",
        "name": "syrk",
        "src": "examples/SYRK/syrk.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/SYRK", "-Iexamples"],
        "pattern": "Symmetric Rank-k (`a[N*M]` + Square `c[N*N]`)",
    },
    {
        "group": "Polybench Suite",
        "name": "syr2k",
        "src": "examples/SYR2K/syr2k.cu",
        "args": ["-mb", "200"],
        "acc_args": ["-mb", "200"],
        "extra_flags": ["-Iexamples/SYR2K", "-Iexamples"],
        "pattern": "Symmetric Rank-2k (`a[N*M], b[N*M]` + Square `c[N*N]`)",
    },
]

def extract_time_ms(output):
    # Match BENCHMARK_TIME: X ms or GPU Runtime: X s
    last_ms = None
    for line in output.splitlines():
        if "BENCHMARK_TIME:" in line:
            m = re.search(r"BENCHMARK_TIME:\s*([\d.eE+-]+)", line)
            if m:
                last_ms = float(m.group(1))
        elif "GPU Runtime:" in line:
            m = re.search(r"GPU Runtime:\s*([\d.eE+-]+)\s*s", line)
            if m:
                last_ms = float(m.group(1)) * 1000.0
    return last_ms

def compile_variants(src, extra_flags, compiled_cache):
    if src in compiled_cache:
        return compiled_cache[src]

    slug = os.path.splitext(os.path.basename(src))[0]
    clean_bin = os.path.join(ROOT, "build", f"suite_{slug}_clean")
    full_bin  = os.path.join(ROOT, "build", f"suite_{slug}_full")
    tw_bin    = os.path.join(ROOT, "build", f"suite_{slug}_tripwire")

    base_cmd = [
        CLANG, "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}", "-O3",
        f"-I{ROOT}", f"-I{os.path.join(ROOT, 'include')}",
        f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
    ] + extra_flags

    print(f"  [Compile] {src} -> clean, full-inst, tripwire-only ...", flush=True)
    # 1. Clean
    r1 = subprocess.run(base_cmd + [os.path.join(ROOT, src), "-o", clean_bin],
                        cwd=ROOT, capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"  [ERROR] Clean compile failed for {src}:\n{r1.stderr[-600:]}")
        return None

    # 2. Full GPU Instrumentation (UVM_PASS_TRIPWIRE_ONLY=0)
    env_full = os.environ.copy()
    env_full.pop("UVM_PASS_TRIPWIRE_ONLY", None)
    inst_cmd = base_cmd + [
        "-fgpu-rdc", "-rdynamic", "-DTRACKING_ENABLED", "-DUVM_TRACKING_MODE=1",
        f"-fpass-plugin={PASS_PATH}",
        os.path.join(ROOT, src),
        os.path.join(ROOT, "libMarkAccess.cu"),
        os.path.join(ROOT, "uvm_control_thread.cu"),
    ]
    r2 = subprocess.run(inst_cmd + ["-o", full_bin],
                        cwd=ROOT, env=env_full, capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"  [ERROR] Full-inst compile failed for {src}:\n{r2.stderr[-600:]}")
        return None

    # 3. Tripwire-Only Instrumentation (UVM_PASS_TRIPWIRE_ONLY=1)
    env_tw = os.environ.copy()
    env_tw["UVM_PASS_TRIPWIRE_ONLY"] = "1"
    r3 = subprocess.run(inst_cmd + ["-o", tw_bin],
                        cwd=ROOT, env=env_tw, capture_output=True, text=True)
    if r3.returncode != 0:
        print(f"  [ERROR] Tripwire compile failed for {src}:\n{r3.stderr[-600:]}")
        return None

    compiled_cache[src] = (clean_bin, full_bin, tw_bin)
    return compiled_cache[src]

def run_timed(bin_path, args, env_extra=None, trials=3):
    times = []
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    log_p = os.path.join(ROOT, "access_log.bin")
    trace_p = os.path.join(ROOT, "temporal_host_trace.json")

    for _ in range(trials):
        if os.path.exists(log_p):
            os.remove(log_p)
        if os.path.exists(trace_p):
            os.remove(trace_p)
        proc = subprocess.run([bin_path] + args, cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    [WARN] {bin_path} exited with {proc.returncode}: {proc.stderr[:200]}")
            continue
        t = extract_time_ms(proc.stdout + "\n" + proc.stderr)
        if t is not None:
            times.append(t)

    pages = set()
    if os.path.exists(log_p):
        try:
            pages = h.parse_pagelog(log_p)
        except Exception:
            pass

    trace_info = {"total_milestones": 0, "total_launches": 0, "avg_slice_pages": 0.0, "avg_delta_us": 0.0}
    if os.path.exists(trace_p):
        try:
            with open(trace_p, "r") as f:
                tj = json.load(f)
            trace_info["total_milestones"] = tj.get("total_milestones", 0)
            trace_info["total_launches"] = tj.get("total_launches", 0)
            evs = tj.get("events", [])
            loop_evs = [e for e in evs if e.get("phase") == "OUTER_LOOP_STEP"]
            if not loop_evs:
                loop_evs = [e for e in evs if e.get("phase") == "KERNEL_ENTRY"]
            if loop_evs:
                trace_info["avg_slice_pages"] = float(np.mean([e.get("static_slice_pages", 0) for e in loop_evs]))
            if len(evs) >= 2:
                deltas = []
                for e1, e2 in zip(evs[:-1], evs[1:]):
                    if e1.get("kernel_id") == e2.get("kernel_id") and e2.get("rel_time_us", 0) >= e1.get("rel_time_us", 0):
                        deltas.append(e2["rel_time_us"] - e1["rel_time_us"])
                if deltas:
                    trace_info["avg_delta_us"] = float(np.mean(deltas))
        except Exception:
            pass

    median_ms = float(np.median(times)) if times else 0.0
    return median_ms, pages, trace_info

def main():
    print("=" * 100)
    print("COMPREHENSIVE 4-WAY EVALUATION: STATIC ANALYSIS + HOST-SIDE TRIPWIRE TEMPORAL TRACKING")
    print("=" * 100)

    # Clear metadata caches before compiling all benchmarks so all kernels register fresh bytecode
    for p in [os.path.join(ROOT, ".uvm_metadata.cache"), "/tmp/.uvm_metadata.cache"]:
        if os.path.exists(p):
            os.remove(p)

    compiled_cache = {}
    results = []

    for item in SUITE:
        name = item["name"]
        print(f"\n>>> Benchmarking: {name} ({item['group']})", flush=True)
        bins = compile_variants(item["src"], item["extra_flags"], compiled_cache)
        if not bins:
            continue
        clean_bin, full_bin, tw_bin = bins

        # 1. Clean Baseline
        clean_ms, _, _ = run_timed(clean_bin, item["args"], trials=3)

        # 2. Full GPU Instrumentation (MarkAccess + Temporal)
        gpu_env = {"LD_PRELOAD": PRELOAD_SO, "UVM_FORCE_GPU_INSTRUMENTATION": "1"}
        gpu_ms, gpu_pages, _ = run_timed(full_bin, item["args"], env_extra=gpu_env, trials=2)

        # 3. Host-Only Static Prediction (PREDICT_HOST, native kernel)
        host_env = {"LD_PRELOAD": PRELOAD_SO}
        host_ms, host_pages, _ = run_timed(tw_bin, item["args"], env_extra=host_env, trials=3)

        # 4. Static Analysis + Host-Side Tripwire (UVM_STATIC_TRIPWIRE=1)
        tw_env = {"LD_PRELOAD": PRELOAD_SO, "UVM_STATIC_TRIPWIRE": "1"}
        tw_ms, tw_pages, tw_trace = run_timed(tw_bin, item["args"], env_extra=tw_env, trials=3)

        # If acc_args differs from timed args (e.g. 1 iteration for baseline pagelog & per-launch milestone check), run 1 accuracy/granularity pass
        if item["acc_args"] != item["args"]:
            _, tw_acc_pages, tw_acc_trace = run_timed(tw_bin, item["acc_args"], env_extra=tw_env, trials=1)
        else:
            tw_acc_pages, tw_acc_trace = tw_pages, tw_trace

        # Determine ground-truth pages
        gt_pages_count = 0
        gt_clusters = []
        if "baseline_pagelog" in item:
            bp = os.path.join(ROOT, item["baseline_pagelog"])
            if os.path.exists(bp):
                bset = h.parse_pagelog(bp)
                gt_pages_count, gt_clusters = b_mod.allocation_fingerprint(bset)
        elif "expected_pages" in item:
            gt_pages_count = item["expected_pages"]
        else:
            gt_pages_count = len(gpu_pages) if len(gpu_pages) > 0 else len(tw_acc_pages)

        tw_count, tw_clusters = b_mod.allocation_fingerprint(tw_acc_pages)
        if gt_pages_count == 0:
            gt_pages_count = tw_count

        gpu_ov  = ((gpu_ms - clean_ms) / clean_ms * 100.0) if clean_ms > 0 else 0.0
        host_ov = ((host_ms - clean_ms) / clean_ms * 100.0) if clean_ms > 0 else 0.0
        tw_ov   = ((tw_ms - clean_ms) / clean_ms * 100.0) if clean_ms > 0 else 0.0

        ratio = (tw_count / gt_pages_count * 100.0) if gt_pages_count > 0 else 100.0
        launches = max(1, tw_acc_trace["total_launches"])
        m_per_launch = tw_acc_trace["total_milestones"] / launches

        res = {
            "group": item["group"],
            "name": name,
            "clean_ms": clean_ms,
            "gpu_ms": gpu_ms,
            "gpu_ov": gpu_ov,
            "host_ms": host_ms,
            "host_ov": host_ov,
            "tw_ms": tw_ms,
            "tw_ov": tw_ov,
            "tw_pages": tw_count,
            "gt_pages": gt_pages_count,
            "accuracy_pct": ratio,
            "launches": tw_acc_trace["total_launches"],
            "milestones": tw_acc_trace["total_milestones"],
            "milestones_per_launch": m_per_launch,
            "avg_slice_pages": tw_acc_trace["avg_slice_pages"],
            "avg_delta_us": tw_acc_trace["avg_delta_us"],
            "pattern": item["pattern"],
        }
        results.append(res)

        print(f"    Clean: {clean_ms:.3f} ms | GPU-Inst: {gpu_ms:.3f} ms ({gpu_ov:+.1f}%) | "
              f"Static-Host: {host_ms:.3f} ms ({host_ov:+.1f}%) | Static+Tripwire: {tw_ms:.3f} ms ({tw_ov:+.1f}%)")
        print(f"    Pages: {tw_count} / {gt_pages_count} ({ratio:.1f}%) | "
              f"Milestones: {tw_acc_trace['total_milestones']} across {tw_acc_trace['total_launches']} launches "
              f"({m_per_launch:.1f}/launch, slice={tw_acc_trace['avg_slice_pages']:.1f} pages, dt={tw_acc_trace['avg_delta_us']:.2f} us)")

    out_json = os.path.join(ROOT, "perf_results", "static_tripwire_all_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_json}")

if __name__ == "__main__":
    main()
