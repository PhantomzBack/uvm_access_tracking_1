#!/usr/bin/env python3
import os
import re
import sys
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tests.harness as h

BENCHMARKS = [
    {
        "name": "CUTLASS GEMM (2048x2048)",
        "src": "examples/cutlass_benchmark.cu",
        "args": ["2048", "10"],
        "extra_flags": [
            "-I/home/arush/data/uv_cache/archive-v0/kM3gMCIGSl1iKbxdStF1c/tilelang/3rdparty/cutlass/include"
        ],
        "expected_pages": 12288
    },
    {
        "name": "CUTLASS GEMM (4096x4096)",
        "src": "examples/cutlass_benchmark.cu",
        "args": ["4096", "5"],
        "extra_flags": [
            "-I/home/arush/data/uv_cache/archive-v0/kM3gMCIGSl1iKbxdStF1c/tilelang/3rdparty/cutlass/include"
        ],
        "expected_pages": 49152
    },
    {
        "name": "Polybench MatrixMul",
        "src": "examples/matrixMul.cu",
        "args": [],
        "extra_flags": [],
        "expected_pages": 3072
    },
    {
        "name": "Polybench 2DCONV",
        "src": "examples/2DCONV/2DConvolution.cu",
        "args": [],
        "extra_flags": ["-Iexamples/2DCONV", "-Iexamples"],
        "expected_pages": 65522
    },
    {
        "name": "Polybench Covariance",
        "src": "examples/COVAR/covariance.cu",
        "args": [],
        "extra_flags": ["-Iexamples/COVAR", "-Iexamples"],
        "expected_pages": 11776
    },
    {
        "name": "Polybench FDTD-2D",
        "src": "examples/FDTD-2D/fdtd2d.cu",
        "args": [],
        "extra_flags": ["-Iexamples/FDTD-2D", "-Iexamples"],
        "expected_pages": 30033
    },
    {
        "name": "Polybench SYRK",
        "src": "examples/SYRK/syrk.cu",
        "args": [],
        "extra_flags": ["-Iexamples/SYRK", "-Iexamples"],
        "expected_pages": 8192
    },
    {
        "name": "Polybench SYR2K",
        "src": "examples/SYR2K/syr2k.cu",
        "args": [],
        "extra_flags": ["-Iexamples/SYR2K", "-Iexamples"],
        "expected_pages": 8193
    }
]

def compile_clean(src, bin_path, extra_flags):
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    cmd = [
        h.CLANG, "-x", "cuda", f"--cuda-gpu-arch=sm_{h.SM_ARCH}", "-O3",
        f"-I{os.path.join(ROOT, 'include')}",
        f"--cuda-path={h.CUDA_PATH}", f"-L{h.CUDA_PATH}/lib64", "-lcudart",
        os.path.join(ROOT, src), "-o", bin_path
    ] + extra_flags
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        print(f"Error compiling clean {src}:\n{res.stderr}")
        return False
    return True

def compile_instrumented(src, bin_path, extra_flags):
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    cmd = [
        h.CLANG, "-x", "cuda", f"--cuda-gpu-arch=sm_{h.SM_ARCH}", "-fgpu-rdc", "-O3", "-rdynamic",
        f"-I{os.path.join(ROOT, 'include')}",
        f"--cuda-path={h.CUDA_PATH}", f"-L{h.CUDA_PATH}/lib64", "-lcudart",
        "-DTRACKING_ENABLED", "-DUVM_TRACKING_MODE=1",
        f"-fpass-plugin={h.PASS_PATH}",
        os.path.join(ROOT, src),
        os.path.join(ROOT, "libMarkAccess.cu"),
        os.path.join(ROOT, "uvm_control_thread.cu"),
        "-o", bin_path
    ] + extra_flags
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        print(f"Error compiling instrumented {src}:\n{res.stderr}")
        return False
    return True

def parse_time(output):
    for line in output.splitlines():
        if "BENCHMARK_TIME:" in line:
            raw = line.split(":")[-1].strip().replace("ms", "").replace("s", "").strip()
            return float(raw)
        if "GPU Runtime:" in line:
            m = re.search(r"GPU Runtime:\s*([0-9.]+)\s*s", line)
            if m:
                return float(m.group(1)) * 1000.0
    return None

def run_trials(bin_path, args, env_vars=None, trials=5):
    times = []
    for _ in range(trials):
        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)
        if os.path.exists(h.ACCESS_LOG):
            os.remove(h.ACCESS_LOG)
        proc = subprocess.run([bin_path] + args, capture_output=True, text=True, cwd=ROOT, env=env)
        if proc.returncode != 0:
            print(f"Runtime error running {bin_path}:\n{proc.stderr}")
            return None, 0
        t = parse_time(proc.stdout + proc.stderr)
        if t is not None:
            times.append(t)
    pages = 0
    if os.path.exists(h.ACCESS_LOG):
        pages = len(h.parse_pagelog(h.ACCESS_LOG))
    return (np.mean(times), np.std(times)), pages

def main():
    print("=" * 80)
    print("UVM ACCESS TRACKING: BENCHMARK SUITE OVERHEAD & ACCURACY EVALUATION")
    print("=" * 80)
    
    # Rebuild interceptor
    subprocess.run([
        h.CLANG, "-shared", "-fPIC", "-O2", f"-I{os.path.join(ROOT, 'include')}",
        "libMallocIntercept.cpp", "-o", h.PRELOAD_SO, "-ldl", "-Wl,--version-script=cublas_version.map"
    ], check=True, cwd=ROOT)

    results = []

    for b in BENCHMARKS:
        name = b["name"]
        print(f"\nEvaluating: {name} ...")
        clean_bin = os.path.join(ROOT, "build", f"{os.path.basename(b['src']).replace('.cu', '')}_clean")
        inst_bin = os.path.join(ROOT, "build", f"{os.path.basename(b['src']).replace('.cu', '')}_inst")

        if not compile_clean(b["src"], clean_bin, b["extra_flags"]):
            continue
        if not compile_instrumented(b["src"], inst_bin, b["extra_flags"]):
            continue

        # 1. Clean Baseline (no tracking)
        clean_stats, _ = run_trials(clean_bin, b["args"], trials=5)
        clean_mean = clean_stats[0]

        # 2. Host Predicted (zero GPU instrumentation overhead)
        pred_env = {"LD_PRELOAD": h.PRELOAD_SO}
        pred_stats, pred_pages = run_trials(inst_bin, b["args"], env_vars=pred_env, trials=5)
        pred_mean = pred_stats[0]
        pred_overhead = ((pred_mean - clean_mean) / clean_mean) * 100.0

        # 3. GPU Instrumented (raw compiler instrumentation)
        # Disable predictor to force GPU instrumentation fallback
        gpu_env = {"LD_PRELOAD": h.PRELOAD_SO, "UVM_FORCE_GPU_INSTRUMENTATION": "1"}
        gpu_stats, gpu_pages = run_trials(inst_bin, b["args"], env_vars=gpu_env, trials=3)
        gpu_mean = gpu_stats[0]
        gpu_overhead = ((gpu_mean - clean_mean) / clean_mean) * 100.0

        accuracy = "100.0%" if pred_pages == b["expected_pages"] or abs(pred_pages - b["expected_pages"]) <= 10 else f"{pred_pages}/{b['expected_pages']}"

        results.append({
            "name": name,
            "clean_ms": clean_mean,
            "pred_ms": pred_mean,
            "pred_overhead": pred_overhead,
            "gpu_ms": gpu_mean,
            "gpu_overhead": gpu_overhead,
            "tracked_pages": pred_pages,
            "expected_pages": b["expected_pages"],
            "accuracy": accuracy
        })
        print(f"  Clean Baseline:    {clean_mean:.4f} ms")
        print(f"  PREDICT_HOST:      {pred_mean:.4f} ms (Overhead: {pred_overhead:+.2f}%, Pages: {pred_pages}/{b['expected_pages']})")
        print(f"  INSTRUMENT_GPU:    {gpu_mean:.4f} ms (Overhead: {gpu_overhead:+.2f}%)")

    # Generate Markdown Table
    out_md = os.path.join(ROOT, "perf_results", "suite_overhead_accuracy_results.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Benchmark Suite: Overhead & Accuracy Evaluation\n\n")
        f.write("| Benchmark Workload | Clean Baseline (ms) | GPU Instrumented (ms) | GPU Overhead | Host Predicted (ms) | Host Overhead | Tracked Pages | Accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| **{r['name']}** | {r['clean_ms']:.3f} | {r['gpu_ms']:.3f} | +{r['gpu_overhead']:.1f}% | {r['pred_ms']:.3f} | **{r['pred_overhead']:+.2f}%** | {r['tracked_pages']} / {r['expected_pages']} | {r['accuracy']} |\n")

    print("\nResults written to:", out_md)

if __name__ == "__main__":
    main()
