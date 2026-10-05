#!/usr/bin/env python3
import os
import re
import sys
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tests.harness as h

NVBIT_SO = "/home/arush/data/ravi/nvbit_release_x86_64/build/instr_count/instr_count.so"

BENCHMARKS = [
    {
        "name": "CUTLASS GEMM (2048x2048)",
        "src": "examples/cutlass_benchmark.cu",
        "args": ["2048", "1"],
        "extra_flags": [
            "-I/home/arush/data/uv_cache/archive-v0/kM3gMCIGSl1iKbxdStF1c/tilelang/3rdparty/cutlass/include"
        ]
    },
    {
        "name": "Polybench MatrixMul",
        "src": "examples/matrixMul.cu",
        "args": [],
        "extra_flags": []
    },
    {
        "name": "Polybench 2DCONV",
        "src": "examples/2DCONV/2DConvolution.cu",
        "args": [],
        "extra_flags": ["-Iexamples/2DCONV", "-Iexamples"]
    },
    {
        "name": "Polybench 2MM",
        "src": "examples/2MM/2mm.cu",
        "args": [],
        "extra_flags": ["-Iexamples/2MM", "-Iexamples"]
    },
    {
        "name": "Polybench FDTD-2D",
        "src": "examples/FDTD-2D/fdtd2d.cu",
        "args": [],
        "extra_flags": ["-Iexamples/FDTD-2D", "-Iexamples"]
    },
    {
        "name": "Polybench SYRK",
        "src": "examples/SYRK/syrk.cu",
        "args": [],
        "extra_flags": ["-Iexamples/SYRK", "-Iexamples"]
    }
]

def compile_nvcc(src, bin_path, extra_flags):
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    cmd = [
        "/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_86",
        f"-I{os.path.join(ROOT, 'include')}",
        os.path.join(ROOT, src), "-o", bin_path
    ] + extra_flags
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        print(f"Error compiling nvcc {src}:\n{res.stderr}")
        return False
    return True

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
            try:
                return float(raw)
            except ValueError:
                pass
        if "GPU Runtime:" in line:
            m = re.search(r"GPU Runtime:\s*([0-9.]+)\s*s", line)
            if m:
                return float(m.group(1)) * 1000.0
    return None

def run_trial(bin_path, args, env_vars=None, timeout=60):
    env = dict(os.environ)
    if env_vars:
        env.update(env_vars)
    if os.path.exists(h.ACCESS_LOG):
        os.remove(h.ACCESS_LOG)
    try:
        proc = subprocess.run([bin_path] + args, capture_output=True, text=True, cwd=ROOT, env=env, timeout=timeout)
        if proc.returncode != 0:
            print(f"Runtime error running {bin_path}:\n{proc.stderr}")
            return None
        return parse_time(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        print(f"Timeout ({timeout}s) running {bin_path}")
        return None

def main():
    print("=" * 90)
    print("EMPIRICAL COMPARISON: 3 ACCESS TRACKING SCENARIOS ACROSS BENCHMARKS")
    print("=" * 90)

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
        nvcc_bin = os.path.join(ROOT, "build", f"{os.path.basename(b['src']).replace('.cu', '')}_nvcc")

        if not compile_clean(b["src"], clean_bin, b["extra_flags"]):
            continue
        if not compile_instrumented(b["src"], inst_bin, b["extra_flags"]):
            continue
        if not compile_nvcc(b["src"], nvcc_bin, b["extra_flags"]):
            continue

        # 1. Baseline Clean
        clean_times = [run_trial(clean_bin, b["args"], timeout=10) for _ in range(3)]
        clean_times = [t for t in clean_times if t is not None]
        clean_ms = np.mean(clean_times) if clean_times else None
        if clean_ms is None:
            continue

        # 2. Scenario 1: NVBit Dynamic SASS Instrumentation
        nvbit_env = {
            "LD_PRELOAD": NVBIT_SO,
            "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
            "NVDISASM": "/usr/local/cuda/bin/nvdisasm"
        }
        nvbit_ms = run_trial(nvcc_bin, b["args"], env_vars=nvbit_env, timeout=45)
        nvbit_overhead = ((nvbit_ms - clean_ms) / clean_ms) * 100.0 if nvbit_ms else None

        # 3. Scenario 2: Compiler Instrumented (No Host Prediction)
        gpu_env = {"LD_PRELOAD": h.PRELOAD_SO, "UVM_FORCE_GPU_INSTRUMENTATION": "1"}
        gpu_times = [run_trial(inst_bin, b["args"], env_vars=gpu_env, timeout=15) for _ in range(2)]
        gpu_times = [t for t in gpu_times if t is not None]
        gpu_ms = np.mean(gpu_times) if gpu_times else None
        gpu_overhead = ((gpu_ms - clean_ms) / clean_ms) * 100.0 if gpu_ms else None

        # 4. Scenario 3: Host Predicted Preferred
        pred_env = {"LD_PRELOAD": h.PRELOAD_SO}
        pred_times = [run_trial(inst_bin, b["args"], env_vars=pred_env, timeout=10) for _ in range(3)]
        pred_times = [t for t in pred_times if t is not None]
        pred_ms = np.mean(pred_times) if pred_times else None
        pred_overhead = ((pred_ms - clean_ms) / clean_ms) * 100.0 if pred_ms else None

        results.append({
            "name": name,
            "clean_ms": clean_ms,
            "nvbit_ms": nvbit_ms,
            "nvbit_overhead": nvbit_overhead,
            "gpu_ms": gpu_ms,
            "gpu_overhead": gpu_overhead,
            "pred_ms": pred_ms,
            "pred_overhead": pred_overhead
        })

        print(f"  Clean Baseline:                         {clean_ms:.3f} ms")
        if nvbit_ms:
            print(f"  [Scenario 1] NVBit SASS Instrumentation: {nvbit_ms:.3f} ms (Overhead: +{nvbit_overhead:.1f}%)")
        else:
            print(f"  [Scenario 1] NVBit SASS Instrumentation: Timeout (> 45s)")
        print(f"  [Scenario 2] Compiler GPU Instrumented: {gpu_ms:.3f} ms (Overhead: +{gpu_overhead:.1f}%)")
        print(f"  [Scenario 3] Host-Predicted Preferred:  {pred_ms:.3f} ms (Overhead: {pred_overhead:+.2f}%)")

    # Generate Markdown Table
    out_md = os.path.join(ROOT, "perf_results", "scenario_comparison_results.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Empirical Comparison: 3 Access Tracking Scenarios\n\n")
        f.write("| Benchmark Workload | Clean Baseline | Scenario 1: NVBit (SASS) | Scenario 1 Overhead | Scenario 2: Compiler (GPU Inst) | Scenario 2 Overhead | Scenario 3: Host Prediction | Scenario 3 Overhead |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            nvbit_str = f"{r['nvbit_ms']:.2f} ms" if r['nvbit_ms'] else "Timeout (>45s)"
            nvbit_ov_str = f"+{r['nvbit_overhead']:.1f}%" if r['nvbit_overhead'] else "> +10,000%"
            f.write(f"| **{r['name']}** | {r['clean_ms']:.3f} ms | {nvbit_str} | {nvbit_ov_str} | {r['gpu_ms']:.3f} ms | +{r['gpu_overhead']:.1f}% | **{r['pred_ms']:.3f} ms** | **{r['pred_overhead']:+.2f}%** |\n")

    print("\n" + "=" * 90)
    print("Results successfully written to:", out_md)
    print("=" * 90)

if __name__ == "__main__":
    main()
