#!/usr/bin/env python3
"""
run_llama_cpp_benchmark.py:
Automated 3-Way Comparative Benchmark for llama.cpp:
1. Native Offload (partial layers fitting physical VRAM budget)
2. Vanilla UVM Demand Paging (GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, all layers offloaded)
3. Synchronous Prefetcher (UVM_PREFETCH_SYNC=1, GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, all layers offloaded)
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LLAMA_CLI_BIN = "/home/arush/data/ravi/llama.cpp/build/bin/llama-cli"
INTERCEPTOR_LIB = "/home/arush/data/ravi/uvm_access_tracking_1/libMallocIntercept.so"
FAULT_STATS_PATH = "/proc/driver/nvidia-uvm/cpu/fault_stats"

cudart = ctypes.CDLL("libcudart.so")


def get_gpu_vram():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    err = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if err != 0:
        raise RuntimeError(f"cudaMemGetInfo failed: code {err}")
    return free.value, total.value


class VRAMBudgetController:
    """Pins physical VRAM with cudaMalloc to enforce a target free VRAM budget for UVM testing."""
    def __init__(self, target_free_gib=None):
        self.target_free_gib = target_free_gib
        self.ptrs = []

    def __enter__(self):
        if self.target_free_gib is None or self.target_free_gib <= 0:
            return self

        free, total = get_gpu_vram()
        target_free_bytes = int(self.target_free_gib * 1024 * 1024 * 1024)
        bytes_to_hold = free - target_free_bytes

        if bytes_to_hold > 0:
            chunk_size = 1024 * 1024 * 1024  # 1 GiB chunks
            while bytes_to_hold > chunk_size:
                p = ctypes.c_void_p()
                err = cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk_size))
                if err != 0:
                    break
                self.ptrs.append(p)
                bytes_to_hold -= chunk_size

            if bytes_to_hold > 0:
                p = ctypes.c_void_p()
                err = cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(bytes_to_hold))
                if err == 0:
                    self.ptrs.append(p)

            new_free, _ = get_gpu_vram()
            print(f"[VRAM Controller] Pinned {len(self.ptrs)} GiB chunks. GPU Free VRAM: {new_free / 1e9:.2f} GB (Target: {self.target_free_gib:.2f} GB)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.ptrs:
            cudart.cudaFree(p)
        self.ptrs.clear()
        free, _ = get_gpu_vram()
        print(f"[VRAM Controller] Released pinned VRAM. GPU Free VRAM: {free / 1e9:.2f} GB")


def get_uvm_page_faults():
    if not os.path.exists(FAULT_STATS_PATH):
        return 0
    try:
        with open(FAULT_STATS_PATH, "r") as f:
            for line in f:
                if "uvm_gpu_faults" in line or "vm_faults" in line:
                    parts = line.strip().split()
                    return int(parts[-1])
    except Exception as e:
        print(f"[Warning] Failed to read fault stats: {e}")
    return 0


def parse_llama_perf(output):
    """
    Parses llama-cli performance timing printouts:
    llama_perf_context_print: prompt eval time =    248.56 ms /    32 tokens (    7.77 ms per token,   128.74 tokens per second)
    llama_perf_context_print:        eval time =   1428.12 ms /   128 runs   (   11.16 ms per token,    89.63 tokens per second)
    llama_perf_context_print:       total time =   1676.68 ms /   160 tokens
    """
    metrics = {
        "prefill_ms": 0.0,
        "prefill_tokens": 0,
        "prefill_tps": 0.0,
        "decode_ms": 0.0,
        "decode_tokens": 0,
        "decode_tps": 0.0,
        "total_ms": 0.0,
        "total_tokens": 0,
        "output_text": "",
    }

    prefill_match = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d\.]+)\s*tokens per second", output)
    if prefill_match:
        metrics["prefill_ms"] = float(prefill_match.group(1))
        metrics["prefill_tokens"] = int(prefill_match.group(2))
        metrics["prefill_tps"] = float(prefill_match.group(3))

    decode_match = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d\.]+)\s*tokens per second", output)
    if decode_match:
        metrics["decode_ms"] = float(decode_match.group(1))
        metrics["decode_tokens"] = int(decode_match.group(2))
        metrics["decode_tps"] = float(decode_match.group(3))

    total_match = re.search(r"total time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if total_match:
        metrics["total_ms"] = float(total_match.group(1))
        metrics["total_tokens"] = int(total_match.group(2))

    summary_match = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
    if summary_match:
        metrics["prefill_tps"] = float(summary_match.group(1))
        metrics["decode_tps"] = float(summary_match.group(2))

    prefetch_calls_match = re.search(r"Total Prefetches:\s*(\d+)\s*calls", output)
    if prefetch_calls_match:
        metrics["prefetch_calls"] = int(prefetch_calls_match.group(1))
    prefetch_mb_match = re.search(r"Total Data Prefetched:\s*([\d\.]+)\s*MB", output)
    if prefetch_mb_match:
        metrics["prefetch_mb"] = float(prefetch_mb_match.group(1))

    return metrics


def run_single_benchmark(mode, model_path, prompt, n_predict, vram_budget_gib, ngl_native, disable_graphs=False):
    print(f"\n{'='*70}")
    print(f"RUNNING BENCHMARK: Mode={mode.upper()} | VRAM Budget={vram_budget_gib} GiB | N_Predict={n_predict} | DisableGraphs={disable_graphs}")
    print(f"{'='*70}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if disable_graphs:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    # Determine command and environment
    if mode == "native":
        # Standard CPU/GPU split without UVM
        cmd = [
            LLAMA_CLI_BIN,
            "-m", model_path,
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", str(ngl_native),
            "-c", "2048",
            "--temp", "0.0",
            "--simple-io",
            "-st",  # single turn non-interactive
            "--perf",
        ]
        env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
        env.pop("LD_PRELOAD", None)
        env.pop("UVM_PREFETCH_SYNC", None)

    elif mode == "uvm_demand":
        # All layers on GPU via UVM, vanilla demand paging
        cmd = [
            LLAMA_CLI_BIN,
            "-m", model_path,
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", "999",
            "-c", "2048",
            "--temp", "0.0",
            "--simple-io",
            "-st",
            "--perf",
        ]
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        env.pop("LD_PRELOAD", None)
        env.pop("UVM_PREFETCH_SYNC", None)

    elif mode == "uvm_prefetch":
        # All layers on GPU via UVM + Synchronous Prefetcher
        cmd = [
            LLAMA_CLI_BIN,
            "-m", model_path,
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", "999",
            "-c", "2048",
            "--temp", "0.0",
            "--simple-io",
            "-st",
            "--perf",
        ]
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        env["LD_PRELOAD"] = INTERCEPTOR_LIB
        env["UVM_PREFETCH_SYNC"] = "1"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    faults_before = get_uvm_page_faults()

    with VRAMBudgetController(vram_budget_gib):
        start_wall = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        end_wall = time.perf_counter()

    faults_after = get_uvm_page_faults()
    faults_delta = max(0, faults_after - faults_before)
    wall_duration = end_wall - start_wall

    output = proc.stdout
    metrics = parse_llama_perf(output)
    metrics["mode"] = mode
    metrics["wall_duration_s"] = wall_duration
    metrics["page_faults"] = faults_delta
    metrics["returncode"] = proc.returncode

    # Extract raw generation text
    metrics["raw_output"] = output

    print(f"[{mode.upper()} DONE] Return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"[{mode.upper()} ERROR OUTPUT]\n{output[-1000:]}")
    print(f"  Wall Duration:     {wall_duration:.2f} s")
    print(f"  Prefill Latency:   {metrics['prefill_ms']:.2f} ms ({metrics['prefill_tps']:.2f} tok/s, {metrics['prefill_tokens']} tokens)")
    print(f"  Decode Latency:    {metrics['decode_ms']:.2f} ms ({metrics['decode_tps']:.2f} tok/s, {metrics['decode_tokens']} tokens)")
    print(f"  Total Latency:     {metrics['total_ms']:.2f} ms ({metrics['total_tokens']} tokens)")
    print(f"  UVM Page Faults:   {faults_delta:,}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="llama.cpp UVM Comparative Benchmark")
    parser.add_argument("--model", type=str, default="/home/arush/data/models_gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    parser.add_argument("--prompt", type=str, default="Write a complete Python implementation of an asynchronous worker pool with priority task queues and worker heartbeat monitoring.")
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--vram-budget-gib", type=float, default=2.0)
    parser.add_argument("--ngl-native", type=int, default=6)
    parser.add_argument("--modes", type=str, default="native,uvm_demand,uvm_prefetch")
    parser.add_argument("--disable-graphs", action="store_true", help="Disable CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1)")
    parser.add_argument("--output-report", type=str, default="/home/arush/data/ravi/uvm_access_tracking_1/perf_results/llama_cpp_uvm_comparison.md")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}

    for mode in modes:
        results[mode] = run_single_benchmark(
            mode=mode,
            model_path=args.model,
            prompt=args.prompt,
            n_predict=args.n_predict,
            vram_budget_gib=args.vram_budget_gib,
            ngl_native=args.ngl_native,
            disable_graphs=args.disable_graphs,
        )
        time.sleep(2)  # Cool down between runs

    # Generate Markdown Report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model).stem
    demand_res = results.get("uvm_demand")
    prefetch_res = results.get("uvm_prefetch")
    native_res = results.get("native")

    lines = []
    lines.append(f"# Comparative Benchmark: llama.cpp UVM Performance Evaluation")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** `{model_name}`")
    lines.append(f"**Target Physical VRAM Budget:** `{args.vram_budget_gib} GiB`")
    lines.append(f"**Tokens Generated:** `{args.n_predict}` (Temperature: `0.0`)")
    lines.append("")
    lines.append("## 1. Executive Performance Summary")
    lines.append("")
    lines.append(f"| Metric | 1. Native Offload (`-ngl {args.ngl_native}`) | 2. Vanilla UVM Demand Paging | 3. Synchronous Prefetcher (`UVM_PREFETCH_SYNC=1`) | Prefetcher Speedup vs Demand |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    def val(m, key, fmt="{:.2f}"):
        if m and key in m:
            return fmt.format(m[key])
        return "N/A"

    nat_pf_tps = val(native_res, "prefill_tps")
    dem_pf_tps = val(demand_res, "prefill_tps")
    pre_pf_tps = val(prefetch_res, "prefill_tps")

    nat_dec_tps = val(native_res, "decode_tps")
    dem_dec_tps = val(demand_res, "decode_tps")
    pre_dec_tps = val(prefetch_res, "decode_tps")

    nat_dec_ms = val(native_res, "decode_ms")
    dem_dec_ms = val(demand_res, "decode_ms")
    pre_dec_ms = val(prefetch_res, "decode_ms")

    nat_tot_s = val(native_res, "wall_duration_s")
    dem_tot_s = val(demand_res, "wall_duration_s")
    pre_tot_s = val(prefetch_res, "wall_duration_s")

    nat_faults = f"{native_res['page_faults']:,}" if native_res else "N/A"
    dem_faults = f"{demand_res['page_faults']:,}" if demand_res else "N/A"
    pre_faults = f"{prefetch_res['page_faults']:,}" if prefetch_res else "N/A"

    speedup_decode = "N/A"
    speedup_total = "N/A"
    if demand_res and prefetch_res:
        if demand_res.get("decode_tps", 0) > 0 and prefetch_res.get("decode_tps", 0) > 0:
            speedup_decode = f"{prefetch_res['decode_tps'] / demand_res['decode_tps']:.2f}x"
        if demand_res.get("wall_duration_s", 0) > 0 and prefetch_res.get("wall_duration_s", 0) > 0:
            speedup_total = f"{demand_res['wall_duration_s'] / prefetch_res['wall_duration_s']:.2f}x"

    lines.append(f"| **Prefill Speed (tok/s)** | {nat_pf_tps} | {dem_pf_tps} | **{pre_pf_tps}** | - |")
    lines.append(f"| **Decode Speed (tok/s)** | {nat_dec_tps} | {dem_dec_tps} | **{pre_dec_tps}** | **{speedup_decode}** |")
    lines.append(f"| **Decode Latency (ms)** | {nat_dec_ms} | {dem_dec_ms} | **{pre_dec_ms}** | - |")
    lines.append(f"| **Total Wall Duration (s)** | {nat_tot_s} | {dem_tot_s} | **{pre_tot_s}** | **{speedup_total}** |")
    lines.append(f"| **Hardware UVM Page Faults** | {nat_faults} | {dem_faults} | **{pre_faults}** | - |")
    lines.append("")

    lines.append("## 2. Detailed Breakdown")
    for mode in modes:
        res = results.get(mode)
        if not res:
            continue
        lines.append(f"### Mode: `{mode}`")
        lines.append(f"- Prefill Speed: {res.get('prefill_tps', 0):.2f} tok/s")
        lines.append(f"- Decode Speed: {res.get('decode_tps', 0):.2f} tok/s")
        lines.append(f"- Total Wall Time: {res.get('wall_duration_s', 0):.2f} s")
        lines.append(f"- Hardware Page Faults: {res.get('page_faults', 0):,}")
        if "prefetch_calls" in res:
            lines.append(f"- Prefetch Calls Issued: {res['prefetch_calls']:,}")
        if "prefetch_mb" in res:
            lines.append(f"- Data Prefetched: {res['prefetch_mb']:.2f} MB")
        lines.append("")

    report_content = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\n[REPORT SAVED] {report_path}")


if __name__ == "__main__":
    main()
