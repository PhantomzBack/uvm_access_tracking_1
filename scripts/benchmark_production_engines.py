#!/usr/bin/env python3
"""
Benchmark Production Open Compute Engines:
1. llama.cpp (LLM Inference: Qwen-2.5-Coder-7B-Instruct Q4_K_M)
2. PyTorch TorchVision (Computer Vision: ResNet-50 CNN Batch Inference)

Measures:
  1. Clean Baseline Runtime & Throughput
  2. Host Static Prediction Runtime & Overhead
  3. Static + Tripwire Runtime & Overhead
  4. Temporal Granularity (Milestones/Launch, Delta-t between trips, Delta-Memory between trips)
  5. Spatial Unawareness Stride (% of total memory)
"""
import os
import re
import sys
import json
import time
import subprocess
import numpy as np

ROOT = "/home/arush/data/ravi/uvm_access_tracking_1"
PRELOAD_SO = os.path.join(ROOT, "libMallocIntercept.so")
LLAMA_CLI = "/home/arush/data/ravi/llama.cpp/build/bin/llama-cli"
GGUF_MODEL = "/home/arush/data/models_gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
PYTHON = "/home/arush/data/vllm/nuvllm/.venv/bin/python"

def run_llama_benchmark(num_tokens=32, trials=3):
    print(f"\n=======================================================")
    print(f" BENCHMARKING: llama.cpp (Qwen 2.5 Coder 7B, n={num_tokens})")
    print(f"=======================================================")
    
    cmd_base = [
        LLAMA_CLI,
        "-m", GGUF_MODEL,
        "-p", "Explain merge sort in Python with quick analysis",
        "-n", str(num_tokens),
        "-ngl", "99",
        "--no-warmup",
        "--single-turn",
        "--simple-io"
    ]
    
    def parse_llama_output(stdout):
        prompt_ts = None
        gen_ts = None
        m_prompt = re.search(r"Prompt:\s*([\d.]+)\s*t/s", stdout)
        m_gen = re.search(r"Generation:\s*([\d.]+)\s*t/s", stdout)
        if m_prompt: prompt_ts = float(m_prompt.group(1))
        if m_gen: gen_ts = float(m_gen.group(1))
        return prompt_ts, gen_ts

    # 1. Clean Baseline
    print("  [1/3] Running Clean Baseline ...", flush=True)
    clean_prompts, clean_gens = [], []
    for _ in range(trials):
        res = subprocess.run(cmd_base, capture_output=True, text=True)
        p, g = parse_llama_output(res.stdout + res.stderr)
        if p and g:
            clean_prompts.append(p)
            clean_gens.append(g)
    med_clean_gen = float(np.median(clean_gens)) if clean_gens else 0.0
    med_clean_prompt = float(np.median(clean_prompts)) if clean_prompts else 0.0
    clean_latency_ms = (num_tokens / med_clean_gen) * 1000.0 if med_clean_gen > 0 else 0.0

    # 2. Host Static Prediction
    print("  [2/3] Running Host Static Prediction ...", flush=True)
    host_env = os.environ.copy()
    host_env["LD_PRELOAD"] = PRELOAD_SO
    host_prompts, host_gens = [], []
    for _ in range(trials):
        res = subprocess.run(cmd_base, env=host_env, capture_output=True, text=True)
        p, g = parse_llama_output(res.stdout + res.stderr)
        if p and g:
            host_prompts.append(p)
            host_gens.append(g)
    med_host_gen = float(np.median(host_gens)) if host_gens else 0.0
    med_host_prompt = float(np.median(host_prompts)) if host_prompts else 0.0
    host_latency_ms = (num_tokens / med_host_gen) * 1000.0 if med_host_gen > 0 else 0.0

    # 3. Static + Tripwire
    print("  [3/3] Running Static + Host Tripwire Tracking ...", flush=True)
    tw_trace_file = "/tmp/llama_tripwire_eval.json"
    if os.path.exists(tw_trace_file): os.remove(tw_trace_file)
    tw_env = os.environ.copy()
    tw_env["LD_PRELOAD"] = PRELOAD_SO
    tw_env["UVM_STATIC_TRIPWIRE"] = "1"
    tw_env["UVM_STATIC_TEMPORAL_OUTPUT"] = tw_trace_file
    tw_prompts, tw_gens = [], []
    for _ in range(trials):
        res = subprocess.run(cmd_base, env=tw_env, capture_output=True, text=True)
        p, g = parse_llama_output(res.stdout + res.stderr)
        if p and g:
            tw_prompts.append(p)
            tw_gens.append(g)
    med_tw_gen = float(np.median(tw_gens)) if tw_gens else 0.0
    med_tw_prompt = float(np.median(tw_prompts)) if tw_prompts else 0.0
    tw_latency_ms = (num_tokens / med_tw_gen) * 1000.0 if med_tw_gen > 0 else 0.0

    # Analyze Tripwire Trace
    total_launches = 0
    avg_delta_us = 0.0
    avg_slice_bytes = 0
    unique_mem_bytes = 0
    if os.path.exists(tw_trace_file):
        with open(tw_trace_file) as f:
            tj = json.load(f)
        evs = tj.get("events", [])
        total_launches = len(evs)
        deltas = []
        slice_bytes = []
        bases = set()
        for i in range(1, len(evs)):
            dt = evs[i]["rel_time_us"] - evs[i-1]["rel_time_us"]
            if 0 < dt < 100000: # filter warmup pauses
                deltas.append(dt)
            sb = evs[i].get("static_slice_bytes", 0)
            if sb > 0:
                slice_bytes.append(sb)
            base = evs[i].get("static_slice_base")
            if base and base != "0x0":
                bases.add(base)
        if deltas: avg_delta_us = float(np.median(deltas))
        if slice_bytes: avg_slice_bytes = int(np.mean(slice_bytes))
        # 7B Q4_K_M model footprint is 4.4 GB
        unique_mem_bytes = 4400 * 1024 * 1024

    host_ov = ((host_latency_ms - clean_latency_ms) / clean_latency_ms * 100.0) if clean_latency_ms > 0 else 0.0
    tw_ov = ((tw_latency_ms - clean_latency_ms) / clean_latency_ms * 100.0) if clean_latency_ms > 0 else 0.0
    avg_slice_pages = (avg_slice_bytes + 4095) // 4096
    total_pages = unique_mem_bytes // 4096
    unawareness_pct = (avg_slice_bytes / unique_mem_bytes * 100.0) if unique_mem_bytes > 0 else 0.0

    res_dict = {
        "workload": "llama.cpp (Qwen-2.5-Coder-7B Q4_K_M)",
        "domain": "LLM Generative Inference",
        "clean_throughput_tok_s": med_clean_gen,
        "clean_latency_ms": clean_latency_ms,
        "host_throughput_tok_s": med_host_gen,
        "host_latency_ms": host_latency_ms,
        "host_overhead_pct": host_ov,
        "tw_throughput_tok_s": med_tw_gen,
        "tw_latency_ms": tw_latency_ms,
        "tw_overhead_pct": tw_ov,
        "total_memory_mb": unique_mem_bytes / (1024 * 1024),
        "total_pages": total_pages,
        "total_launches": total_launches,
        "milestones_per_token": total_launches / float(num_tokens),
        "avg_delta_us": avg_delta_us,
        "delta_mem_kb": avg_slice_bytes / 1024.0,
        "delta_mem_pages": avg_slice_pages,
        "unawareness_pct": unawareness_pct,
    }
    print(f"  -> llama.cpp Clean: {med_clean_gen:.1f} tok/s ({clean_latency_ms:.1f} ms)")
    print(f"  -> llama.cpp Static: {med_host_gen:.1f} tok/s ({host_latency_ms:.1f} ms, {host_ov:+.1f}%)")
    print(f"  -> llama.cpp Tripwire: {med_tw_gen:.1f} tok/s ({tw_latency_ms:.1f} ms, {tw_ov:+.1f}%)")
    print(f"  -> Granularity: {total_launches} trips, {avg_delta_us:.1f} us delta-t, {avg_slice_bytes/1024/1024:.1f} MB/trip ({unawareness_pct:.2f}% unawareness)")
    return res_dict

def run_pytorch_resnet_benchmark(batch_size=32, num_passes=30, trials=3):
    print(f"\n=======================================================")
    print(f" BENCHMARKING: PyTorch TorchVision (ResNet-50, batch={batch_size})")
    print(f"=======================================================")
    
    runner_code = f"""
import torch
import torchvision.models as models
import time

model = models.resnet50().eval().cuda()
inp = torch.randn({batch_size}, 3, 224, 224, device="cuda")

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(inp)
torch.cuda.synchronize()

# Timed run
start = time.perf_counter()
with torch.no_grad():
    for _ in range({num_passes}):
        _ = model(inp)
torch.cuda.synchronize()
dur_ms = (time.perf_counter() - start) / {num_passes} * 1000.0
print(f"RESNET_TIME: {{dur_ms:.4f}} ms")
"""

    def run_sub(env_extra=None):
        env = os.environ.copy()
        if env_extra: env.update(env_extra)
        times = []
        for _ in range(trials):
            res = subprocess.run([PYTHON, "-c", runner_code], env=env, capture_output=True, text=True)
            m = re.search(r"RESNET_TIME:\s*([\d.]+)\s*ms", res.stdout + res.stderr)
            if m:
                times.append(float(m.group(1)))
        return float(np.median(times)) if times else 0.0

    print("  [1/3] Running Clean Baseline ...", flush=True)
    clean_ms = run_sub()

    print("  [2/3] Running Host Static Prediction ...", flush=True)
    host_ms = run_sub({"LD_PRELOAD": PRELOAD_SO})

    print("  [3/3] Running Static + Host Tripwire Tracking ...", flush=True)
    tw_trace_file = "/tmp/torch_resnet_eval.json"
    if os.path.exists(tw_trace_file): os.remove(tw_trace_file)
    tw_ms = run_sub({
        "LD_PRELOAD": PRELOAD_SO,
        "UVM_STATIC_TRIPWIRE": "1",
        "UVM_STATIC_TEMPORAL_OUTPUT": tw_trace_file
    })

    # Analyze Tripwire Trace
    total_launches = 0
    avg_delta_us = 0.0
    avg_slice_bytes = 0
    unique_mem_bytes = 148 * 1024 * 1024 # ~102 MB weights + ~46 MB activations
    if os.path.exists(tw_trace_file):
        with open(tw_trace_file) as f:
            tj = json.load(f)
        evs = tj.get("events", [])
        total_launches = len(evs) // num_passes if num_passes > 0 else len(evs)
        deltas = []
        slice_bytes = []
        for i in range(1, len(evs)):
            dt = evs[i]["rel_time_us"] - evs[i-1]["rel_time_us"]
            if 0 < dt < 5000: # normal kernel launch gap
                deltas.append(dt)
            sb = evs[i].get("static_slice_bytes", 0)
            if sb > 0:
                slice_bytes.append(sb)
        if deltas: avg_delta_us = float(np.median(deltas))
        if slice_bytes: avg_slice_bytes = int(np.mean(slice_bytes))

    host_ov = ((host_ms - clean_ms) / clean_ms * 100.0) if clean_ms > 0 else 0.0
    tw_ov = ((tw_ms - clean_ms) / clean_ms * 100.0) if clean_ms > 0 else 0.0
    avg_slice_pages = (avg_slice_bytes + 4095) // 4096
    total_pages = unique_mem_bytes // 4096
    unawareness_pct = (avg_slice_bytes / unique_mem_bytes * 100.0) if unique_mem_bytes > 0 else 0.0

    res_dict = {
        "workload": f"PyTorch TorchVision (ResNet-50 CNN b={batch_size})",
        "domain": "Computer Vision Inference",
        "clean_latency_ms": clean_ms,
        "host_latency_ms": host_ms,
        "host_overhead_pct": host_ov,
        "tw_latency_ms": tw_ms,
        "tw_overhead_pct": tw_ov,
        "total_memory_mb": unique_mem_bytes / (1024 * 1024),
        "total_pages": total_pages,
        "total_launches": total_launches,
        "milestones_per_pass": total_launches,
        "avg_delta_us": avg_delta_us,
        "delta_mem_kb": avg_slice_bytes / 1024.0,
        "delta_mem_pages": avg_slice_pages,
        "unawareness_pct": unawareness_pct,
    }
    print(f"  -> ResNet-50 Clean: {clean_ms:.2f} ms")
    print(f"  -> ResNet-50 Static: {host_ms:.2f} ms ({host_ov:+.1f}%)")
    print(f"  -> ResNet-50 Tripwire: {tw_ms:.2f} ms ({tw_ov:+.1f}%)")
    print(f"  -> Granularity: {total_launches} trips/pass, {avg_delta_us:.1f} us delta-t, {avg_slice_bytes/1024/1024:.1f} MB/trip ({unawareness_pct:.2f}% unawareness)")
    return res_dict

def main():
    print("=" * 80)
    print("EVALUATING STATIC ANALYSIS + TRIPWIRE METHOD ON PRODUCTION ENGINES")
    print("=" * 80)
    
    results = []
    
    # 1. llama.cpp LLM Inference
    r_llama = run_llama_benchmark(num_tokens=32, trials=3)
    results.append(r_llama)
    
    # 2. PyTorch TorchVision ResNet-50
    r_resnet = run_pytorch_resnet_benchmark(batch_size=32, num_passes=20, trials=3)
    results.append(r_resnet)

    out_json = os.path.join(ROOT, "perf_results", "production_engines_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved production benchmark results to {out_json}")

if __name__ == "__main__":
    main()
