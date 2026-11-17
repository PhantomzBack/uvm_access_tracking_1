#!/usr/bin/env python3
"""
vLLM UVM Oversubscription Comparator UI & Benchmark Tool
-------------------------------------------------------
Enables side-by-side or single-mode testing of vLLM serving on oversubscribed GPU memory:
  - Mode 1: Vanilla UVM Demand Paging (Hardware MMU Faults across PCIe)
  - Mode 2: UVM with Our Synchronous Static-Guided Prefetcher (Zero / Low Faults)

Supports both:
  1. Synthetic Throughput Benchmarking (vllm bench throughput)
  2. Interactive Custom Prompt Generation (arbitrary custom prompts, live text output & metrics)

Both modes execute with the TRITON Attention backend (--attention-backend TRITON_ATTN).

Usage:
  # Launch the Web UI (browser-based dashboard on port 8080):
  python vllm_uvm_comparator_ui.py --web --port 8080

  # Or run in interactive terminal CLI mode:
  python vllm_uvm_comparator_ui.py --cli
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from typing import AsyncGenerator, Dict, Any

# Ensure working environment paths
VENV_PYTHON = "/home/arush/data/vllm/nuvllm/.venv/bin/python"
VENV_VLLM = "/home/arush/data/vllm/nuvllm/.venv/bin/vllm"
LIB_INTERCEPT = "/home/arush/data/ravi/uvm_access_tracking_1/libMallocIntercept.so"
HF_CACHE = "/home/arush/data/huggingface_cache"
FAULT_STATS_PATH = "/proc/driver/nvidia-uvm/cpu/fault_stats"
RUN_PROMPT_SCRIPT = "/home/arush/data/ravi/uvm_access_tracking_1/scripts/run_custom_prompt.py"


def read_fault_stats() -> Dict[str, int]:
    """Reads hardware UVM fault statistics from /proc/driver/nvidia-uvm/cpu/fault_stats."""
    stats = {"cpu_faults": 0, "gpu_faults": 0, "proactive_prefetches": 0}
    if not os.path.exists(FAULT_STATS_PATH):
        return stats
    try:
        with open(FAULT_STATS_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    k, v = parts[0].strip(), int(parts[1].strip())
                    if "cpu_faults" in k:
                        stats["cpu_faults"] = v
                    elif "gpu_faults" in k:
                        stats["gpu_faults"] = v
                    elif "proactive_prefetches" in k:
                        stats["proactive_prefetches"] = v
    except Exception:
        pass
    return stats


def reset_fault_stats() -> None:
    """Attempts to reset fault counters to zero."""
    if os.path.exists(FAULT_STATS_PATH):
        try:
            with open(FAULT_STATS_PATH, "w") as f:
                f.write("0\n")
        except Exception:
            pass


def run_benchmark_subprocess(
    mode: str,
    num_prompts: int = 20,
    input_len: int = 512,
    output_len: int = 64,
    gpu_budget_gb: float = 2.0,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> Dict[str, Any]:
    """
    Executes vllm bench throughput under either:
      - mode='demand_paging': Vanilla UVM demand paging
      - mode='prefetcher': UVM with libMallocIntercept synchronous prefetching
    """
    reset_fault_stats()
    before_faults = read_fault_stats()

    env = os.environ.copy()
    env["HF_HOME"] = HF_CACHE
    env["VLLM_USE_UVM"] = "3"
    env["VLLM_UVM_GPU_BUDGET_GB"] = str(gpu_budget_gb)

    if mode == "prefetcher":
        env["LD_PRELOAD"] = LIB_INTERCEPT
        env["UVM_PREFETCH_SYNC"] = "1"
    else:
        env.pop("LD_PRELOAD", None)
        env.pop("UVM_PREFETCH_SYNC", None)
        env["UVM_PREFETCH_ENABLE"] = "0"

    cmd = [
        VENV_VLLM,
        "bench",
        "throughput",
        "--model", model,
        "--attention-backend", "TRITON_ATTN",
        "--max-model-len", "4096",
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--dataset-name", "random",
        "--enforce-eager",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd="/tmp",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    after_faults = read_fault_stats()
    delta_gpu_faults = max(0, after_faults["gpu_faults"] - before_faults["gpu_faults"])

    output = proc.stdout

    throughput = 0.0
    total_tok_s = 0.0
    output_tok_s = 0.0

    match_tp = re.search(r"Throughput:\s*([\d\.]+)\s*requests/s,\s*([\d\.]+)\s*total tokens/s,\s*([\d\.]+)\s*output tokens/s", output)
    if match_tp:
        throughput = float(match_tp.group(1))
        total_tok_s = float(match_tp.group(2))
        output_tok_s = float(match_tp.group(3))

    prefetch_calls = 0
    prefetch_mb = 0.0
    match_calls = re.search(r"Total Prefetches:\s*(\d+)\s*calls", output)
    if match_calls:
        prefetch_calls = int(match_calls.group(1))
    match_mb = re.search(r"Total Data Prefetched:\s*([\d\.]+)\s*MB", output)
    if match_mb:
        prefetch_mb = float(match_mb.group(1))

    return {
        "mode": mode,
        "model": model,
        "backend": "TRITON_ATTN",
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "gpu_budget_gb": gpu_budget_gb,
        "throughput_req_s": throughput,
        "total_tok_s": total_tok_s,
        "output_tok_s": output_tok_s,
        "elapsed_sec": round(elapsed, 2),
        "gpu_page_faults": delta_gpu_faults,
        "prefetch_calls": prefetch_calls,
        "prefetch_mb": prefetch_mb,
        "success": (proc.returncode == 0 and throughput > 0),
        "raw_log_tail": "\n".join(output.strip().splitlines()[-25:]),
    }


def run_custom_prompt_subprocess(
    mode: str,
    prompt: str,
    max_tokens: int = 64,
    batch_size: int = 5,
    gpu_budget_gb: float = 2.0,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> Dict[str, Any]:
    """
    Executes a custom user prompt through vLLM with TRITON attention and UVM oversubscription.
    Captures:
      - Generated text output
      - Generation speed (tokens/sec)
      - Elapsed generation latency (seconds)
      - Hardware GPU page faults
    """
    reset_fault_stats()
    before_faults = read_fault_stats()

    env = os.environ.copy()
    env["HF_HOME"] = HF_CACHE
    env["VLLM_USE_UVM"] = "3"
    env["VLLM_UVM_GPU_BUDGET_GB"] = str(gpu_budget_gb)

    if mode == "prefetcher":
        env["LD_PRELOAD"] = LIB_INTERCEPT
        env["UVM_PREFETCH_SYNC"] = "1"
    else:
        env.pop("LD_PRELOAD", None)
        env.pop("UVM_PREFETCH_SYNC", None)
        env["UVM_PREFETCH_ENABLE"] = "0"

    cmd = [
        VENV_PYTHON,
        RUN_PROMPT_SCRIPT,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--batch-size", str(batch_size),
        "--gpu-budget-gb", str(gpu_budget_gb),
        "--model", model,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd="/tmp",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    after_faults = read_fault_stats()
    delta_gpu_faults = max(0, after_faults["gpu_faults"] - before_faults["gpu_faults"])

    res = {
        "mode": mode,
        "prompt": prompt,
        "sample_output": "",
        "batch_size": batch_size,
        "total_output_tokens": 0,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": 0.0,
        "gpu_page_faults": delta_gpu_faults,
        "success": False,
        "raw_output": proc.stdout,
    }

    if "---JSON_START---" in proc.stdout and "---JSON_END---" in proc.stdout:
        try:
            json_text = proc.stdout.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
            parsed = json.loads(json_text)
            res.update(parsed)
            res["mode"] = mode
            res["success"] = True
        except Exception as e:
            res["error"] = str(e)

    return res


# ==============================================================================
# Terminal CLI Interface
# ==============================================================================

def run_cli_ui():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.prompt import Prompt, IntPrompt
    except ImportError:
        print("[!] Package 'rich' not found. Please install or use default python.")
        return

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]vLLM UVM Oversubscription Comparator[/bold cyan]\n"
        "[dim]Compare Vanilla Demand Paging vs Synchronous Prefetcher (TRITON Attention)[/dim]",
        border_style="cyan"
    ))

    while True:
        console.print("\n[bold yellow]Select Mode:[/bold yellow]")
        console.print("  [1] ⚡ Head-to-Head Comparison (Standard Benchmark: 20 prompts, 512 in / 64 out)")
        console.print("  [2] 🚀 Fast Benchmark (5 prompts, 128 in / 32 out)")
        console.print("  [3] 🐢 Demand Paging Mode Only (Vanilla UVM Benchmark)")
        console.print("  [4] ⚡ Our Prefetcher Mode Only (Synchronous Interceptor Benchmark)")
        console.print("  [5] 🛠️ Custom Benchmark Settings (Prompts, Lengths, GPU Budget)")
        console.print("  [6] 💬 Interactive Custom Prompt (Type custom prompt, see text output & metrics)")
        console.print("  [0] Exit")

        choice = Prompt.ask("Choose option", choices=["0", "1", "2", "3", "4", "5", "6"], default="6")
        if choice == "0":
            console.print("[dim]Exiting comparator.[/dim]")
            break

        # ----------------------------------------------------------------------
        # Choice 6: Interactive Custom Prompt Mode
        # ----------------------------------------------------------------------
        if choice == "6":
            console.print("\n[bold cyan]─── Interactive Custom Prompt Mode ───[/bold cyan]")
            console.print("[dim]Type an arbitrary prompt, run under TRITON attention with UVM oversubscription, and compare generated response text, tokens/s, and hardware faults.[/dim]\n")
            
            custom_prompt = Prompt.ask(
                "[bold]Enter prompt text[/bold]",
                default="def quicksort(arr: list[int]) -> list[int]:\n    \"\"\"Sort list in ascending order.\"\"\""
            )
            batch_size = IntPrompt.ask(
                "Batch count / Concurrency [dim](5-10 recommended to stress KV-cache oversubscription)[/dim]",
                default=5
            )
            max_tokens = IntPrompt.ask("Max output tokens to generate", default=64)
            budget_gb = float(Prompt.ask("GPU Budget in GiB", default="2.0"))

            console.print("\n[bold yellow]Select mode to test:[/bold yellow]")
            console.print("  [1] ⚡ Compare Both (Demand Paging vs Our Prefetcher)")
            console.print("  [2] 🐢 Demand Paging Only")
            console.print("  [3] 🚀 Our Prefetcher Only")
            c_mode = Prompt.ask("Choose", choices=["1", "2", "3"], default="1")

            modes = ["demand_paging", "prefetcher"] if c_mode == "1" else (["demand_paging"] if c_mode == "2" else ["prefetcher"])

            p_results = {}
            for m in modes:
                m_label = "Vanilla Demand Paging" if m == "demand_paging" else "Our Synchronous Prefetcher"
                with console.status(f"[bold green]Generating response with {m_label} (Triton Attention, Budget={budget_gb} GB)...[/bold green]"):
                    res = run_custom_prompt_subprocess(
                        mode=m,
                        prompt=custom_prompt,
                        max_tokens=max_tokens,
                        batch_size=batch_size,
                        gpu_budget_gb=budget_gb,
                    )
                    p_results[m] = res
                console.print(f"[bold]Done {m_label}[/bold]: {res.get('tokens_per_sec', 0):.2f} tok/s, {res.get('gpu_page_faults', 0):,} faults, {res.get('elapsed_sec', 0):.2f}s.")

            # Display generated response text
            for m in modes:
                title_col = "red" if m == "demand_paging" else "green"
                title_lbl = "Demand Paging Generated Output" if m == "demand_paging" else "Our Prefetcher Generated Output"
                output_txt = p_results[m].get("sample_output", "").strip() or "(No output generated)"
                console.print(Panel(
                    output_txt,
                    title=f"[bold {title_col}]{title_lbl}[/bold {title_col}]",
                    border_style=title_col,
                    expand=False
                ))

            # Display comparative table
            table = Table(
                title=f"Custom Prompt Performance (Batch={batch_size}, MaxTokens={max_tokens}, Budget={budget_gb}GB)",
                border_style="cyan"
            )
            table.add_column("Metric", style="bold white")
            if "demand_paging" in p_results:
                table.add_column("Vanilla Demand Paging", justify="right", style="red")
            if "prefetcher" in p_results:
                table.add_column("Our Prefetcher", justify="right", style="green")
            if len(p_results) == 2:
                table.add_column("Improvement", justify="right", style="bold yellow")

            dp = p_results.get("demand_paging", {})
            pf = p_results.get("prefetcher", {})

            def fmt_prow(label, dp_val, pf_val, delta_str=""):
                row = [label]
                if "demand_paging" in p_results:
                    row.append(str(dp_val))
                if "prefetcher" in p_results:
                    row.append(str(pf_val))
                if len(p_results) == 2:
                    row.append(delta_str)
                table.add_row(*row)

            if len(p_results) == 2:
                dp_tok_s = dp.get('tokens_per_sec', 0)
                pf_tok_s = pf.get('tokens_per_sec', 0)
                tok_mult = (pf_tok_s / max(0.001, dp_tok_s)) if dp_tok_s > 0 else 0
                fmt_prow("Generation Speed (tok/s)", f"{dp_tok_s:.2f}", f"{pf_tok_s:.2f}", f"{tok_mult:.2f}x Speedup")

                dp_time = dp.get('elapsed_sec', 0)
                pf_time = pf.get('elapsed_sec', 0)
                time_saved = dp_time - pf_time
                time_pct = (time_saved / max(0.01, dp_time)) * 100
                fmt_prow("Generation Latency", f"{dp_time:.2f} s", f"{pf_time:.2f} s", f"-{time_saved:.2f} s (-{time_pct:.1f}%)")

                dp_faults = dp.get('gpu_page_faults', 0)
                pf_faults = pf.get('gpu_page_faults', 0)
                fault_red = dp_faults - pf_faults
                fault_pct = (fault_red / max(1, dp_faults)) * 100
                fmt_prow("GPU Page Faults", f"{dp_faults:,}", f"{pf_faults:,}", f"-{fault_pct:.1f}% Eliminated")
            else:
                r = list(p_results.values())[0]
                fmt_prow("Generation Speed (tok/s)", f"{r.get('tokens_per_sec', 0):.2f}", "")
                fmt_prow("Generation Latency", f"{r.get('elapsed_sec', 0):.2f} s", "")
                fmt_prow("GPU Page Faults", f"{r.get('gpu_page_faults', 0):,}", "")

            console.print(table)
            continue

        # ----------------------------------------------------------------------
        # Choices 1-5: Throughput Benchmark Mode
        # ----------------------------------------------------------------------
        num_prompts = 20
        in_len = 512
        out_len = 64
        budget_gb = 2.0

        if choice == "2":
            num_prompts = 5
            in_len = 128
            out_len = 32
        elif choice == "5":
            num_prompts = IntPrompt.ask("Number of prompts", default=10)
            in_len = IntPrompt.ask("Input token length", default=256)
            out_len = IntPrompt.ask("Output token length", default=64)
            budget_gb = float(Prompt.ask("GPU Budget in GiB", default="2.0"))

        modes_to_run = []
        if choice in ["1", "2", "5"]:
            modes_to_run = ["demand_paging", "prefetcher"]
        elif choice == "3":
            modes_to_run = ["demand_paging"]
        elif choice == "4":
            modes_to_run = ["prefetcher"]

        results = {}
        for m in modes_to_run:
            m_label = "Vanilla UVM Demand Paging" if m == "demand_paging" else "UVM with Our Prefetcher"
            with console.status(f"[bold green]Running {m_label} (Triton Attention, Budget={budget_gb} GB)...[/bold green]"):
                res = run_benchmark_subprocess(
                    mode=m,
                    num_prompts=num_prompts,
                    input_len=in_len,
                    output_len=out_len,
                    gpu_budget_gb=budget_gb,
                )
                results[m] = res
            console.print(f"[bold]Finished {m_label}[/bold]: {res['throughput_req_s']:.2f} req/s, {res['gpu_page_faults']:,} faults.")

        # Tabulate results
        table = Table(title=f"Benchmark Results (Triton Attention, GPU Budget={budget_gb} GiB)", border_style="cyan")
        table.add_column("Metric", style="bold white")
        if "demand_paging" in results:
            table.add_column("Vanilla Demand Paging", justify="right", style="red")
        if "prefetcher" in results:
            table.add_column("Our Prefetcher", justify="right", style="green")
        if len(results) == 2:
            table.add_column("Improvement", justify="right", style="bold yellow")

        dp = results.get("demand_paging", {})
        pf = results.get("prefetcher", {})

        def fmt_row(label, dp_val, pf_val, delta_str=""):
            row = [label]
            if "demand_paging" in results:
                row.append(str(dp_val))
            if "prefetcher" in results:
                row.append(str(pf_val))
            if len(results) == 2:
                row.append(delta_str)
            table.add_row(*row)

        if len(results) == 2:
            tp_mult = (pf.get('throughput_req_s', 0) / dp.get('throughput_req_s', 1)) if dp.get('throughput_req_s', 0) > 0 else 0
            fmt_row("Throughput (req/s)", f"{dp.get('throughput_req_s', 0):.2f}", f"{pf.get('throughput_req_s', 0):.2f}", f"{tp_mult:.2f}x Speedup")

            out_mult = (pf.get('output_tok_s', 0) / dp.get('output_tok_s', 1)) if dp.get('output_tok_s', 0) > 0 else 0
            fmt_row("Output Tokens/s", f"{dp.get('output_tok_s', 0):.2f}", f"{pf.get('output_tok_s', 0):.2f}", f"{out_mult:.2f}x Speedup")

            tot_mult = (pf.get('total_tok_s', 0) / dp.get('total_tok_s', 1)) if dp.get('total_tok_s', 0) > 0 else 0
            fmt_row("Total Tokens/s", f"{dp.get('total_tok_s', 0):.2f}", f"{pf.get('total_tok_s', 0):.2f}", f"{tot_mult:.2f}x Speedup")

            time_red = dp.get('elapsed_sec', 0) - pf.get('elapsed_sec', 0)
            fmt_row("Elapsed Time", f"{dp.get('elapsed_sec', 0):.2f} s", f"{pf.get('elapsed_sec', 0):.2f} s", f"-{time_red:.2f} s (-{(time_red/max(1, dp.get('elapsed_sec', 1)))*100:.1f}%)")

            fault_red = dp.get('gpu_page_faults', 0) - pf.get('gpu_page_faults', 0)
            fault_pct = (fault_red / max(1, dp.get('gpu_page_faults', 1))) * 100
            fmt_row("GPU Page Faults", f"{dp.get('gpu_page_faults', 0):,}", f"{pf.get('gpu_page_faults', 0):,}", f"-{fault_pct:.1f}% Eliminated")
        else:
            r = list(results.values())[0]
            fmt_row("Throughput (req/s)", f"{r['throughput_req_s']:.2f}", "")
            fmt_row("Output Tokens/s", f"{r['output_tok_s']:.2f}", "")
            fmt_row("Total Tokens/s", f"{r['total_tok_s']:.2f}", "")
            fmt_row("Elapsed Time", f"{r['elapsed_sec']:.2f} s", "")
            fmt_row("GPU Page Faults", f"{r['gpu_page_faults']:,}", "")

        console.print(table)


# ==============================================================================
# Web UI Interface (FastAPI + HTML5 Dashboard)
# ==============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vLLM UVM Oversubscription Comparator</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
    .card { background-color: #131c2e; border: 1px solid #1e293b; border-radius: 0.75rem; }
    .log-terminal { background-color: #050811; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body class="p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="flex items-center justify-between pb-6 border-b border-slate-800">
      <div>
        <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
          vLLM UVM Oversubscription Comparator
        </h1>
        <p class="text-sm text-slate-400 mt-1">
          Real-world comparison: <span class="text-rose-400 font-semibold">Demand Paging (Vanilla UVM)</span> vs. 
          <span class="text-emerald-400 font-semibold">Our Prefetcher (Static-Analysis Guided)</span> on 
          <span class="text-sky-300 font-mono">Qwen/Qwen2.5-Coder-7B</span> under <span class="text-amber-300 font-mono">TRITON Attention</span>.
        </p>
      </div>
      <div class="flex items-center space-x-2">
        <span class="px-3 py-1 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700">
          NVIDIA A40 (48GB)
        </span>
        <span class="px-3 py-1 text-xs font-semibold rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800">
          UVM Budget: 2.0 GiB
        </span>
      </div>
    </div>

    <!-- Mode Tabs -->
    <div class="flex space-x-4 border-b border-slate-800 mt-4">
      <button id="tabBtnCustomPrompt" onclick="switchTab('customPrompt')" class="py-2.5 px-4 font-bold text-sm border-b-2 border-cyan-400 text-cyan-400 transition flex items-center">
        💬 Interactive Custom Prompt
      </button>
      <button id="tabBtnBench" onclick="switchTab('bench')" class="py-2.5 px-4 font-medium text-sm text-slate-400 border-b-2 border-transparent hover:text-slate-200 transition flex items-center">
        📊 Throughput Benchmark
      </button>
    </div>

    <!-- =================================================================== -->
    <!-- TAB 1: Interactive Custom Prompt Sandbox                             -->
    <!-- =================================================================== -->
    <div id="tabCustomPrompt" class="block">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 my-6">
        <!-- Input Configuration Card -->
        <div class="card p-5 md:col-span-1">
          <h2 class="text-lg font-bold text-slate-200 mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
            Prompt Configuration
          </h2>

          <div class="space-y-4 text-sm">
            <div>
              <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Custom Prompt</label>
              <textarea id="promptText" rows="4" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs">def quicksort(arr: list[int]) -> list[int]:
    \"\"\"Sort list in ascending order.\"\"\"</textarea>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Concurrency / Batch</label>
                <input type="number" id="promptBatchSize" value="5" min="1" max="20" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
                <span class="text-[10px] text-slate-500">5-10 stresses UVM memory</span>
              </div>
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Max Tokens</label>
                <input type="number" id="promptMaxTokens" value="64" min="16" max="256" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
              </div>
            </div>

            <div>
              <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">GPU Budget (GiB)</label>
              <input type="number" step="0.5" id="promptGpuBudget" value="2.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
            </div>

            <div class="pt-2 flex flex-col space-y-2">
              <button onclick="startPromptRun('compare')" id="btnPromptCompare" class="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-lg flex items-center justify-center transition">
                ⚡ Run Head-to-Head Comparison
              </button>
              <div class="grid grid-cols-2 gap-2">
                <button onclick="startPromptRun('demand_paging')" id="btnPromptDemand" class="bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800 text-xs font-semibold py-2 px-3 rounded-lg transition">
                  🐢 Demand Paging
                </button>
                <button onclick="startPromptRun('prefetcher')" id="btnPromptPrefetch" class="bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 text-xs font-semibold py-2 px-3 rounded-lg transition">
                  🚀 Our Prefetcher
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Generated Text & Comparison Output -->
        <div class="card p-5 md:col-span-2 flex flex-col justify-between">
          <h2 class="text-lg font-bold text-slate-200 mb-3 flex items-center justify-between">
            <span>Model Generated Output & Performance</span>
            <span id="promptBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400">Idle</span>
          </h2>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Demand Paging Response Box -->
            <div class="bg-slate-900/80 border border-rose-950/80 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-xs font-bold text-rose-400 uppercase tracking-wider">Demand Paging Output</span>
                  <span class="text-xs px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-900">UVM Faulting</span>
                </div>
                <div class="mt-2 bg-slate-950 border border-slate-800 rounded-lg p-2.5 h-36 overflow-y-auto text-xs font-mono text-slate-300 whitespace-pre-wrap" id="promptDpOutput">(Generated output will appear here...)</div>
              </div>
              <div class="mt-3 pt-3 border-t border-slate-800/80 space-y-1 text-xs">
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">Generation Speed:</span>
                  <span class="font-mono font-bold text-slate-100" id="promptDpTokens">-- tok/s</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">Generation Time:</span>
                  <span class="font-mono text-slate-300" id="promptDpTime">-- s</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">GPU Page Faults:</span>
                  <span class="font-mono font-bold text-rose-400" id="promptDpFaults">--</span>
                </div>
              </div>
            </div>

            <!-- Our Prefetcher Response Box -->
            <div class="bg-slate-900/80 border border-emerald-950/80 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Our Prefetcher Output</span>
                  <span class="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-900">Static-Guided</span>
                </div>
                <div class="mt-2 bg-slate-950 border border-slate-800 rounded-lg p-2.5 h-36 overflow-y-auto text-xs font-mono text-emerald-300 whitespace-pre-wrap" id="promptPfOutput">(Generated output will appear here...)</div>
              </div>
              <div class="mt-3 pt-3 border-t border-slate-800/80 space-y-1 text-xs">
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">Generation Speed:</span>
                  <span class="font-mono font-bold text-emerald-300" id="promptPfTokens">-- tok/s</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">Generation Time:</span>
                  <span class="font-mono text-slate-300" id="promptPfTime">-- s</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-slate-400">GPU Page Faults:</span>
                  <span class="font-mono font-bold text-emerald-400" id="promptPfFaults">--</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Delta Banner -->
          <div id="promptDeltaBanner" class="mt-4 p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-around text-center">
            <div>
              <span class="block text-xs text-slate-400">Speedup Multiplier</span>
              <span class="text-xl font-black text-amber-400" id="promptDeltaSpeedup">--</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div>
              <span class="block text-xs text-slate-400">Faults Eliminated</span>
              <span class="text-xl font-black text-cyan-400" id="promptDeltaFaultReduction">--</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div>
              <span class="block text-xs text-slate-400">Latency Saved</span>
              <span class="text-xl font-black text-emerald-400" id="promptDeltaLatency">--</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- =================================================================== -->
    <!-- TAB 2: Throughput Benchmark Mode                                   -->
    <!-- =================================================================== -->
    <div id="tabBench" class="hidden">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 my-6">
        <div class="card p-5 md:col-span-1">
          <h2 class="text-lg font-bold text-slate-200 mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
            Benchmark Configuration
          </h2>

          <div class="space-y-4 text-sm">
            <div>
              <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Preset Workload</label>
              <select id="presetSelect" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500" onchange="onPresetChange()">
                <option value="standard" selected>Standard Severe (20 req, 512 in / 64 out)</option>
                <option value="fast">Fast Verification (5 req, 128 in / 32 out)</option>
                <option value="custom">Custom Parameters</option>
              </select>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Prompts</label>
                <input type="number" id="numPrompts" value="20" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">GPU Budget (GiB)</label>
                <input type="number" step="0.5" id="gpuBudget" value="2.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
              </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Input Length</label>
                <input type="number" id="inputLen" value="512" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Output Length</label>
                <input type="number" id="outputLen" value="64" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-cyan-500">
              </div>
            </div>

            <div class="pt-2 flex flex-col space-y-2">
              <button onclick="startRun('compare')" id="btnCompare" class="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-lg flex items-center justify-center transition">
                ⚡ Run Head-to-Head Comparison
              </button>
              <div class="grid grid-cols-2 gap-2">
                <button onclick="startRun('demand_paging')" id="btnDemand" class="bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800 text-xs font-semibold py-2 px-3 rounded-lg transition">
                  🐢 Demand Paging
                </button>
                <button onclick="startRun('prefetcher')" id="btnPrefetch" class="bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 text-xs font-semibold py-2 px-3 rounded-lg transition">
                  🚀 Our Prefetcher
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Live Comparison Cards -->
        <div class="card p-5 md:col-span-2 flex flex-col justify-between">
          <h2 class="text-lg font-bold text-slate-200 mb-3 flex items-center justify-between">
            <span>Results Summary</span>
            <span id="runBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400">Idle</span>
          </h2>

          <div class="grid grid-cols-2 gap-4">
            <!-- Demand Paging Card -->
            <div class="bg-slate-900/80 border border-rose-950/80 rounded-xl p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-rose-400 uppercase tracking-wider">Demand Paging (Vanilla)</span>
                <span class="text-xs px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-900">UVM Default</span>
              </div>
              <div class="mt-2">
                <span class="text-3xl font-extrabold text-slate-100" id="dpThroughput">--</span>
                <span class="text-xs text-slate-400">req/s</span>
              </div>
              <div class="text-xs text-slate-400 mt-1">
                <span class="font-semibold text-slate-200" id="dpTokens">--</span> output tok/s
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800/80 flex justify-between items-center text-xs">
                <span class="text-slate-400">GPU Page Faults:</span>
                <span class="font-mono font-bold text-rose-400" id="dpFaults">--</span>
              </div>
              <div class="flex justify-between items-center text-xs mt-1">
                <span class="text-slate-400">Elapsed Time:</span>
                <span class="font-mono text-slate-300" id="dpTime">--</span>
              </div>
            </div>

            <!-- Our Prefetcher Card -->
            <div class="bg-slate-900/80 border border-emerald-950/80 rounded-xl p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Our Synchronous Prefetcher</span>
                <span class="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-900">Static-Guided</span>
              </div>
              <div class="mt-2">
                <span class="text-3xl font-extrabold text-emerald-300" id="pfThroughput">--</span>
                <span class="text-xs text-slate-400">req/s</span>
              </div>
              <div class="text-xs text-slate-400 mt-1">
                <span class="font-semibold text-emerald-200" id="pfTokens">--</span> output tok/s
              </div>
              <div class="mt-4 pt-3 border-t border-slate-800/80 flex justify-between items-center text-xs">
                <span class="text-slate-400">GPU Page Faults:</span>
                <span class="font-mono font-bold text-emerald-400" id="pfFaults">--</span>
              </div>
              <div class="flex justify-between items-center text-xs mt-1">
                <span class="text-slate-400">Elapsed Time:</span>
                <span class="font-mono text-slate-300" id="pfTime">--</span>
              </div>
            </div>
          </div>

          <!-- Delta Banner -->
          <div id="deltaBanner" class="mt-4 p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-around text-center">
            <div>
              <span class="block text-xs text-slate-400">Speedup Multiplier</span>
              <span class="text-xl font-black text-amber-400" id="deltaSpeedup">--</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div>
              <span class="block text-xs text-slate-400">Faults Eliminated</span>
              <span class="text-xl font-black text-cyan-400" id="deltaFaultReduction">--</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div>
              <span class="block text-xs text-slate-400">Latency Saved</span>
              <span class="text-xl font-black text-emerald-400" id="deltaLatency">--</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Execution Terminal -->
    <div class="card p-5">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-bold text-slate-300 flex items-center">
          <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 mr-2 animate-pulse"></span>
          Execution Log Terminal
        </h3>
        <button onclick="document.getElementById('terminal').textContent=''" class="text-xs text-slate-500 hover:text-slate-300">Clear</button>
      </div>
      <div id="terminal" class="log-terminal h-60 p-4 text-xs text-slate-300 rounded-lg overflow-y-auto leading-relaxed border border-slate-900">
Ready for test execution...
      </div>
    </div>
  </div>

  <script>
    function switchTab(tab) {
      const tabCustom = document.getElementById('tabCustomPrompt');
      const tabBench = document.getElementById('tabBench');
      const btnCustom = document.getElementById('tabBtnCustomPrompt');
      const btnBench = document.getElementById('tabBtnBench');

      if (tab === 'customPrompt') {
        tabCustom.classList.remove('hidden');
        tabBench.classList.add('hidden');
        btnCustom.className = "py-2.5 px-4 font-bold text-sm border-b-2 border-cyan-400 text-cyan-400 transition flex items-center";
        btnBench.className = "py-2.5 px-4 font-medium text-sm text-slate-400 border-b-2 border-transparent hover:text-slate-200 transition flex items-center";
      } else {
        tabCustom.classList.add('hidden');
        tabBench.classList.remove('hidden');
        btnBench.className = "py-2.5 px-4 font-bold text-sm border-b-2 border-cyan-400 text-cyan-400 transition flex items-center";
        btnCustom.className = "py-2.5 px-4 font-medium text-sm text-slate-400 border-b-2 border-transparent hover:text-slate-200 transition flex items-center";
      }
    }

    function onPresetChange() {
      const p = document.getElementById('presetSelect').value;
      if (p === 'standard') {
        document.getElementById('numPrompts').value = 20;
        document.getElementById('inputLen').value = 512;
        document.getElementById('outputLen').value = 64;
        document.getElementById('gpuBudget').value = 2.0;
      } else if (p === 'fast') {
        document.getElementById('numPrompts').value = 5;
        document.getElementById('inputLen').value = 128;
        document.getElementById('outputLen').value = 32;
        document.getElementById('gpuBudget').value = 2.0;
      }
    }

    function logToTerminal(msg) {
      const t = document.getElementById('terminal');
      t.textContent += msg + "\\n";
      t.scrollTop = t.scrollHeight;
    }

    // -------------------------------------------------------------------------
    // Custom Prompt Execution
    // -------------------------------------------------------------------------
    async function startPromptRun(action) {
      const btnCompare = document.getElementById('btnPromptCompare');
      const btnDemand = document.getElementById('btnPromptDemand');
      const btnPrefetch = document.getElementById('btnPromptPrefetch');
      const badge = document.getElementById('promptBadge');

      btnCompare.disabled = true;
      btnDemand.disabled = true;
      btnPrefetch.disabled = true;

      const promptText = document.getElementById('promptText').value;
      const batchSize = parseInt(document.getElementById('promptBatchSize').value);
      const maxTokens = parseInt(document.getElementById('promptMaxTokens').value);
      const gpuBudget = parseFloat(document.getElementById('promptGpuBudget').value);

      logToTerminal(`>>> Launching Custom Prompt Run [${action}] (Batch: ${batchSize}, MaxTokens: ${maxTokens}, Budget: ${gpuBudget} GiB)`);
      badge.textContent = "Generating...";
      badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 animate-pulse";

      try {
        const response = await fetch('/api/run_prompt', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            action: action,
            prompt: promptText,
            batch_size: batchSize,
            max_tokens: maxTokens,
            gpu_budget_gb: gpuBudget,
          }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'log') {
                logToTerminal(data.message);
              } else if (data.type === 'prompt_result_dp') {
                document.getElementById('promptDpOutput').textContent = data.sample_output || "(No output generated)";
                document.getElementById('promptDpTokens').textContent = data.tokens_per_sec.toFixed(2) + ' tok/s';
                document.getElementById('promptDpFaults').textContent = data.gpu_page_faults.toLocaleString();
                document.getElementById('promptDpTime').textContent = data.elapsed_sec.toFixed(2) + ' s';
              } else if (data.type === 'prompt_result_pf') {
                document.getElementById('promptPfOutput').textContent = data.sample_output || "(No output generated)";
                document.getElementById('promptPfTokens').textContent = data.tokens_per_sec.toFixed(2) + ' tok/s';
                document.getElementById('promptPfFaults').textContent = data.gpu_page_faults.toLocaleString();
                document.getElementById('promptPfTime').textContent = data.elapsed_sec.toFixed(2) + ' s';
              } else if (data.type === 'prompt_delta') {
                document.getElementById('promptDeltaSpeedup').textContent = data.speedup + 'x';
                document.getElementById('promptDeltaFaultReduction').textContent = '-' + data.fault_reduction_pct + '%';
                document.getElementById('promptDeltaLatency').textContent = '-' + data.latency_saved_sec + 's';
              }
            }
          }
        }
      } catch (err) {
        logToTerminal('[!] Error during prompt run: ' + err);
      } finally {
        btnCompare.disabled = false;
        btnDemand.disabled = false;
        btnPrefetch.disabled = false;
        badge.textContent = "Done";
        badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300";
        logToTerminal(">>> Custom prompt run finished.");
      }
    }

    // -------------------------------------------------------------------------
    // Throughput Benchmark Execution
    // -------------------------------------------------------------------------
    async function startRun(action) {
      const btnCompare = document.getElementById('btnCompare');
      const btnDemand = document.getElementById('btnDemand');
      const btnPrefetch = document.getElementById('btnPrefetch');
      const badge = document.getElementById('runBadge');

      btnCompare.disabled = true;
      btnDemand.disabled = true;
      btnPrefetch.disabled = true;

      const numPrompts = parseInt(document.getElementById('numPrompts').value);
      const inputLen = parseInt(document.getElementById('inputLen').value);
      const outputLen = parseInt(document.getElementById('outputLen').value);
      const gpuBudget = parseFloat(document.getElementById('gpuBudget').value);

      const params = {
        action: action,
        num_prompts: numPrompts,
        input_len: inputLen,
        output_len: outputLen,
        gpu_budget_gb: gpuBudget,
      };

      logToTerminal(`>>> Launching Benchmark [${action}] (Prompts: ${numPrompts}, In: ${inputLen}, Out: ${outputLen}, Budget: ${gpuBudget} GiB)`);
      badge.textContent = "Benchmarking...";
      badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 animate-pulse";

      try {
        const response = await fetch('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(params),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'log') {
                logToTerminal(data.message);
              } else if (data.type === 'result_dp') {
                document.getElementById('dpThroughput').textContent = data.throughput_req_s.toFixed(2);
                document.getElementById('dpTokens').textContent = data.output_tok_s.toFixed(2);
                document.getElementById('dpFaults').textContent = data.gpu_page_faults.toLocaleString();
                document.getElementById('dpTime').textContent = data.elapsed_sec.toFixed(2) + ' s';
              } else if (data.type === 'result_pf') {
                document.getElementById('pfThroughput').textContent = data.throughput_req_s.toFixed(2);
                document.getElementById('pfTokens').textContent = data.output_tok_s.toFixed(2);
                document.getElementById('pfFaults').textContent = data.gpu_page_faults.toLocaleString();
                document.getElementById('pfTime').textContent = data.elapsed_sec.toFixed(2) + ' s';
              } else if (data.type === 'delta') {
                document.getElementById('deltaSpeedup').textContent = data.speedup + 'x';
                document.getElementById('deltaFaultReduction').textContent = '-' + data.fault_reduction_pct + '%';
                document.getElementById('deltaLatency').textContent = '-' + data.latency_saved_sec + 's';
              }
            }
          }
        }
      } catch (err) {
        logToTerminal('[!] Error during benchmark: ' + err);
      } finally {
        btnCompare.disabled = false;
        btnDemand.disabled = false;
        btnPrefetch.disabled = false;
        badge.textContent = "Done";
        badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300";
        logToTerminal(">>> Benchmark execution complete.");
      }
    }
  </script>
</body>
</html>
"""


def create_web_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="vLLM UVM Comparator")

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    async def get_index():
        return HTML_PAGE

    @app.post("/api/run_prompt")
    async def api_run_prompt(request: Request):
        body = await request.json()
        action = body.get("action", "compare")
        prompt = body.get("prompt", "def quicksort(arr):")
        max_tokens = int(body.get("max_tokens", 64))
        batch_size = int(body.get("batch_size", 5))
        gpu_budget_gb = float(body.get("gpu_budget_gb", 2.0))

        async def event_generator() -> AsyncGenerator[str, None]:
            dp_res = None
            pf_res = None

            if action in ["compare", "demand_paging"]:
                yield f"data: {json.dumps({'type': 'log', 'message': '[1/2] Generating with Vanilla Demand Paging (TRITON Attention)...'})}\n\n"
                loop = asyncio.get_event_loop()
                dp_res = await loop.run_in_executor(
                    None,
                    lambda: run_custom_prompt_subprocess(
                        mode="demand_paging",
                        prompt=prompt,
                        max_tokens=max_tokens,
                        batch_size=batch_size,
                        gpu_budget_gb=gpu_budget_gb,
                    )
                )
                dp_toks_str = f"{dp_res['tokens_per_sec']:.2f}"
                dp_faults_str = f"{dp_res['gpu_page_faults']:,}"
                dp_msg = f"[Demand Paging Done] Generation Speed: {dp_toks_str} tok/s, GPU Faults: {dp_faults_str}, Latency: {dp_res['elapsed_sec']}s"
                yield f"data: {json.dumps({'type': 'log', 'message': dp_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'prompt_result_dp', **dp_res})}\n\n"

            if action in ["compare", "prefetcher"]:
                step_prefix = "[2/2]" if action == "compare" else "[1/1]"
                yield f"data: {json.dumps({'type': 'log', 'message': f'{step_prefix} Generating with Our Synchronous Prefetcher (TRITON Attention)...'})}\n\n"
                loop = asyncio.get_event_loop()
                pf_res = await loop.run_in_executor(
                    None,
                    lambda: run_custom_prompt_subprocess(
                        mode="prefetcher",
                        prompt=prompt,
                        max_tokens=max_tokens,
                        batch_size=batch_size,
                        gpu_budget_gb=gpu_budget_gb,
                    )
                )
                pf_toks_str = f"{pf_res['tokens_per_sec']:.2f}"
                pf_faults_str = f"{pf_res['gpu_page_faults']:,}"
                pf_msg = f"[Prefetcher Done] Generation Speed: {pf_toks_str} tok/s, GPU Faults: {pf_faults_str}, Latency: {pf_res['elapsed_sec']}s"
                yield f"data: {json.dumps({'type': 'log', 'message': pf_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'prompt_result_pf', **pf_res})}\n\n"

            if dp_res and pf_res:
                dp_toks = max(0.001, dp_res.get("tokens_per_sec", 0.001))
                pf_toks = pf_res.get("tokens_per_sec", 0.0)
                speedup = round(pf_toks / dp_toks, 2)

                dp_faults = max(1, dp_res.get("gpu_page_faults", 1))
                pf_faults = pf_res.get("gpu_page_faults", 0)
                fault_red_pct = round(((dp_faults - pf_faults) / dp_faults) * 100, 1)

                latency_saved = round(max(0, dp_res.get("elapsed_sec", 0) - pf_res.get("elapsed_sec", 0)), 2)

                delta_payload = {
                    "type": "prompt_delta",
                    "speedup": speedup,
                    "fault_reduction_pct": fault_red_pct,
                    "latency_saved_sec": latency_saved,
                }
                yield f"data: {json.dumps(delta_payload)}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': f'>>> PROMPT COMPARISON RESULT: {speedup}x Generation Speedup, {fault_red_pct}% GPU Faults Eliminated, {latency_saved}s Latency Saved!'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/run")
    async def api_run(request: Request):
        body = await request.json()
        action = body.get("action", "compare")
        num_prompts = int(body.get("num_prompts", 20))
        input_len = int(body.get("input_len", 512))
        output_len = int(body.get("output_len", 64))
        gpu_budget_gb = float(body.get("gpu_budget_gb", 2.0))

        async def event_generator() -> AsyncGenerator[str, None]:
            dp_res = None
            pf_res = None

            if action in ["compare", "demand_paging"]:
                yield f"data: {json.dumps({'type': 'log', 'message': '[1/2] Starting Vanilla Demand Paging Benchmark (TRITON Attention)...'})}\n\n"
                loop = asyncio.get_event_loop()
                dp_res = await loop.run_in_executor(
                    None,
                    lambda: run_benchmark_subprocess(
                        mode="demand_paging",
                        num_prompts=num_prompts,
                        input_len=input_len,
                        output_len=output_len,
                        gpu_budget_gb=gpu_budget_gb,
                    )
                )
                dp_tp_str = f"{dp_res['throughput_req_s']:.2f}"
                dp_faults_str = f"{dp_res['gpu_page_faults']:,}"
                dp_msg = f"[Demand Paging Done] Throughput: {dp_tp_str} req/s, GPU Faults: {dp_faults_str}, Elapsed: {dp_res['elapsed_sec']}s"
                yield f"data: {json.dumps({'type': 'log', 'message': dp_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'result_dp', **dp_res})}\n\n"

            if action in ["compare", "prefetcher"]:
                step_prefix = "[2/2]" if action == "compare" else "[1/1]"
                yield f"data: {json.dumps({'type': 'log', 'message': f'{step_prefix} Starting Our Synchronous Prefetcher Benchmark (TRITON Attention)...'})}\n\n"
                loop = asyncio.get_event_loop()
                pf_res = await loop.run_in_executor(
                    None,
                    lambda: run_benchmark_subprocess(
                        mode="prefetcher",
                        num_prompts=num_prompts,
                        input_len=input_len,
                        output_len=output_len,
                        gpu_budget_gb=gpu_budget_gb,
                    )
                )
                pf_tp_str = f"{pf_res['throughput_req_s']:.2f}"
                pf_faults_str = f"{pf_res['gpu_page_faults']:,}"
                pf_msg = f"[Prefetcher Done] Throughput: {pf_tp_str} req/s, GPU Faults: {pf_faults_str}, Elapsed: {pf_res['elapsed_sec']}s"
                yield f"data: {json.dumps({'type': 'log', 'message': pf_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'result_pf', **pf_res})}\n\n"

            if dp_res and pf_res:
                dp_tp = max(0.001, dp_res.get("throughput_req_s", 0.001))
                pf_tp = pf_res.get("throughput_req_s", 0.0)
                speedup = round(pf_tp / dp_tp, 2)

                dp_faults = max(1, dp_res.get("gpu_page_faults", 1))
                pf_faults = pf_res.get("gpu_page_faults", 0)
                fault_red_pct = round(((dp_faults - pf_faults) / dp_faults) * 100, 1)

                latency_saved = round(max(0, dp_res.get("elapsed_sec", 0) - pf_res.get("elapsed_sec", 0)), 2)

                delta_payload = {
                    "type": "delta",
                    "speedup": speedup,
                    "fault_reduction_pct": fault_red_pct,
                    "latency_saved_sec": latency_saved,
                }
                yield f"data: {json.dumps(delta_payload)}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': f'>>> HEAD-TO-HEAD RESULT: {speedup}x Throughput Speedup, {fault_red_pct}% GPU Faults Eliminated, {latency_saved}s Latency Saved!'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="vLLM UVM Oversubscription Comparator UI")
    parser.add_argument("--web", action="store_true", help="Launch FastAPI Web UI dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web UI host (default: 0.0.0.0)")
    parser.add_argument("--cli", action="store_true", help="Launch interactive Terminal CLI UI")
    args = parser.parse_args()

    if args.web or not args.cli:
        app = create_web_app()
        import uvicorn
        print(f"\n==========================================================================")
        print(f"🚀 vLLM UVM Comparator Web UI listening on: http://localhost:{args.port}")
        print(f"   (Access locally or forward port via: ssh -L {args.port}:localhost:{args.port} <host>)")
        print(f"==========================================================================\n")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        run_cli_ui()


if __name__ == "__main__":
    main()
