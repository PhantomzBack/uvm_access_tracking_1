#!/usr/bin/env python3
"""
Simple script to launch vLLM server in either of the two UVM modes:
  1. Vanilla UVM Demand Paging (hardware MMU faulting)
  2. UVM with Our Synchronous Prefetcher (static-analysis guided, low/zero faults)

Always uses TRITON attention (--attention-backend TRITON_ATTN) and 2.0 GiB GPU budget.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

VENV_VLLM = "/home/arush/data/vllm/nuvllm/.venv/bin/vllm"
LIB_INTERCEPT = "/home/arush/data/ravi/uvm_access_tracking_1/libMallocIntercept.so"
HF_CACHE = "/home/arush/data/huggingface_cache"
STATUS_FILE = "/home/arush/data/ravi/uvm_access_tracking_1/.server_status.json"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


def check_server_health(port: int = 8000) -> bool:
    try:
        url = f"http://localhost:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server with UVM oversubscription and Triton attention")
    parser.add_argument("--mode", choices=["demand_paging", "prefetcher"], help="UVM execution mode")
    parser.add_argument("--oversubscribe-weights", action="store_true", help="Oversubscribe model weights in UVM (Option C)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--budget", type=float, default=2.0, help="GPU memory budget in GiB (default: 2.0)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model context len (default: 4096)")
    args = parser.parse_args()

    mode = args.mode
    oversubscribe_weights = args.oversubscribe_weights

    if not mode:
        print("\n=======================================================")
        print("  Select vLLM UVM Execution Mode:")
        print("  [1] 🐢 Vanilla Demand Paging (Hardware MMU Faults)")
        print("  [2] 🚀 Our Synchronous Prefetcher (Static-Guided)")
        print("=======================================================")
        choice = input("Enter choice [1 or 2] (default: 1): ").strip()
        mode = "prefetcher" if choice == "2" else "demand_paging"

        weights_choice = input("Oversubscribe model weights as well (Option C)? [y/N] (default: N): ").strip().lower()
        oversubscribe_weights = (weights_choice == "y" or weights_choice == "yes")

    # Setup environment
    env = os.environ.copy()
    env["HF_HOME"] = HF_CACHE
    env["VLLM_USE_UVM"] = "3"
    env["VLLM_UVM_GPU_BUDGET_GB"] = str(args.budget)
    if oversubscribe_weights:
        env["VLLM_USE_WEIGHTS_UVM"] = "1"

    if mode == "prefetcher":
        env["LD_PRELOAD"] = LIB_INTERCEPT
        env["UVM_PREFETCH_SYNC"] = "1"
        mode_title = "🚀 UVM with Our Synchronous Prefetcher"
        mode_desc = "libMallocIntercept active, prefetching pages ahead of Triton kernels."
    else:
        env.pop("LD_PRELOAD", None)
        env.pop("UVM_PREFETCH_SYNC", None)
        env["UVM_PREFETCH_ENABLE"] = "0"
        mode_title = "🐢 Vanilla UVM Demand Paging"
        mode_desc = "Default Linux/NVIDIA UVM driver with demand paging page faults."

    cmd = [
        VENV_VLLM,
        "serve",
        MODEL_NAME,
        "--attention-backend", "TRITON_ATTN",
        "--max-model-len", str(args.max_model_len),
        "--enforce-eager",
        "--port", str(args.port),
    ]

    print("\n" + "=" * 70)
    print(f"Starting vLLM Server")
    print(f"  Mode:            {mode_title}")
    print(f"  Description:     {mode_desc}")
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Attention:       TRITON_ATTN")
    weights_desc = "Weights + KV-Cache offloaded to Host DRAM (Option C)" if oversubscribe_weights else "KV-Cache offloaded to Host DRAM"
    print(f"  GPU Budget:      {args.budget} GiB ({weights_desc})")
    print(f"  Serving Port:    {args.port}")
    print("=" * 70 + "\n")

    # Record status for UI
    status_data = {
        "status": "starting",
        "mode": mode,
        "mode_title": mode_title,
        "port": args.port,
        "budget_gb": args.budget,
        "model": MODEL_NAME,
        "backend": "TRITON_ATTN",
        "oversubscribe_weights": oversubscribe_weights,
        "start_time": time.time(),
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f)

    # Launch subprocess from /tmp so cwd doesn't conflict with editable install
    proc = subprocess.Popen(cmd, cwd="/tmp", env=env)

    def handle_sigint(sig, frame):
        print("\n[!] Stopping vLLM server...")
        try:
            status_data["status"] = "offline"
            with open(STATUS_FILE, "w") as f:
                json.dump(status_data, f)
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Monitor until ready
    print("⏳ Loading weights and initializing KV-cache (takes ~25 seconds)...")
    ready = False
    for _ in range(60):
        if proc.poll() is not None:
            print(f"[!] Server process exited unexpectedly with code {proc.returncode}")
            status_data["status"] = "failed"
            with open(STATUS_FILE, "w") as f:
                json.dump(status_data, f)
            sys.exit(proc.returncode)

        if check_server_health(args.port):
            ready = True
            break
        time.sleep(1)

    if ready:
        status_data["status"] = "online"
        with open(STATUS_FILE, "w") as f:
            json.dump(status_data, f)

        print("\n" + "#" * 70)
        print(f"✅ vLLM SERVER IS LIVE AND READY!")
        print(f"   URL:   http://localhost:{args.port}")
        print(f"   Mode:  {mode_title}")
        print(f"   Prompt with:  python simple_prompt_ui.py")
        print(f"                 or open Web UI on http://localhost:8080")
        print("#" * 70 + "\n")

    # Keep running and streaming logs
    proc.wait()


if __name__ == "__main__":
    main()
