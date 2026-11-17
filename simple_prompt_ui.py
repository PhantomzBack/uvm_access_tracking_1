#!/usr/bin/env python3
"""
Simple UI to prompt the running vLLM UVM server.
Can run as:
  1. Web UI: python simple_prompt_ui.py --web --port 8080 (default)
  2. Terminal CLI: python simple_prompt_ui.py --cli
  3. One-shot CLI: python simple_prompt_ui.py --prompt "def foo():"
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

FAULT_STATS_PATH = "/proc/driver/nvidia-uvm/cpu/fault_stats"
STATUS_FILE = "/home/arush/data/ravi/uvm_access_tracking_1/.server_status.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def read_gpu_faults() -> int:
    """Reads current GPU page faults count from UVM driver."""
    if not os.path.exists(FAULT_STATS_PATH):
        return 0
    try:
        with open(FAULT_STATS_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and "gpu_faults" in parts[0]:
                    return int(parts[1].strip())
    except Exception:
        pass
    return 0


def get_server_status(server_url: str = "http://localhost:8000") -> dict:
    online = False
    try:
        req = urllib.request.Request(f"{server_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            online = (resp.status == 200)
    except Exception:
        online = False

    meta = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                meta = json.load(f)
        except Exception:
            pass

    return {
        "online": online,
        "mode": meta.get("mode", "unknown"),
        "mode_title": meta.get("mode_title", "Unknown Mode"),
        "budget_gb": meta.get("budget_gb", 2.0),
        "model": meta.get("model", DEFAULT_MODEL),
        "backend": meta.get("backend", "TRITON_ATTN"),
        "oversubscribe_weights": meta.get("oversubscribe_weights", False),
    }


def send_prompt(
    prompt: str,
    server_url: str = "http://localhost:8000",
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> dict:
    """Sends a single prompt to vLLM server, measuring latency and GPU page faults."""
    faults_before = read_gpu_faults()
    t0 = time.perf_counter()

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    req = urllib.request.Request(
        f"{server_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output_text": "",
            "tokens": 0,
            "elapsed_sec": 0.0,
            "tokens_per_sec": 0.0,
            "gpu_faults": 0,
        }

    elapsed = time.perf_counter() - t0
    faults_after = read_gpu_faults()
    delta_faults = max(0, faults_after - faults_before)

    choice = data.get("choices", [{}])[0]
    output_text = choice.get("text", "")
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(output_text.split()))

    tok_per_sec = completion_tokens / max(0.001, elapsed)

    return {
        "success": True,
        "prompt": prompt,
        "output_text": output_text,
        "completion_tokens": completion_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tok_per_sec, 2),
        "gpu_faults": delta_faults,
    }


# ==============================================================================
# Web UI (FastAPI)
# ==============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vLLM UVM Prompt UI</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
    .card { background-color: #131c2e; border: 1px solid #1e293b; border-radius: 0.75rem; }
    .code-box { background-color: #060913; border: 1px solid #1e293b; font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body class="p-6 min-h-screen flex flex-col justify-between">
  <div class="max-w-4xl mx-auto w-full">
    <!-- Header -->
    <div class="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
      <div>
        <h1 class="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
          vLLM UVM Simple Prompt UI
        </h1>
        <p class="text-xs text-slate-400 mt-1">Single request prompt tester for oversubscribed GPU memory with TRITON attention</p>
      </div>

      <!-- Server Status Badge -->
      <div id="serverStatusBadge" class="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800 text-xs border border-slate-700">
        <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
        <span id="serverStatusText">Checking Server...</span>
      </div>
    </div>

    <!-- Active Mode Banner -->
    <div id="modeBanner" class="p-4 rounded-xl bg-slate-900 border border-slate-800 mb-6 flex items-center justify-between">
      <div>
        <span class="text-xs text-slate-400 uppercase tracking-wider block font-semibold">Active Server Mode</span>
        <span class="text-base font-bold text-slate-200" id="modeTitle">Connecting...</span>
      </div>
      <div class="text-right text-xs text-slate-400">
        <div>Model: <span class="text-slate-200 font-mono">Qwen2.5-Coder-7B</span></div>
        <div>Budget: <span class="text-cyan-400 font-mono">2.0 GiB GPU VRAM</span></div>
      </div>
    </div>

    <!-- Prompt Box -->
    <div class="card p-5 mb-6">
      <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Input Prompt</label>
      <textarea id="promptInput" rows="3" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-sm leading-relaxed">def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.\"\"\"</textarea>

      <div class="grid grid-cols-2 gap-4 mt-3">
        <div>
          <label class="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">Max Generated Tokens</label>
          <input type="number" id="maxTokens" value="64" min="8" max="512" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 text-xs focus:outline-none focus:border-cyan-500">
        </div>
        <div>
          <label class="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">Temperature</label>
          <input type="number" id="temperature" value="0.0" step="0.1" min="0" max="1" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 text-xs focus:outline-none focus:border-cyan-500">
        </div>
      </div>

      <button onclick="submitPrompt()" id="btnSubmit" class="mt-4 w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-lg flex items-center justify-center transition">
        ⚡ Send Prompt
      </button>
    </div>

    <!-- Output Box -->
    <div class="card p-5">
      <div class="flex items-center justify-between mb-2">
        <label class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Generated Output</label>
        <span id="outputBadge" class="text-[11px] text-slate-500">Awaiting input</span>
      </div>

      <pre id="outputText" class="code-box p-4 rounded-lg text-xs text-slate-200 overflow-x-auto min-h-[120px] whitespace-pre-wrap leading-relaxed">(Response text will appear here immediately after generation...)</pre>

      <!-- Metrics Bar -->
      <div class="grid grid-cols-3 gap-3 mt-4 text-center">
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5">
          <span class="block text-[10px] text-slate-400 uppercase font-semibold">Generation Speed</span>
          <span id="metricSpeed" class="text-lg font-black text-cyan-400">--</span>
          <span class="text-[10px] text-slate-500 block">tok/s</span>
        </div>
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5">
          <span class="block text-[10px] text-slate-400 uppercase font-semibold">Latency</span>
          <span id="metricTime" class="text-lg font-black text-slate-200">--</span>
          <span class="text-[10px] text-slate-500 block">seconds</span>
        </div>
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5">
          <span class="block text-[10px] text-slate-400 uppercase font-semibold">GPU Page Faults</span>
          <span id="metricFaults" class="text-lg font-black text-amber-400">--</span>
          <span class="text-[10px] text-slate-500 block">PCIe faults</span>
        </div>
      </div>
    </div>
  </div>

  <footer class="mt-8 text-center text-xs text-slate-600">
    vLLM + NVIDIA UVM Access Tracking & Prefetching Research
  </footer>

  <script>
    async function checkStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('serverStatusBadge');
        const text = document.getElementById('serverStatusText');
        const title = document.getElementById('modeTitle');

        if (data.online) {
          badge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950 text-emerald-300 text-xs border border-emerald-800";
          badge.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span><span>Server Online</span>';
          title.textContent = data.mode_title || "Server Online";
          if (data.mode === 'prefetcher') {
            title.className = "text-base font-bold text-emerald-400";
          } else {
            title.className = "text-base font-bold text-rose-400";
          }
        } else {
          badge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full bg-rose-950 text-rose-300 text-xs border border-rose-800";
          badge.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span><span>Server Offline</span>';
          title.textContent = "Server Offline (Run start_server.sh first)";
          title.className = "text-base font-bold text-slate-500";
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function submitPrompt() {
      const btn = document.getElementById('btnSubmit');
      const badge = document.getElementById('outputBadge');
      const prompt = document.getElementById('promptInput').value;
      const maxTokens = parseInt(document.getElementById('maxTokens').value);
      const temp = parseFloat(document.getElementById('temperature').value);

      btn.disabled = true;
      btn.textContent = "⏳ Generating...";
      badge.textContent = "Running inference...";

      try {
        const res = await fetch('/api/prompt', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompt: prompt,
            max_tokens: maxTokens,
            temperature: temp,
          })
        });

        const data = await res.json();
        if (data.success) {
          document.getElementById('outputText').textContent = data.output_text || "(Empty response)";
          document.getElementById('metricSpeed').textContent = data.tokens_per_sec.toFixed(1);
          document.getElementById('metricTime').textContent = data.elapsed_sec.toFixed(2);
          document.getElementById('metricFaults').textContent = data.gpu_faults.toLocaleString();
          badge.textContent = `Completed (${data.completion_tokens} tokens)`;
        } else {
          document.getElementById('outputText').textContent = "[Error] " + data.error;
          badge.textContent = "Failed";
        }
      } catch (err) {
        document.getElementById('outputText').textContent = "[Error] " + err;
        badge.textContent = "Failed";
      } finally {
        btn.disabled = false;
        btn.textContent = "⚡ Send Prompt";
      }
    }

    // Poll status periodically
    checkStatus();
    setInterval(checkStatus, 3000);
  </script>
</body>
</html>
"""


def run_web(port: int = 8080):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    app = FastAPI(title="vLLM UVM Prompt UI")

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    async def get_index():
        return HTML_PAGE

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(get_server_status())

    @app.post("/api/prompt")
    async def api_prompt(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "def hello():")
        max_tokens = int(body.get("max_tokens", 64))
        temp = float(body.get("temperature", 0.0))

        res = send_prompt(prompt, max_tokens=max_tokens, temperature=temp)
        return JSONResponse(res)

    print("\n" + "=" * 65)
    print(f"🚀 Simple Prompt Web UI listening on: http://localhost:{port}")
    print(f"   (Access locally or forward port via: ssh -L {port}:localhost:{port} <host>)")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


# ==============================================================================
# Terminal CLI
# ==============================================================================

def run_cli_interactive():
    print("\n=======================================================")
    print("  vLLM UVM Simple Terminal Prompt REPL")
    print("=======================================================")

    status = get_server_status()
    if not status["online"]:
        print("\n[!] vLLM server is currently OFFLINE at http://localhost:8000")
        print("    Please start the server first using:")
        print("    ./start_server.sh")
        print("")
        sys.exit(1)

    print(f"Connected to vLLM Server: {status['mode_title']}")
    print(f"Model: {status['model']} | Budget: {status['budget_gb']} GiB GPU VRAM\n")
    print("Type your prompt and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("Prompt >>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            print("⏳ Generating...")
            res = send_prompt(prompt, max_tokens=64, temperature=0.0)

            if not res["success"]:
                print(f"[!] Error: {res['error']}\n")
                continue

            print("\n" + "-" * 50)
            print(res["output_text"])
            print("-" * 50)
            print(f"⚡ {res['completion_tokens']} tokens in {res['elapsed_sec']}s ({res['tokens_per_sec']} tok/s) | GPU Page Faults: {res['gpu_faults']:,}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Simple Prompt UI for vLLM UVM server")
    parser.add_argument("--web", action="store_true", help="Launch Web UI dashboard (default)")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal REPL")
    parser.add_argument("--prompt", type=str, help="Execute a single one-shot prompt from command line")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens (default: 64)")
    args = parser.parse_args()

    if args.prompt:
        res = send_prompt(args.prompt, max_tokens=args.max_tokens)
        if res["success"]:
            print(res["output_text"])
            print(f"\n[{res['completion_tokens']} tokens in {res['elapsed_sec']}s ({res['tokens_per_sec']} tok/s) | GPU Page Faults: {res['gpu_faults']:,}]")
        else:
            print(f"[!] Error: {res['error']}", file=sys.stderr)
            sys.exit(1)
    elif args.cli:
        run_cli_interactive()
    else:
        run_web(port=args.port)


if __name__ == "__main__":
    main()
