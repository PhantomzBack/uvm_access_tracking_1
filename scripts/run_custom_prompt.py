#!/usr/bin/env python3
"""
Custom prompt worker for vLLM under UVM oversubscription with TRITON attention.
Outputs JSON metrics and generated text.
"""
import argparse
import json
import os
import sys
import time

FAULT_STATS_PATH = "/proc/driver/nvidia-uvm/cpu/fault_stats"

def read_faults() -> int:
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

def reset_faults() -> None:
    if os.path.exists(FAULT_STATS_PATH):
        try:
            with open(FAULT_STATS_PATH, "w") as f:
                f.write("0\n")
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max generated tokens")
    parser.add_argument("--batch-size", type=int, default=1, help="Concurrent batch count")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--gpu-budget-gb", type=float, default=2.0, help="UVM GPU budget in GiB")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    os.environ["VLLM_USE_UVM"] = "3"
    os.environ["VLLM_UVM_GPU_BUDGET_GB"] = str(args.gpu_budget_gb)
    os.environ["HF_HOME"] = "/home/arush/data/huggingface_cache"

    from vllm import LLM, SamplingParams

    # Initialize engine
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        attention_backend="TRITON_ATTN",
        enforce_eager=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    prompts = [args.prompt] * args.batch_size

    # Reset fault counter right before inference
    reset_faults()
    faults_start = read_faults()

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.perf_counter()

    faults_end = read_faults()
    delta_faults = max(0, faults_end - faults_start)

    elapsed = t_end - t_start
    total_output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    tok_per_sec = total_output_tokens / max(0.001, elapsed)

    sample_output = outputs[0].outputs[0].text

    result = {
        "prompt": args.prompt,
        "sample_output": sample_output,
        "batch_size": args.batch_size,
        "total_output_tokens": total_output_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tok_per_sec, 2),
        "gpu_page_faults": delta_faults,
    }

    # Print clean marker so caller can extract JSON easily
    print("\n---JSON_START---")
    print(json.dumps(result))
    print("---JSON_END---")

    # Clean exit to prevent process group hang
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    os._exit(0)

if __name__ == "__main__":
    main()
