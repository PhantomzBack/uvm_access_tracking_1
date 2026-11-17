#!/usr/bin/env python3
"""
Context Inflation Benchmark for vLLM under UVM.
Runs sequential, back-to-back prompts that inflate the context window across turns.
Measures inference latency, tokens/sec, and GPU page faults with temperature=0.0.
"""

import argparse
import json
import os
import sys
import time

FAULT_STATS_PATH = "/proc/driver/nvidia-uvm/cpu/fault_stats"


def read_gpu_faults() -> int:
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


def main():
    parser = argparse.ArgumentParser(description="Multi-turn context inflation benchmark")
    parser.add_argument("--mode", choices=["demand_paging", "prefetcher"], default="prefetcher")
    parser.add_argument("--weights-uvm", type=int, default=1, help="1 to oversubscribe weights in UVM, 0 for KV-only")
    parser.add_argument("--gpu-budget-gb", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per turn")
    parser.add_argument("--turns", type=int, default=3, help="Number of inflating turns")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    os.environ["VLLM_USE_UVM"] = "3"
    os.environ["VLLM_UVM_GPU_BUDGET_GB"] = str(args.gpu_budget_gb)
    os.environ["VLLM_USE_WEIGHTS_UVM"] = str(args.weights_uvm)
    os.environ["HF_HOME"] = "/home/arush/data/huggingface_cache"

    from vllm import LLM, SamplingParams

    print("=" * 80)
    print("CONTEXT INFLATION BENCHMARK (MULTI-TURN INFERENCE)")
    print(f"  Mode:              {args.mode.upper()}")
    print(f"  Model:             {args.model}")
    print(f"  GPU Budget:        {args.gpu_budget_gb} GiB")
    print(f"  Weights in UVM:    {'YES (Option C)' if args.weights_uvm else 'NO (KV-cache only)'}")
    print(f"  Max Tokens / Turn: {args.max_tokens}")
    print(f"  Number of Turns:   {args.turns}")
    print(f"  Sampling:          Greedy (temperature=0.0)")
    print("=" * 80 + "\n")

    t_init_0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        attention_backend="TRITON_ATTN",
        enforce_eager=True,
        disable_log_stats=True,
    )
    t_init_1 = time.perf_counter()
    init_duration = t_init_1 - t_init_0
    print(f"[*] Engine initialization completed in {init_duration:.2f} seconds\n")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    # Progressive, technical prompts that build on top of each other
    prompts_sequence = [
        # Turn 1: Initial detailed specification (~350-400 tokens)
        (
            "Write a production-grade, thread-safe LRU Cache in Python from scratch without using any external libraries.\n"
            "Requirements:\n"
            "1. Implement a DoublyLinkedList with Node objects (key, value, prev, next).\n"
            "2. Implement LRUCache with maximum capacity, get(key), and put(key, value) in O(1) time complexity.\n"
            "3. Support optional per-key TTL (time-to-live in seconds) expiration.\n"
            "4. Include threading.RLock for thread safety across all operations.\n"
            "5. Provide an eviction callback hook when entries are evicted.\n"
            "Include complete type hints, docstrings, and robust edge-case handling.\n"
        ),
        # Turn 2: Follow-up that requests sharding and consistent hashing
        (
            "\n\nNow, extend the previous LRU cache implementation to create a DistributedShardedCache class.\n"
            "Requirements:\n"
            "1. Partition entries across N internal LRUCache shards based on consistent hashing.\n"
            "2. Implement a virtual node ring using a 32-bit hash function (like fnv1a or crc32) to balance keys.\n"
            "3. Support dynamic shard rebalancing and atomic batch_get(keys) / batch_put(items).\n"
            "4. Implement an asynchronous background worker that periodically purges expired TTL items.\n"
            "Provide the complete Python class with type annotations and clean error handling.\n"
        ),
        # Turn 3: Follow-up that requests monitoring and benchmarking
        (
            "\n\nNow, write a high-concurrency stress test and observability harness for the DistributedShardedCache.\n"
            "Requirements:\n"
            "1. Use concurrent.futures.ThreadPoolExecutor with configurable worker threads and operation counts.\n"
            "2. Generate key accesses following a Zipfian distribution to simulate hot-key contention.\n"
            "3. Collect detailed performance metrics: total ops, throughput (ops/sec), p50, p95, p99 latencies in ms.\n"
            "4. Verify thread-safety, ensuring no race conditions or corrupted cache state under high load.\n"
            "Write runnable executable Python code with an example if __name__ == '__main__' block.\n"
        ),
    ]

    accumulated_context = ""
    results = []

    for turn_idx in range(min(args.turns, len(prompts_sequence))):
        turn_num = turn_idx + 1
        new_prompt_segment = prompts_sequence[turn_idx]
        accumulated_context += new_prompt_segment

        print(f"\n{'─' * 70}")
        print(f"▶ STARTING TURN {turn_num} (Inflating Context Window)")
        print(f"{'─' * 70}")

        faults_before = read_gpu_faults()
        t_start = time.perf_counter()

        outputs = llm.generate([accumulated_context], sampling_params)

        t_end = time.perf_counter()
        faults_after = read_gpu_faults()

        elapsed = t_end - t_start
        delta_faults = max(0, faults_after - faults_before)

        out_obj = outputs[0]
        prompt_token_count = len(out_obj.prompt_token_ids)
        output_token_count = len(out_obj.outputs[0].token_ids)
        total_tokens = prompt_token_count + output_token_count
        tok_per_sec = output_token_count / max(0.001, elapsed)
        generated_text = out_obj.outputs[0].text

        # Append generated response to context for subsequent turns
        accumulated_context += generated_text

        turn_result = {
            "turn": turn_num,
            "prompt_tokens": prompt_token_count,
            "output_tokens": output_token_count,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(tok_per_sec, 2),
            "gpu_page_faults": delta_faults,
            "preview": generated_text[:120].replace("\n", " ").strip() + "...",
        }
        results.append(turn_result)

        print(f"Turn {turn_num} Complete:")
        print(f"  • Prompt Context Tokens: {prompt_token_count}")
        print(f"  • Generated Tokens:      {output_token_count}")
        print(f"  • Total Sequence Tokens: {total_tokens}")
        print(f"  • Turn Elapsed Time:     {elapsed:.2f} s")
        print(f"  • Generation Speed:      {tok_per_sec:.2f} tok/s")
        print(f"  • Turn GPU Page Faults:  {delta_faults}")
        print(f"  • Output Snippet:        {turn_result['preview']}")

    total_gen_time = sum(r["elapsed_sec"] for r in results)
    total_out_tokens = sum(r["output_tokens"] for r in results)
    total_faults = sum(r["gpu_page_faults"] for r in results)
    avg_speed = total_out_tokens / max(0.001, total_gen_time)

    final_summary = {
        "mode": args.mode,
        "weights_uvm": args.weights_uvm,
        "gpu_budget_gb": args.gpu_budget_gb,
        "init_sec": round(init_duration, 2),
        "total_generation_sec": round(total_gen_time, 2),
        "total_output_tokens": total_out_tokens,
        "overall_tokens_per_sec": round(avg_speed, 2),
        "total_turn_faults": total_faults,
        "turns": results,
    }

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: {args.mode.upper()}")
    print("=" * 80)
    print(f"Total Generation Time:   {total_gen_time:.2f} s")
    print(f"Total Output Tokens:     {total_out_tokens}")
    print(f"Overall Generation Speed: {avg_speed:.2f} tok/s")
    print(f"Total Page Faults:       {total_faults}")
    print("=" * 80)

    print("\n---BENCHMARK_SUMMARY_JSON_START---")
    print(json.dumps(final_summary, indent=2))
    print("---BENCHMARK_SUMMARY_JSON_END---")


if __name__ == "__main__":
    main()
