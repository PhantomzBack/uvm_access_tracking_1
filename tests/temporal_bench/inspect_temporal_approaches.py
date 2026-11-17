import subprocess
import re
import sys
import os

CUDA_BIN = "./build/temporal_microbench"

def run_bench(milestones):
    cmd = [CUDA_BIN, str(milestones)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running benchmark with {milestones} milestones:")
        print(res.stderr)
        return None

    # Parse table output
    lines = res.stdout.strip().split("\n")
    data = []
    for line in lines:
        if "|" in line and "WORKLOAD" not in line and "---" not in line and "===" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                name = parts[0]
                clean = float(parts[1].replace("ms", "").strip())
                tripwire = float(parts[2].replace("ms", "").strip())
                tiled = float(parts[3].replace("ms", "").strip())
                trip_oh = float(parts[4].replace("%", "").strip())
                tiled_oh = float(parts[5].replace("%", "").strip())
                data.append({
                    "workload": name,
                    "clean_ms": clean,
                    "tripwire_ms": tripwire,
                    "tiled_ms": tiled,
                    "tripwire_overhead_pct": trip_oh,
                    "tiled_overhead_pct": tiled_oh,
                })
    return data

def main():
    print("=" * 80)
    print(" TEMPORAL TRACKING DECISION EXPERIMENT: TRIPWIRES VS TILED SIGNALING")
    print("=" * 80)

    milestone_counts = [4, 8, 16, 32]
    all_results = {}

    for m in milestone_counts:
        print(f"\n--> Running with {m} temporal milestones / kernel...")
        data = run_bench(m)
        if data:
            all_results[m] = data

    print("\n" + "=" * 80)
    print(" SCALING COMPARISON TABLE ACROSS TEMPORAL RESOLUTIONS")
    print("=" * 80)

    for m, rows in all_results.items():
        print(f"\n[ Resolution: {m} Milestones / Epochs ]")
        print(f"{'Workload':<35} | {'Clean':<10} | {'Tripwire (Fault)':<18} | {'Tiled (PCIe Atomic)':<20}")
        print("-" * 90)
        for r in rows:
            print(f"{r['workload']:<35} | {r['clean_ms']:>7.3f} ms | {r['tripwire_ms']:>7.3f} ms ({r['tripwire_overhead_pct']:+6.1f}%) | {r['tiled_ms']:>7.3f} ms ({r['tiled_overhead_pct']:+6.1f}%)")

if __name__ == "__main__":
    main()
