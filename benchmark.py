import subprocess
import re
import os
import shutil
import sys

# ── Config ────────────────────────────────────────────────────────────────────
PASS_PATH  = "./build/UvmTrackingPass.so"
LIB_PATH   = "./libMarkAccess.cu"
CUDA_PATH  = "/usr/local/cuda"
CLANG      = "clang++-20"
LOG_DIR    = "bench_logs"
SM_ARCH    = subprocess.getoutput(
    "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
)
BASE_FLAGS = [
    "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}",
    "-fgpu-rdc", "-O2", "-I./include",
    f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
]

# ── Benchmark definitions ─────────────────────────────────────────────────────
# Each entry: (display_name, source_file, kernel_id, expected_pass_strategy)
BENCHMARKS = [
    # Original suite
    ("Coalesced",  "examples/benchmark_kernel.cu",  0, "BatchMarkAccess (runtime count)"),
    ("Stride",     "examples/benchmark_kernel.cu",  1, "BatchMarkAccess (runtime count)"),
    ("Random",     "examples/benchmark_kernel.cu",  2, "Fallback (non-affine index)"),
    ("Stencil",    "examples/benchmark_kernel.cu",  3, "BatchMarkAccess x3 (runtime count)"),
    ("Atomic",     "examples/benchmark_kernel.cu",  4, "Hoisted (loop-invariant ptr)"),
    # Extended suite
    ("SAXPY",      "examples/benchmark_extended.cu", 0,    "BatchMarkAccess x2 (runtime count)"),
    ("Reduction",  "examples/benchmark_extended.cu", 1,    "BatchMarkAccess + Hoisted"),
    ("Histogram",  "examples/benchmark_extended.cu", 2,    "BatchMarkAccess + Fallback (scatter)"),
    ("Transpose",  "examples/benchmark_extended.cu", 3,    "Fallback (no loop / shared mem)"),
    ("GEMV",       "examples/benchmark_extended.cu", 4,    "BatchMarkAccess (nested loops)"),
    # Real-world: tiled GEMM from CUDA Samples
    ("MatrixMul",  "examples/matrixMul.cu",          None, "BatchMarkAccess (tiled A,B; SLE loop)"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, **kw)

def compile_pair(src, out_normal, out_instrumented):
    """Compile one source file into a normal and an instrumented binary."""
    common = BASE_FLAGS + [src]

    r = run([CLANG] + common + ["-o", out_normal])
    if r.returncode != 0:
        print(f"\n  ✗ Normal compile failed for {src}")
        print(r.stderr[-2000:])
        return False

    r = run([CLANG] + common + [
        "-DTRACKING_ENABLED",
        f"-fpass-plugin={PASS_PATH}",
        LIB_PATH,
        "-o", out_instrumented,
    ])
    if r.returncode != 0:
        print(f"\n  ✗ Instrumented compile failed for {src}")
        print(r.stderr[-2000:])
        return False

    return True

def extract_time(output):
    m = re.search(r"BENCHMARK_TIME: ([\d.]+)", output)
    return float(m.group(1)) if m else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    force_rebuild = "--rebuild" in sys.argv
    check_pagelogs = "--check" in sys.argv

    # Parse optional output markdown filename
    md_filename = "results.md"
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if args:
        md_filename = args[0]

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("build",  exist_ok=True)

    # ── Compile phase ──────────────────────────────────────────────────────────
    sources = sorted({src for _, src, _, _ in BENCHMARKS})
    binaries = {}  # src → (normal_bin, instrumented_bin)

    print("── Compiling ─────────────────────────────────────────────────────────")
    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]
        normal_bin       = f"build/{stem}Normal"
        instrumented_bin = f"build/{stem}Instrumented"
        binaries[src]    = (normal_bin, instrumented_bin)

        src_mtime  = os.path.getmtime(src) if os.path.exists(src) else 0
        pass_mtime = os.path.getmtime(PASS_PATH) if os.path.exists(PASS_PATH) else 0
        lib_mtime  = os.path.getmtime(LIB_PATH)  if os.path.exists(LIB_PATH)  else 0
        bin_mtime  = min(
            os.path.getmtime(normal_bin)       if os.path.exists(normal_bin)       else 0,
            os.path.getmtime(instrumented_bin) if os.path.exists(instrumented_bin) else 0,
        )
        needs_build = force_rebuild or (max(src_mtime, pass_mtime, lib_mtime) > bin_mtime)

        if needs_build:
            print(f"  Building {stem}...", end=" ", flush=True)
            ok = compile_pair(src, normal_bin, instrumented_bin)
            print("ok" if ok else "FAILED")
            if not ok:
                print("Aborting.")
                sys.exit(1)
        else:
            print(f"  {stem}: up-to-date (pass --rebuild to force)")

    # ── Run phase ──────────────────────────────────────────────────────────────
    print("\n── Running benchmarks ────────────────────────────────────────────────")
    results = []

    for name, src, kid, strategy in BENCHMARKS:
        normal_bin, instrumented_bin = binaries[src]
        log_path = os.path.join(LOG_DIR, f"{name.lower()}.log")
        print(f"  {name:<12}", end=" ", flush=True)

        cmd_n = [f"./{normal_bin}"]       + ([str(kid)] if kid is not None else [])
        cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else [])
        out_n = run(cmd_n)
        out_i = run(cmd_i)
        # Move any generated pagelog
        if os.path.exists("access_log.bin"):
            os.rename("access_log.bin", f"{name.lower()}.pagelog")

        with open(log_path, "w") as lf:
            lf.write(f"=== CLEAN ===\n{out_n.stdout}\n{out_n.stderr}\n\n")
            lf.write(f"=== TRACKED ===\n{out_i.stdout}\n{out_i.stderr}\n\n")

        t_clean = extract_time(out_n.stdout)
        t_track = extract_time(out_i.stdout)

        if t_clean is not None and t_track is not None:
            overhead = (t_track - t_clean) / t_clean * 100
            results.append((name, strategy, t_clean, t_track, overhead))
            marker = "✓" if overhead < 50 else ("~" if overhead < 200 else "✗")
            print(f"{marker}  clean={t_clean:.3f}ms  tracked={t_track:.3f}ms  overhead={overhead:+.1f}%")
        else:
            results.append((name, strategy, None, None, None))
            print("FAILED (check log)")

    # ── Results table ──────────────────────────────────────────────────────────
    hdr  = "| {:<14}| {:<40}| {:<12}| {:<12}| {:<10}|".format(
        "Kernel", "Pass strategy", "Clean (ms)", "Tracked (ms)", "Overhead")
    sep  = "| " + ":---" + " " * 10 + "| " + ":---" + " " * 36 + "| " + ":---" + " " * 8 + "| " + ":---" + " " * 8 + "| " + ":---" + " " * 6 + "|"
    rows = []
    for name, strategy, tc, tt, oh in results:
        if tc is not None:
            rows.append("| {:<14}| {:<40}| {:<12}| {:<12}| {:>+9.2f}% |".format(
                name, strategy, f"{tc:.3f}", f"{tt:.3f}", oh))
        else:
            rows.append(f"| {name:<14}| {strategy:<40}| FAILED     | FAILED     | N/A       |")

    table = "\n".join([hdr, sep] + rows)
    md = f"# Benchmark Results\n\nSM arch: sm_{SM_ARCH}\n\n{table}\n"

    with open(md_filename, "w") as f:
        f.write(md)

    print(f"\n── Summary ───────────────────────────────────────────────────────────")
    print(md)
    print(f"Full logs in '{LOG_DIR}/', page access logs in *.pagelog")
    print(f"Markdown results written to: {md_filename}")

    # Optional: run pagelog correctness check using scripts/run_pgelog_test.py
    if check_pagelogs:
        try:
            from scripts.run_pgelog_test import PagelogTester
        except Exception as e:
            print(f"\n✗ Could not import PagelogTester: {e}")
            sys.exit(2)

        print("\n── Pagelog correctness check (--check) ───────────────────────────────")
        tester = PagelogTester(root_dir=os.getcwd())

        # Backup generated pagelogs into tester.current_dir (will copy *.pagelog)
        ok = tester.backup_generated_pagelogs()
        if not ok:
            print("\n✗ No pagelog files found to check")
            sys.exit(2)

        passed, results = tester.compare_pagelogs()
        tester.print_summary(passed, results)

        if not passed:
            print("\n✗ Pagelog verification failed")
            sys.exit(3)
        else:
            print("\n✓ Pagelog verification passed")
            # fall through and exit normally

if __name__ == "__main__":
    main()
