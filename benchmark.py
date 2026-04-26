import subprocess
import re
import os
import shutil
import struct
import sys
import argparse
import statistics

# ── Config ────────────────────────────────────────────────────────────────────
PASS_PATH    = "./build/UvmTrackingPass.so"
LIB_PATH     = "./libMarkAccess.cu"
CTL_PATH     = "./uvm_control_thread.cu"
CUDA_PATH    = "/usr/local/cuda"
CLANG        = "clang++-20"
LOG_DIR      = "bench_logs"
BASELINE_DIR = "baselines"
PRELOAD_SO   = "./libMallocIntercept.so"
try:
    SM_ARCH = subprocess.getoutput(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
    ).strip()
    if not SM_ARCH or not SM_ARCH.isdigit():
        SM_ARCH = "61"
except Exception:
    SM_ARCH = "61"
BASE_FLAGS = [
    "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}",
    "-fgpu-rdc", "-O2", "-I./",
    f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
]

# ── Benchmark definitions ─────────────────────────────────────────────────────
# Each entry: (display_name, source_file, kernel_id, expected_pass_strategy)
BENCHMARKS = [
    # Original suite
    ("Coalesced",  "examples/benchmark_kernel.cu",  ["0"], "BatchMarkAccess (runtime count)"),
    ("Stride",     "examples/benchmark_kernel.cu",  ["1"], "BatchMarkAccess (runtime count)"),
    ("Random",     "examples/benchmark_kernel.cu",  ["2"], "Fallback (non-affine index)"),
    ("Stencil",    "examples/benchmark_kernel.cu",  ["3"], "BatchMarkAccess x3 (runtime count)"),
    ("Atomic",     "examples/benchmark_kernel.cu",  ["4"], "Hoisted (loop-invariant ptr)"),
    # Extended suite
    ("SAXPY",      "examples/benchmark_extended.cu", ["0"], "BatchMarkAccess x2 (runtime count)"),
    ("Reduction",  "examples/benchmark_extended.cu", ["1"], "BatchMarkAccess + Hoisted"),
    ("Histogram",  "examples/benchmark_extended.cu", ["2"], "BatchMarkAccess + Fallback (scatter)"),
    ("Transpose",  "examples/benchmark_extended.cu", ["3"], "Fallback (no loop / shared mem)"),
    ("GEMV",       "examples/benchmark_extended.cu", ["4"], "BatchMarkAccess (nested loops)"),
    # Real-world: tiled GEMM from CUDA Samples
    ("MatrixMul",  "examples/matrixMul.cu",          [], "BatchMarkAccess (tiled A,B; SLE loop)"),
    # ("matrixMul", "examples/matrixMul.cu", "", "Testcase Description"),
    # ("2DConvolution", "examples/2DCONV/2DConvolution.cu", ["-mb", "5000"], "Testcase Description"),
    # ("2mm", "examples/2MM/2mm.cu", ["-mb", "200"], "Testcase Description"),
    # ("3DConvolution", "examples/3DCONV/3DConvolution.cu", ["-mb", "6000"], "Testcase Description"),
    # ("atax", "examples/ATAX/atax.cu", ["-mb", "6000"], "Testcase Description"),
    # ("bicg", "examples/BICG/bicg.cu", ["-mb", "5000"], "Testcase Description"),
    # ("covariance", "examples/COVAR/covariance.cu", [], "Testcase Description"),
    # ("fdtd2d", "examples/FDTD-2D/fdtd2d.cu", ["-mb", "2000"], "Testcase Description"),
    # ("gramschmidt", "examples/GRAMSCHM/gramschmidt.cu", [], "Testcase Description"),
    # ("needle", "examples/nw/needle.cu", ["-mb", "5600"], "Testcase Description"),
    # ("syr2k", "examples/SYR2K/syr2k.cu", [], "Testcase Description"),
    # ("syrk", "examples/SYRK/syrk.cu", [], "Testcase Description"),
    ("2DConvolution", "examples/2DCONV/2DConvolution.cu", [], "Testcase Description"),
    ("2mm", "examples/2MM/2mm.cu", [], "Testcase Description"),
    ("3DConvolution", "examples/3DCONV/3DConvolution.cu", [], "Testcase Description"),
    ("atax", "examples/ATAX/atax.cu", [], "Testcase Description"),
    ("bicg", "examples/BICG/bicg.cu", [], "Testcase Description"),
    ("covariance", "examples/COVAR/covariance.cu", ["50000000"], "Testcase Description"),
    ("fdtd2d", "examples/FDTD-2D/fdtd2d.cu", ["-mb", "2000"], "Testcase Description"),
    ("gramschmidt", "examples/GRAMSCHM/gramschmidt.cu", [], "Testcase Description"),
    ("needle", "examples/nw/needle.cu", ["-mb", "4000"], "Testcase Description"),
    ("syr2k", "examples/SYR2K/syr2k.cu", [], "Testcase Description"),
    ("syrk", "examples/SYRK/syrk.cu", [], "Testcase Description"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, **kw)

def compile_pair(src, out_normal, out_instrumented, rdynamic=False, mode=0):
    """Compile one source file into a normal and an instrumented binary."""
    common = BASE_FLAGS + [src]
    if rdynamic:
        common = common + ["-rdynamic"]

    r = run([CLANG] + common + ["-o", out_normal])
    if r.returncode != 0:
        print(f"\n  ✗ Normal compile failed for {src}")
        print(r.stderr[-2000:])
        return False

    inst_flags = common + [
        "-DTRACKING_ENABLED",
        f"-fpass-plugin={PASS_PATH}",
        LIB_PATH,
        CTL_PATH,
    ]
    if mode != 0:
        inst_flags = [f"-DUVM_TRACKING_MODE={mode}"] + inst_flags

    r = run([CLANG] + inst_flags + ["-o", out_instrumented])
    if r.returncode != 0:
        print(f"\n  ✗ Instrumented compile failed for {src}")
        print(r.stderr[-2000:])
        return False

    return True

def extract_time(output):
    patterns = [
        r"BENCHMARK_TIME:\s*([\d.]+)",     # your format
        r"GPU(?:\s+Runtime)?[:\s]+([\d.]+)"  # matches "GPU: 0.123" or "GPU Runtime: 0.123"
    ]
    
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            return float(m.group(1))
    return None

# ── Pagelog helpers ───────────────────────────────────────────────────────────
_HEADER_FMT = "<IHHHHIIIQ"   # magic, version, l1_entries, l2_entries, l3_bytes,
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  #   l1_shift, l2_shift, l3_shift, num_leaves
_INDEX_FMT   = "<HHQ"        # l1_idx, l2_idx, offset
_INDEX_SIZE  = struct.calcsize(_INDEX_FMT)
_PAGELOG_MAGIC = 0x50474C47

def parse_pagelog(path):
    """Read a binary pagelog and return a set of accessed page base addresses."""
    with open(path, "rb") as f:
        raw = f.read()
    (magic, _ver, _l1e, _l2e, l3_bytes,
     l1_shift, l2_shift, l3_shift, num_leaves) = struct.unpack_from(_HEADER_FMT, raw)
    if magic != _PAGELOG_MAGIC:
        raise ValueError(f"bad pagelog magic 0x{magic:08X} in {path}")
    pages = set()
    for k in range(num_leaves):
        l1_idx, l2_idx, data_off = struct.unpack_from(
            _INDEX_FMT, raw, _HEADER_SIZE + k * _INDEX_SIZE)
        for w in range(l3_bytes // 8):
            word = struct.unpack_from("<Q", raw, data_off + w * 8)[0]
            if not word:
                continue
            for b in range(64):
                if word & (1 << b):
                    l3_off = w * 64 + b
                    pages.add((l1_idx << l1_shift)
                               | (l2_idx << l2_shift)
                               | (l3_off  << l3_shift))
    return pages

def allocation_fingerprint(pages):
    """VA-independent summary: (total_pages, sorted list of contiguous cluster sizes)."""
    if not pages:
        return 0, []
    PAGE = 4096
    sp = sorted(pages)
    sizes, sz = [], 1
    for prev, cur in zip(sp, sp[1:]):
        if cur == prev + PAGE:
            sz += 1
        else:
            sizes.append(sz)
            sz = 1
    sizes.append(sz)
    return len(pages), sorted(sizes)

def compare_pagelogs(baseline_path, current_path):
    """Compare two pagelogs by fingerprint. Returns (ok: bool, detail: str, page_count: int)."""
    try:
        b_pages = parse_pagelog(baseline_path)
        c_pages = parse_pagelog(current_path)
    except Exception as e:
        return False, f"parse error: {e}", 0
    b_total, b_sizes = allocation_fingerprint(b_pages)
    c_total, c_sizes = allocation_fingerprint(c_pages)
    if b_total != c_total:
        return False, f"page count {b_total}→{c_total}", c_total
    if b_sizes != c_sizes:
        return False, f"cluster layout differs ({len(b_sizes)} vs {len(c_sizes)} allocs)", c_total
    return True, f"{c_total} pages, {len(c_sizes)} alloc(s)", c_total

def parse_results_md(path):
    """Parse a results markdown file and return a dict of kernel -> {clean, tracked, overhead, baseline}."""
    results = {}
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the table (skip header lines until we see the separator)
    in_table = False
    for i, line in enumerate(lines):
        if line.startswith("| :---"):
            in_table = True
            continue
        if in_table and line.startswith("|") and "Kernel" not in line:
            # Parse row: | Kernel | Strategy | Clean | Tracked | Overhead | Baseline |
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 5:
                name = parts[0]
                clean = parts[2]
                tracked = parts[3]
                overhead = parts[4]
                baseline = parts[5] if len(parts) > 5 else ""
                results[name] = {
                    "clean": clean,
                    "tracked": tracked,
                    "overhead": overhead,
                    "baseline": baseline,
                }
    return results

def diff_results(file1, file2):
    """Compare two result markdown files and print differences."""
    print(f"── Diffing results: {file1} vs {file2} ────────────────────────────────────────")

    if not os.path.exists(file1):
        print(f"Error: {file1} not found")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: {file2} not found")
        sys.exit(1)

    r1 = parse_results_md(file1)
    r2 = parse_results_md(file2)

    all_kernels = sorted(set(r1.keys()) | set(r2.keys()))

    print(f"\n{'Kernel':<14} | {'File1':<30} | {'File2':<30} | {'Diff'}")
    print("-" * 100)

    diff_count = 0
    for kernel in all_kernels:
        d1 = r1.get(kernel, {})
        d2 = r2.get(kernel, {})

        v1 = d1.get("overhead", "N/A")
        v2 = d2.get("overhead", "N/A")

        # Try to parse as float for comparison
        try:
            f1 = float(v1.replace("%", ""))
            try:
                f2 = float(v2.replace("%", ""))
                diff = f2 - f1
                diff_str = f"{diff:+.2f}%"
            except:
                diff_str = "N/A"
        except:
            diff_str = "N/A"

        if v1 != v2:
            diff_count += 1
            marker = "←"
        else:
            marker = "="

        print(f"{kernel:<14} | {v1:<30} | {v2:<30} | {marker} {diff_str}")

    print("-" * 100)
    print(f"Total differences: {diff_count}/{len(all_kernels)} kernels")

def safe_slug(path):
    # Make a filesystem-safe unique name from a source path relative to examples/
    rel = os.path.relpath(os.path.normpath(path), "examples")
    rel = os.path.splitext(rel)[0]
    rel = rel.replace(os.sep, "_")
    rel = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
    return rel

def find_cu_files(root="examples"):
    cu_files = []
    skip_files = {
        "needle_kernel.cu",   # add more here if needed
        "sgemm_cutlass.cu",
        "test_kernel.cu",
        "test_kernel_loop.cu",
        "long_running_test.cu",
        "malloc_intercept_example.cu",
        "slow-cuda-kernel.cu",
        "thread_mode_tests.cu",
        "benchmark_kernel_stride.cu",
        "thread_mode_tests.cu",
        "malloc_intercept_example.cu"
        }
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".cu") and f not in skip_files:
                cu_files.append(os.path.normpath(os.path.join(dirpath, f)))
    return sorted(cu_files)

def extract_time(output):
    # Take the LAST timing occurrence, because some benchmarks print timing more than once.
    pattern = re.compile(
        r"BENCHMARK_TIME:\s*([\d.eE+-]+)|GPU(?:\s+Runtime)?[:\s]+([\d.eE+-]+)"
    )
    last = None
    for m in pattern.finditer(output):
        last = m.group(1) or m.group(2)
    return float(last) if last is not None else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="UVM Access Tracking Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 benchmark.py                           # Run benchmarks, output to results.md
  python3 benchmark.py results.md                # Custom output filename
  python3 benchmark.py --rebuild                 # Force rebuild binaries
  python3 benchmark.py --generate-baseline       # Generate baseline pagelogs
  python3 benchmark.py --preload                 # Use LD_PRELOAD for malloc intercept
  python3 benchmark.py --mode preload-only       # Use PRELOAD_ONLY mode
  python3 benchmark.py --check                   # Run pagelog correctness checks
  python3 benchmark.py --diff a.md b.md          # Diff two result files
  python3 benchmark.py -n 5                      # Run each benchmark 5 times
  python3 benchmark.py -n 5 --metric median      # Use median of 5 runs
        """
    )
    parser.add_argument("output_file", nargs="?", default="results.md",
                        help="Output markdown filename (default: results.md)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of all binaries")
    parser.add_argument("--generate-baseline", action="store_true",
                        help="Generate baseline pagelogs for correctness checking")
    parser.add_argument("--check", action="store_true",
                        help="Run pagelog correctness checks after benchmarks")
    parser.add_argument("--preload", action="store_true",
                        help="Use LD_PRELOAD for malloc intercept")
    parser.add_argument("--mode", choices=["no-preload", "preload-alloc", "preload-only"],
                        default="no-preload",
                        help="Tracking mode (default: no-preload)")
    parser.add_argument("--iterations", "-n", type=int, default=1,
                        help="Number of iterations to run each benchmark (default: 1)")
    parser.add_argument("--metric", choices=["mean", "median", "min"], default="mean",
                        help="Metric to compute overhead from multiple runs (default: mean)")
    parser.add_argument("--diff", nargs=2, metavar=("FILE1", "FILE2"),
                        help="Diff two result markdown files")

    args = parser.parse_args()

    if args.diff:
        diff_results(args.diff[0], args.diff[1])
        return

    force_rebuild     = args.rebuild
    gen_baseline      = args.generate_baseline
    check_pagelogs    = args.check
    use_preload       = args.preload
    mode_name         = args.mode
    md_filename       = args.output_file
    n_iterations      = args.iterations
    metric_name       = args.metric

    mode_map = {"no-preload": 0, "preload-alloc": 1, "preload-only": 2}
    tracking_mode = mode_map[mode_name]

    # Modes 1 and 2 require the preload wrapper
    if tracking_mode in (1, 2):
        use_preload = True

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("build", exist_ok=True)
    if gen_baseline:
        os.makedirs(BASELINE_DIR, exist_ok=True)

    # Build preload env early so both baseline mode and run mode can use it
    preload_env = None
    if use_preload:
        preload_env = os.environ.copy()
        preload_env["LD_PRELOAD"] = os.path.abspath(PRELOAD_SO)

    # ── Build preload library if needed ────────────────────────────────────────
    if use_preload:
        print("── Preload mode enabled ─────────────────────────────────────────────-")
        if not os.path.exists(PRELOAD_SO) or force_rebuild:
            print(f"  Building {PRELOAD_SO}...", end=" ", flush=True)
            r = run([
                "clang++-20", "-shared", "-fPIC", "-O2",
                f"-I{CUDA_PATH}/include",
                "libMallocIntercept.cpp",
                "-o", PRELOAD_SO,
                "-ldl"
            ])
            if r.returncode != 0:
                print("FAILED")
                print(r.stderr[-2000:])
                sys.exit(1)
            print("ok")
        else:
            print(f"  {PRELOAD_SO}: up-to-date")

    # ── Build benchmark list: original + auto-discovered .cu files ─────────────
    existing_sources = {os.path.normpath(src) for _, src, _, _ in BENCHMARKS}
    auto_benchmarks = []

    for src in find_cu_files("examples"):
        src = os.path.normpath(src)
        if src not in existing_sources:
            # Unique display name derived from path, so files in different folders do not collide.
            name = safe_slug(src)
            auto_benchmarks.append((name, src, [], ""))

    benchmarks = [
        (name, os.path.normpath(src), kid, strategy)
        for (name, src, kid, strategy) in (BENCHMARKS + auto_benchmarks)
    ]
    print("\nFinal benchmark sources:")
    for _, src, _, _ in benchmarks:
        print(src)
    # Build source set from the combined benchmark list
    sources = sorted({src for _, src, _, _ in benchmarks})
    binaries = {}  # src -> (normal_bin, instrumented_bin)

    print("── Compiling ─────────────────────────────────────────────────────────")
    for src in sources:
        stem = safe_slug(src)
        normal_bin = f"build/{stem}Normal"
        # Include mode in instrumented binary name so mode changes don't require --rebuild
        if tracking_mode == 0:
            instrumented_bin = f"build/{stem}Instrumented"
        else:
            instrumented_bin = f"build/{stem}Instrumented_mode{tracking_mode}"
        binaries[src] = (normal_bin, instrumented_bin)

        src_mtime  = os.path.getmtime(src) if os.path.exists(src) else 0
        pass_mtime = os.path.getmtime(PASS_PATH) if os.path.exists(PASS_PATH) else 0
        lib_mtime  = os.path.getmtime(LIB_PATH) if os.path.exists(LIB_PATH) else 0
        bin_mtime  = min(
            os.path.getmtime(normal_bin) if os.path.exists(normal_bin) else 0,
            os.path.getmtime(instrumented_bin) if os.path.exists(instrumented_bin) else 0,
        )
        needs_build = force_rebuild or (max(src_mtime, pass_mtime, lib_mtime) > bin_mtime)

        if use_preload and not os.path.exists(normal_bin):
            needs_build = True

        if needs_build:
            print(f"  Building {stem}...", end=" ", flush=True)
            ok = compile_pair(src, normal_bin, instrumented_bin,
                              rdynamic=use_preload, mode=tracking_mode)
            print("ok" if ok else "FAILED")
            if not ok:
                print("Aborting.")
                sys.exit(1)
        else:
            print(f"  {stem}: up-to-date (pass --rebuild to force)")

    # ── Generate-baseline mode ─────────────────────────────────────────────────
    if gen_baseline:
        print("\n── Generating baselines ──────────────────────────────────────────────")
        for name, src, kid, strategy in benchmarks:
            src = os.path.normpath(src)
            _, instrumented_bin = binaries[src]
            baseline_path = os.path.join(BASELINE_DIR, f"{name.lower()}.pagelog")
            print(f"  {name:<12}", end=" ", flush=True)

            # Clean up stale access_log.bin before each baseline generation
            if os.path.exists("access_log.bin"):
                os.remove("access_log.bin")
            cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else [])
            out_i = run(cmd_i, env=preload_env)

            if out_i.returncode != 0:
                print("FAILED (binary error)")
                print(out_i.stderr[-500:])
                continue

            if os.path.exists("access_log.bin"):
                os.rename("access_log.bin", baseline_path)
                pages, clusters = allocation_fingerprint(parse_pagelog(baseline_path))
                print(f"saved → {baseline_path}  ({pages} pages, {len(clusters)} alloc(s))")
            else:
                print("FAILED (no access_log.bin produced)")

        print(f"\nBaselines written to '{BASELINE_DIR}/'")
        return

    # ── Run phase ──────────────────────────────────────────────────────────────
    print("\n── Running benchmarks ────────────────────────────────────────────────")
    results = []

    for name, src, kid, strategy in benchmarks:
        src = os.path.normpath(src)
        # print("Looking for:", src)
        # print("Available keys:", list(binaries.keys())[:5])
        custom_input = {}
        normal_bin, instrumented_bin = binaries[src]
        log_path      = os.path.join(LOG_DIR, f"{safe_slug(src)}.log")
        pagelog_path  = os.path.join(LOG_DIR, f"{safe_slug(src)}.pagelog")
        baseline_path = os.path.join(BASELINE_DIR, f"{safe_slug(src)}.pagelog")

        print(f"  {name:<12}", end=" ", flush=True)
        # extra_args = ["-mb", "1000"]
        # cmd_n = [f"./{normal_bin}"] + ([str(kid)] if kid is not None else []) + extra_args
        # cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else []) + extra_args
        # flags = kid
        # cmd_n = [f"./{normal_bin}"]       + ([str(kid)] if kid is not None else [])
        # cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else [])
        cmd_n = [f"./{normal_bin}"]       + kid # if kid is not None else []
        cmd_i = [f"./{instrumented_bin}"] + kid # if kid is not None else []
        
        print(f"Running normal as: {cmd_n}")
        print(f"Running instrumented as: {cmd_i}")

        # Run multiple iterations
        times_clean = []
        times_track = []
        # Ensure no stale pagelog from a previous run/suite
        if os.path.exists(pagelog_path):
            os.remove(pagelog_path)
        for i in range(n_iterations):
            # Clean up any leftover access_log.bin before each run
            if os.path.exists("access_log.bin"):
                os.remove("access_log.bin")
            out_n = run(cmd_n, env=preload_env)
            out_i = run(cmd_i, env=preload_env)

            t_clean = extract_time(out_n.stdout)
            t_track = extract_time(out_i.stdout)
            if t_clean is not None:
                times_clean.append(t_clean)
            if t_track is not None:
                times_track.append(t_track)

        # Brief cooldown between benchmarks to let GPU/driver settle
        # (helps avoid intermittent CUDA context races)
        import time
        time.sleep(0.1)

        # Only keep the last pagelog (others are overwritten)
        if os.path.exists("access_log.bin"):
            os.rename("access_log.bin", pagelog_path)

        with open(log_path, "w") as lf:
            lf.write(f"=== CLEAN ===\n{out_n.stdout}\n{out_n.stderr}\n\n")
            lf.write(f"=== TRACKED ===\n{out_i.stdout}\n{out_i.stderr}\n\n")

        # Compute metric from multiple runs
        if times_clean and times_track:
            if metric_name == "mean":
                t_clean = statistics.mean(times_clean)
                t_track = statistics.mean(times_track)
            elif metric_name == "median":
                t_clean = statistics.median(times_clean)
                t_track = statistics.median(times_track)
            else:  # min
                t_clean = min(times_clean)
                t_track = min(times_track)
        else:
            t_clean = None
            t_track = None

        # ── Baseline check ────────────────────────────────────────────────────
        page_count = 0
        if os.path.exists(baseline_path) and os.path.exists(pagelog_path):
            bl_ok, bl_detail, page_count = compare_pagelogs(baseline_path, pagelog_path)
        elif not os.path.exists(baseline_path):
            bl_ok, bl_detail = None, "no baseline"
        else:
            bl_ok, bl_detail = False, "no pagelog generated"

        if t_clean is not None and t_track is not None and t_clean != 0:
            overhead = (t_track - t_clean) / t_clean * 100
            results.append((name, strategy, t_clean, t_track, overhead, bl_ok, bl_detail, page_count))
            marker = "✓" if overhead < 50 else ("~" if overhead < 200 else "✗")
            bl_str = (f"  baseline={'PASS' if bl_ok else 'FAIL'}({bl_detail})"
                      if bl_ok is not None else f"  [{bl_detail}]")
            iter_str = f" (n={n_iterations}, {metric_name})" if n_iterations > 1 else ""
            print(f"{marker}  clean={t_clean:.3f}ms  tracked={t_track:.3f}ms"
                  f"  overhead={overhead:+.1f}%{iter_str}{bl_str}")
        else:
            results.append((name, strategy, None, None, None, bl_ok, bl_detail, 0))
            print("FAILED (check log)")

    # ── Results table ──────────────────────────────────────────────────────────
    hdr = "| {:<14}| {:<40}| {:<12}| {:<12}| {:<10}| {:<8}| {:<22}|".format(
        "Kernel", "Pass strategy", "Clean (ms)", "Tracked (ms)", "Overhead", "Pages", "Baseline")
    sep = ("| :---" + " " * 9 + "| :---" + " " * 35 + "| :---" + " " * 7
           + "| :---" + " " * 7 + "| :---" + " " * 5 + "| :---" + " " * 3
           + "| :---" + " " * 17 + "|")
    rows = []
    for name, strategy, tc, tt, oh, bl_ok, bl_detail, page_count in results:
        bl_cell = ("PASS" if bl_ok else ("FAIL: " + bl_detail if bl_ok is False else bl_detail))
        pages_str = str(page_count) if page_count > 0 else "N/A"
        if tc is not None:
            rows.append("| {:<14}| {:<40}| {:<12}| {:<12}| {:>+9.2f}% | {:<8}| {:<22}|".format(
                name, strategy, f"{tc:.3f}", f"{tt:.3f}", oh, pages_str, bl_cell))
        else:
            rows.append(f"| {name:<14}| {strategy:<40}| FAILED     | FAILED     | N/A       | N/A     | {bl_cell:<22}|")

    table = "\n".join([hdr, sep] + rows)
    md = f"# Benchmark Results\n\nSM arch: sm_{SM_ARCH}\nMode: {mode_name}\n\n{table}\n"

    with open(md_filename, "w") as f:
        f.write(md)

    # ── Baseline summary ───────────────────────────────────────────────────────
    bl_results = [(name, ok, det) for name, _, _, _, _, ok, det, _ in results]
    n_checked = sum(1 for _, ok, _ in bl_results if ok is not None)
    n_passed  = sum(1 for _, ok, _ in bl_results if ok is True)
    n_failed  = sum(1 for _, ok, _ in bl_results if ok is False)

    # Total pages touched
    total_pages = sum(pc for _, _, _, _, _, _, _, pc in results if pc > 0)

    print(f"\n── Summary ───────────────────────────────────────────────────────────")
    print(md)
    print(f"Full logs in '{LOG_DIR}/', page access logs in *.pagelog")
    print(f"Markdown results written to: {md_filename}")
    if total_pages > 0:
        print(f"Total pages touched: {total_pages}")

    if n_checked:
        print(f"Baseline checks: {n_passed}/{n_checked} passed"
              + (f"  ← {n_failed} FAILED" if n_failed else ""))
        for name, ok, det in bl_results:
            if ok is False:
                print(f"  FAIL {name}: {det}")

if __name__ == "__main__":
    main()
