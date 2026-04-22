import subprocess
import re
import os
import shutil
import struct
import sys

# ── Config ────────────────────────────────────────────────────────────────────
PASS_PATH    = "./build/UvmTrackingPass.so"
LIB_PATH     = "./libMarkAccess.cu"
CUDA_PATH    = "/usr/local/cuda"
CLANG        = "clang++-20"
LOG_DIR      = "bench_logs"
BASELINE_DIR = "baselines"
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
    """Compare two pagelogs by fingerprint. Returns (ok: bool, detail: str)."""
    try:
        b_pages = parse_pagelog(baseline_path)
        c_pages = parse_pagelog(current_path)
    except Exception as e:
        return False, f"parse error: {e}"
    b_total, b_sizes = allocation_fingerprint(b_pages)
    c_total, c_sizes = allocation_fingerprint(c_pages)
    if b_total != c_total:
        return False, f"page count {b_total}→{c_total}"
    if b_sizes != c_sizes:
        return False, f"cluster layout differs ({len(b_sizes)} vs {len(c_sizes)} allocs)"
    return True, f"{c_total} pages, {len(c_sizes)} alloc(s)"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    force_rebuild     = "--rebuild"           in sys.argv
    gen_baseline      = "--generate-baseline" in sys.argv
    check_pagelogs = "--check" in sys.argv

    # Parse optional output markdown filename
    md_filename = "results.md"
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if args:
        md_filename = args[0]

    os.makedirs(LOG_DIR,      exist_ok=True)
    os.makedirs("build",      exist_ok=True)
    if gen_baseline:
        os.makedirs(BASELINE_DIR, exist_ok=True)

    # ── Compile phase ──────────────────────────────────────────────────────────
    sources  = sorted({src for _, src, _, _ in BENCHMARKS})
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

    # ── Generate-baseline mode ─────────────────────────────────────────────────
    if gen_baseline:
        print("\n── Generating baselines ──────────────────────────────────────────────")
        for name, src, kid, strategy in BENCHMARKS:
            _, instrumented_bin = binaries[src]
            baseline_path = os.path.join(BASELINE_DIR, f"{name.lower()}.pagelog")
            print(f"  {name:<12}", end=" ", flush=True)

            cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else [])
            out_i = run(cmd_i)

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
    results = []  # (name, strategy, t_clean, t_track, overhead, baseline_ok, baseline_detail)

    for name, src, kid, strategy in BENCHMARKS:
        normal_bin, instrumented_bin = binaries[src]
        log_path      = os.path.join(LOG_DIR, f"{name.lower()}.log")
        pagelog_path  = f"{name.lower()}.pagelog"
        baseline_path = os.path.join(BASELINE_DIR, f"{name.lower()}.pagelog")
        print(f"  {name:<12}", end=" ", flush=True)

        cmd_n = [f"./{normal_bin}"]       + ([str(kid)] if kid is not None else [])
        cmd_i = [f"./{instrumented_bin}"] + ([str(kid)] if kid is not None else [])
        out_n = run(cmd_n)
        out_i = run(cmd_i)

        if os.path.exists("access_log.bin"):
            os.rename("access_log.bin", pagelog_path)

        with open(log_path, "w") as lf:
            lf.write(f"=== CLEAN ===\n{out_n.stdout}\n{out_n.stderr}\n\n")
            lf.write(f"=== TRACKED ===\n{out_i.stdout}\n{out_i.stderr}\n\n")

        t_clean = extract_time(out_n.stdout)
        t_track = extract_time(out_i.stdout)

        # ── Baseline check ────────────────────────────────────────────────────
        if os.path.exists(baseline_path) and os.path.exists(pagelog_path):
            bl_ok, bl_detail = compare_pagelogs(baseline_path, pagelog_path)
        elif not os.path.exists(baseline_path):
            bl_ok, bl_detail = None, "no baseline"
        else:
            bl_ok, bl_detail = False, "no pagelog generated"

        if t_clean is not None and t_track is not None:
            overhead = (t_track - t_clean) / t_clean * 100
            results.append((name, strategy, t_clean, t_track, overhead, bl_ok, bl_detail))
            marker = "✓" if overhead < 50 else ("~" if overhead < 200 else "✗")
            bl_str = (f"  baseline={'PASS' if bl_ok else 'FAIL'}({bl_detail})"
                      if bl_ok is not None else f"  [{bl_detail}]")
            print(f"{marker}  clean={t_clean:.3f}ms  tracked={t_track:.3f}ms"
                  f"  overhead={overhead:+.1f}%{bl_str}")
        else:
            results.append((name, strategy, None, None, None, bl_ok, bl_detail))
            print("FAILED (check log)")

    # ── Results table ──────────────────────────────────────────────────────────
    hdr = "| {:<14}| {:<40}| {:<12}| {:<12}| {:<10}| {:<22}|".format(
        "Kernel", "Pass strategy", "Clean (ms)", "Tracked (ms)", "Overhead", "Baseline")
    sep = ("| :---" + " " * 9 + "| :---" + " " * 35 + "| :---" + " " * 7
           + "| :---" + " " * 7 + "| :---" + " " * 5 + "| :---" + " " * 17 + "|")
    rows = []
    for name, strategy, tc, tt, oh, bl_ok, bl_detail in results:
        bl_cell = ("PASS" if bl_ok else ("FAIL: " + bl_detail if bl_ok is False else bl_detail))
        if tc is not None:
            rows.append("| {:<14}| {:<40}| {:<12}| {:<12}| {:>+9.2f}% | {:<22}|".format(
                name, strategy, f"{tc:.3f}", f"{tt:.3f}", oh, bl_cell))
        else:
            rows.append(f"| {name:<14}| {strategy:<40}| FAILED     | FAILED     | N/A       | {bl_cell:<22}|")

    table = "\n".join([hdr, sep] + rows)
    md = f"# Benchmark Results\n\nSM arch: sm_{SM_ARCH}\n\n{table}\n"

    with open(md_filename, "w") as f:
        f.write(md)

    # ── Baseline summary ───────────────────────────────────────────────────────
    bl_results = [(name, ok, det) for name, _, _, _, _, ok, det in results]
    n_checked = sum(1 for _, ok, _ in bl_results if ok is not None)
    n_passed  = sum(1 for _, ok, _ in bl_results if ok is True)
    n_failed  = sum(1 for _, ok, _ in bl_results if ok is False)

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
    if n_checked:
        print(f"Baseline checks: {n_passed}/{n_checked} passed"
              + (f"  ← {n_failed} FAILED" if n_failed else ""))
        for name, ok, det in bl_results:
            if ok is False:
                print(f"  FAIL {name}: {det}")

if __name__ == "__main__":
    main()
