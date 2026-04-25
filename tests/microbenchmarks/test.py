#!/usr/bin/env python3
"""
Microbenchmark correctness test.

Compiles basic_kernels.cu (coalesced/stride/random/stencil/atomic) and
extended_kernels.cu (saxpy/reduction/histogram/transpose/gemv) with the UVM
tracking pass, runs each kernel, and verifies the generated pagelog matches the
stored baseline by allocation fingerprint (total pages + contiguous cluster layout).

Run from the project root via run_tests.py or directly:
    python3 tests/microbenchmarks/test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import (
    compile, run_kernel, fingerprint_match, skip_if_missing,
    PASS_PATH, LIB_PATH, CTL_PATH, ACCESS_LOG,
)

TEST_DIR  = "tests/microbenchmarks"
BASELINES = f"{TEST_DIR}/baselines"

# (display_name, source_file, kernel_id, baseline_stem)
KERNELS = [
    ("Coalesced",  f"{TEST_DIR}/basic_kernels.cu",    0, "coalesced"),
    ("Stride",     f"{TEST_DIR}/basic_kernels.cu",    1, "stride"),
    ("Random",     f"{TEST_DIR}/basic_kernels.cu",    2, "random"),
    ("Stencil",    f"{TEST_DIR}/basic_kernels.cu",    3, "stencil"),
    ("Atomic",     f"{TEST_DIR}/basic_kernels.cu",    4, "atomic"),
    ("SAXPY",      f"{TEST_DIR}/extended_kernels.cu", 0, "saxpy"),
    ("Reduction",  f"{TEST_DIR}/extended_kernels.cu", 1, "reduction"),
    ("Histogram",  f"{TEST_DIR}/extended_kernels.cu", 2, "histogram"),
    ("Transpose",  f"{TEST_DIR}/extended_kernels.cu", 3, "transpose"),
    ("GEMV",       f"{TEST_DIR}/extended_kernels.cu", 4, "gemv"),
]


def main():
    # ── Preflight ──────────────────────────────────────────────────────────────
    skip_if_missing(PASS_PATH, LIB_PATH, CTL_PATH,
                    message="build artifacts missing — run build_and_time.sh first")

    # ── Compile ────────────────────────────────────────────────────────────────
    os.makedirs("build", exist_ok=True)
    sources  = sorted({src for _, src, _, _ in KERNELS})
    binaries = {}

    print("Compiling:")
    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]
        out  = f"build/microbench_{stem}_instr"
        print(f"  {stem:<28}", end="", flush=True)
        ok, stderr = compile(src, out)
        if not ok:
            print("FAILED")
            print(stderr[-800:])
            print("FAIL")
            sys.exit(1)
        print("ok" if ok else "up-to-date")
        binaries[src] = out

    # ── Run & compare ──────────────────────────────────────────────────────────
    failures = []
    print("\nKernel fingerprint checks:")
    for name, src, kid, baseline_stem in KERNELS:
        binary        = binaries[src]
        baseline_path = f"{BASELINES}/{baseline_stem}.pagelog"
        label         = f"  {name:<14}"

        if not os.path.exists(baseline_path):
            print(f"{label}SKIP  (no baseline)")
            continue

        log_path, stderr = run_kernel(binary, binary_flags=[str(kid)])
        if log_path is None:
            print(f"{label}FAIL  (run error: {stderr[:120]})")
            failures.append(f"{name}: run error")
            continue

        ok, detail = fingerprint_match(log_path, baseline_path)
        os.remove(log_path)
        print(f"{label}{'PASS' if ok else 'FAIL'}  ({detail})")
        if not ok:
            failures.append(f"{name}: {detail}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_total = len(KERNELS)
    n_fail  = len(failures)
    n_pass  = n_total - n_fail
    print(f"\n{n_pass}/{n_total} kernels matched baseline.")

    if failures:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
