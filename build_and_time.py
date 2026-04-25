#!/usr/bin/env python3
"""
build_and_time.py — Modern build and timing harness using shared compilation utilities.

Replaces build_and_time.sh with a Pythonic interface supporting:
  - Multiple compilation modes (no-preload, preload-alloc, preload-only)
  - Both instrumented and normal binaries
  - Automatic timing and overhead calculation
  - Convenient flags for quick iteration

Usage
-----
    build_and_time.py <source_file.cu> [options]

Options
-------
    --mode {no-preload,preload-alloc,preload-only}
        Tracking mode (default: no-preload)
    --run
        Compile both versions and run benchmarks
    --normal-only
        Skip instrumented compilation
    --instrumented-only
        Skip normal compilation
    --force
        Rebuild even if outputs are fresh
    --timeout SECONDS
        Timeout for kernel runs (default: 60)

Examples
--------
    # Compile in no-preload mode:
    ./build_and_time.py examples/benchmark_kernel_single.cu

    # Compile and run benchmarks in preload-alloc mode:
    ./build_and_time.py examples/benchmark_kernel_single.cu --mode preload-alloc --run

    # Force rebuild and quick benchmark:
    ./build_and_time.py examples/benchmark_kernel_single.cu --force --run
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add tests dir to path for harness import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import harness


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile and optionally benchmark CUDA kernels with UVM tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "source",
        metavar="SOURCE",
        help="Path to CUDA source file (.cu)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["no-preload", "preload-alloc", "preload-only"],
        default="no-preload",
        help="Tracking mode (default: no-preload)"
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Compile both versions and run benchmarks"
    )
    
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="Skip instrumented compilation"
    )
    
    parser.add_argument(
        "--instrumented-only",
        action="store_true",
        help="Skip normal compilation"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if outputs are fresh"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        metavar="SECONDS",
        help="Timeout for kernel runs (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Validate that we don't skip both
    if args.normal_only and args.instrumented_only:
        parser.error("Cannot use both --normal-only and --instrumented-only")
    
    return args


def mode_to_tracking_mode(mode_str):
    """Convert mode string to harness tracking mode integer."""
    mapping = {
        "no-preload": 0,
        "preload-alloc": 1,
        "preload-only": 2,
    }
    return mapping[mode_str]


def get_output_paths(source, mode):
    """Generate output binary paths."""
    src_path = Path(source)
    stem = src_path.stem
    build_dir = Path(harness.ROOT) / "build"
    build_dir.mkdir(exist_ok=True)
    
    base_name = f"{stem}"
    if mode != "no-preload":
        instrumented_name = f"{stem}_instrumented_{mode}"
    else:
        instrumented_name = f"{stem}_instrumented"
    
    normal_exe = build_dir / f"{base_name}_normal"
    instrumented_exe = build_dir / instrumented_name
    
    return str(normal_exe), str(instrumented_exe)


def compile_binaries(source, normal_exe, instrumented_exe, mode, force, skip_normal, skip_instrumented):
    """Compile normal and instrumented versions."""
    tracking_mode = mode_to_tracking_mode(mode)
    use_preload = mode != "no-preload"
    
    results = {}
    
    if not skip_instrumented:
        print(f"[Compiling] {instrumented_exe} (instrumented, mode={mode})")
        start = time.time()
        ok, stderr = harness.compile(
            source,
            instrumented_exe,
            instrumented=True,
            mode=tracking_mode,
            rdynamic=use_preload,
            force=force
        )
        elapsed = time.time() - start
        
        if ok:
            print(f"  ✓ Built in {elapsed:.2f}s")
            results["instrumented"] = (instrumented_exe, True)
        else:
            print(f"  ✗ Failed")
            print(f"  Error: {stderr}")
            results["instrumented"] = (instrumented_exe, False)
    
    if not skip_normal:
        print(f"[Compiling] {normal_exe} (normal)")
        start = time.time()
        ok, stderr = harness.compile(
            source,
            normal_exe,
            instrumented=False,
            force=force
        )
        elapsed = time.time() - start
        
        if ok:
            print(f"  ✓ Built in {elapsed:.2f}s")
            results["normal"] = (normal_exe, True)
        else:
            print(f"  ✗ Failed")
            print(f"  Error: {stderr}")
            results["normal"] = (normal_exe, False)
    
    return results


def run_benchmarks(results, mode, timeout):
    """Run compiled binaries and measure execution time."""
    use_preload = mode != "no-preload"
    
    print("\n" + "=" * 60)
    print("Running Benchmarks")
    print("=" * 60)
    
    times = {}
    
    if "normal" in results:
        exe, success = results["normal"]
        if not success:
            print(f"[Skipping] Normal binary failed to compile")
            return None
        
        print(f"\n[Running] {exe}")
        start = time.time()
        logpath, stderr = harness.run_kernel(
            exe,
            env=harness._make_env(with_preload=use_preload),
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if logpath:
            print(f"  ✓ Completed in {elapsed:.4f}s")
            times["normal"] = elapsed
        else:
            print(f"  ✗ Failed: {stderr}")
            return None
    
    if "instrumented" in results:
        exe, success = results["instrumented"]
        if not success:
            print(f"[Skipping] Instrumented binary failed to compile")
            return None
        
        print(f"\n[Running] {exe}")
        start = time.time()
        logpath, stderr = harness.run_kernel(
            exe,
            env=harness._make_env(with_preload=use_preload),
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if logpath:
            print(f"  ✓ Completed in {elapsed:.4f}s")
            times["instrumented"] = elapsed
        else:
            print(f"  ✗ Failed: {stderr}")
            return None
    
    # Calculate and display results
    if len(times) == 2:
        normal_time = times["normal"]
        inst_time = times["instrumented"]
        ratio = inst_time / normal_time
        overhead_pct = (ratio - 1) * 100
        
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Normal:       {normal_time:>10.4f}s")
        print(f"Instrumented: {inst_time:>10.4f}s")
        print(f"Overhead:     {ratio:>10.4f}x ({overhead_pct:+.1f}%)")
        print("=" * 60)
    
    return times


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate source file
    source = Path(args.source)
    if not source.is_absolute():
        source = Path(harness.ROOT) / source
    
    if not source.exists():
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        return 1
    
    print(f"Source: {source}")
    print(f"Mode:   {args.mode}")
    
    # Get output paths
    normal_exe, instrumented_exe = get_output_paths(args.source, args.mode)
    
    # Compile
    skip_normal = args.instrumented_only
    skip_instrumented = args.normal_only
    
    results = compile_binaries(
        str(source),
        normal_exe,
        instrumented_exe,
        args.mode,
        args.force,
        skip_normal,
        skip_instrumented
    )
    
    # Check if compilation succeeded
    compile_ok = all(success for _, success in results.values())
    if not compile_ok:
        print("\nCompilation failed.", file=sys.stderr)
        return 1
    
    # Run benchmarks if requested
    if args.run:
        times = run_benchmarks(results, args.mode, args.timeout)
        if times is None:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
