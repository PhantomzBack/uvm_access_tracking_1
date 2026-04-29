#!/usr/bin/env python3
"""
build_and_time.py — Modern build and timing harness using shared compilation utilities.

Replaces build_and_time.sh with a Pythonic interface supporting:
  - Multiple compilation modes (no-preload, preload-alloc, preload-only)
  - Batch compilation across multiple modes in a single invocation
  - Both instrumented and normal binaries
  - Automatic timing and overhead calculation
  - Convenient flags for quick iteration
  - Control thread configuration
  - Additional compiler flags
  - Dry run mode for command inspection

Usage
-----
    build_and_time.py <source_file.cu> [options]

Options
-------
    --mode MODE
        Tracking mode (default: no-preload). Can be:
        - Single mode: "no-preload", "preload-alloc", "preload-only", "0", "1", "2"
        - Multiple modes: "{no-preload,preload-alloc}", "{0,1,2}", "{0,preload-only}"
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
    --no-control-thread
        Disable control thread (adds -DUVM_NO_CONTROL_THREAD)
    --extra-flags FLAGS
        Additional compiler flags (space-separated)
    --dry-run
        Print compilation commands without executing

Examples
--------
    # Compile in no-preload mode (default):
    ./build_and_time.py examples/benchmark_kernel.cu

    # Compile in preload-alloc mode:
    ./build_and_time.py examples/benchmark_kernel.cu --mode preload-alloc

    # Compile in preload-alloc mode using index:
    ./build_and_time.py examples/benchmark_kernel.cu --mode 1

    # Compile all three modes:
    ./build_and_time.py examples/benchmark_kernel.cu --mode "{0,1,2}"
    
    # Compile multiple modes with names:
    ./build_and_time.py examples/benchmark_kernel.cu --mode "{no-preload,preload-only}"

    # Compile and run benchmarks in preload-alloc mode:
    ./build_and_time.py examples/benchmark_kernel.cu --mode preload-alloc --run

    # Dry run to inspect all compilation commands for modes 0 and 2:
    ./build_and_time.py examples/benchmark_kernel.cu --mode "{0,2}" --dry-run
    
    # Disable control thread:
    ./build_and_time.py examples/benchmark_kernel.cu --no-control-thread
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
        type=str,
        default="no-preload",
        metavar="MODE",
        help="Tracking mode (default: no-preload). Can be a single mode (no-preload, preload-alloc, preload-only)"
             " or index (0, 1, 2), or multiple modes in braces (e.g., {no-preload,preload-alloc} or {0,2})"
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
    
    parser.add_argument(
        "--no-control-thread",
        action="store_true",
        help="Disable control thread (adds -DUVM_NO_CONTROL_THREAD)"
    )
    
    parser.add_argument(
        "--extra-flags",
        type=str,
        default="",
        metavar="FLAGS",
        help="Additional compiler flags (space-separated)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print compilation commands without executing"
    )
    
    args = parser.parse_args()
    
    # Validate that we don't skip both
    if args.normal_only and args.instrumented_only:
        parser.error("Cannot use both --normal-only and --instrumented-only")
    
    return args


def mode_to_tracking_mode(mode_str):
    """Convert mode string or index to harness tracking mode integer."""
    mapping = {
        "no-preload": 0,
        "preload-alloc": 1,
        "preload-only": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }
    if mode_str not in mapping:
        raise ValueError(f"Unknown mode: {mode_str}")
    return mapping[mode_str]


def parse_modes(mode_str):
    """
    Parse mode argument which can be:
    - Single mode: "no-preload" or "0"
    - Multiple modes in braces: "{no-preload,preload-alloc}" or "{0,1,2}"
    
    Returns a list of mode strings (e.g., ["no-preload", "preload-alloc"])
    """
    mode_str = mode_str.strip()
    
    # Check if multiple modes are specified in braces
    if mode_str.startswith("{") and mode_str.endswith("}"):
        modes_raw = mode_str[1:-1].split(",")
        modes = [m.strip() for m in modes_raw]
    else:
        modes = [mode_str]
    
    # Validate and expand indices to names
    mode_names = []
    index_to_name = {
        "0": "no-preload",
        "1": "preload-alloc",
        "2": "preload-only",
    }
    
    for mode in modes:
        if mode in index_to_name:
            # Convert index to name
            mode_names.append(index_to_name[mode])
        elif mode in ["no-preload", "preload-alloc", "preload-only"]:
            mode_names.append(mode)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: no-preload, preload-alloc, preload-only, 0, 1, 2")
    
    return mode_names


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


def compile_binaries(source, normal_exe, instrumented_exe, mode, force, skip_normal, skip_instrumented, 
                     control_thread=True, extra_flags=None, dry_run=False):
    """Compile normal and instrumented versions."""
    tracking_mode = mode_to_tracking_mode(mode)
    use_preload = mode != "no-preload"
    extra_flags_list = extra_flags.split() if extra_flags else None
    
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
            control_thread=control_thread,
            extra_flags=extra_flags_list,
            force=force,
            dry_run=dry_run
        )
        elapsed = time.time() - start
        
        if ok:
            if dry_run:
                print(f"  (dry run)")
            else:
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
            extra_flags=extra_flags_list,
            force=force,
            dry_run=dry_run
        )
        elapsed = time.time() - start
        
        if ok:
            if dry_run:
                print(f"  (dry run)")
            else:
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
    
    # Parse modes (single or multiple)
    try:
        modes = parse_modes(args.mode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    print(f"Source: {source}")
    if len(modes) == 1:
        print(f"Mode:   {modes[0]}")
    else:
        print(f"Modes:  {', '.join(modes)}")
    
    # Compile
    skip_normal = args.instrumented_only
    skip_instrumented = args.normal_only
    
    all_success = True
    
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Compiling for mode: {mode}")
        print(f"{'=' * 60}")
        
        # Get output paths for this mode
        normal_exe, instrumented_exe = get_output_paths(args.source, mode)
        
        results = compile_binaries(
            str(source),
            normal_exe,
            instrumented_exe,
            mode,
            args.force,
            skip_normal,
            skip_instrumented,
            control_thread=not args.no_control_thread,
            extra_flags=args.extra_flags,
            dry_run=args.dry_run
        )
        
        # Check if compilation succeeded
        compile_ok = all(success for _, success in results.values())
        if not compile_ok:
            print(f"\nCompilation failed for mode: {mode}", file=sys.stderr)
            all_success = False
            continue
        
        # Run benchmarks if requested
        if args.run:
            times = run_benchmarks(results, mode, args.timeout)
            if times is None:
                all_success = False
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
