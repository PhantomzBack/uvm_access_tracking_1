#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"
ROOT_DIR  = Path(__file__).parent


def discover():
    return sorted(d for d in TESTS_DIR.iterdir()
                  if d.is_dir() and (d / "test.py").exists())


def run_all(verbose=False, name_filter=None):
    test_dirs = discover()

    if name_filter:
        test_dirs = [d for d in test_dirs if name_filter.lower() in d.name.lower()]

    if not test_dirs:
        print("No tests found.")
        return True

    print(f"Running {len(test_dirs)} test(s) from tests/\n")

    results = []
    for test_dir in test_dirs:
        name = test_dir.name
        print(f"  {name:<34}", end="", flush=True)

        proc = subprocess.run(
            [sys.executable, str(test_dir / "test.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0
        print("PASS" if passed else "FAIL")

        show_output = not passed or verbose
        if show_output:
            for line in (proc.stdout + proc.stderr).splitlines():
                print(f"    {line}")

        results.append((name, passed))

    n_pass = sum(1 for _, p in results if p)
    n_fail = len(results) - n_pass

    print(f"\n{'─' * 38}")
    if n_fail == 0:
        print(f"All {n_pass} test(s) passed.")
    else:
        print(f"{n_pass}/{len(results)} passed — {n_fail} FAILED:")
        for name, passed in results:
            if not passed:
                print(f"  • {name}")

    return n_fail == 0


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose     = "-v" in args or "--verbose" in args
    name_filter = next((a for a in args if not a.startswith("-")), None)
    sys.exit(0 if run_all(verbose=verbose, name_filter=name_filter) else 1)
