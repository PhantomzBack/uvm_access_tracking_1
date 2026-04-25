#!/usr/bin/env python3

import subprocess
import sys
import json
import datetime
import shutil
from pathlib import Path

# ================= CONFIG =================

# Files that define "compiler pass changes"
TARGET_FILES = [
    "UvmTrackingPass.cpp",
    "libMarkAccess.cu",
    "libMallocIntercept.cpp",
    "common.h",
    "include/tracking.h",
    "uvm_control_thread.cu",
    "uvm_control_thread.h",
]

# Expanded paths to ensure dependency consistency
TARGET_PATHS = [
    "UvmTrackingPass.cpp",
    "libMarkAccess.cu",
    "libMallocIntercept.cpp",
    "common.h",
    "include/tracking.h",
    "uvm_control_thread.cu",
    "uvm_control_thread.h",
]

RESULTS_DIR = Path("perf_results")


# ================= UTIL =================

def run(cmd, capture=False):
    if capture:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    else:
        subprocess.run(cmd, shell=True, check=True)


def safe_run(cmd):
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


# ================= GIT =================

def get_current_branch():
    return run("git rev-parse --abbrev-ref HEAD", capture=True)


def get_commits_last_n(n):
    cmd = f"git log -n {n} --pretty=format:%H"
    commits = run(cmd, capture=True).splitlines()
    return commits[::-1]


def get_commits_range(rng):
    cmd = f"git log {rng} --pretty=format:%H"
    commits = run(cmd, capture=True).splitlines()
    return commits[::-1]


def get_commits_touching_files():
    files = " ".join(TARGET_FILES)
    cmd = f"git log --pretty=format:%H -- {files}"
    commits = run(cmd, capture=True).splitlines()
    return list(dict.fromkeys(commits[::-1]))


def get_commit_metadata(commit):
    return {
        "commit": commit,
        "message": run(f"git log -1 --pretty=%B {commit}", capture=True).strip(),
        "author": run(f"git log -1 --pretty=%an {commit}", capture=True),
        "date": run(f"git log -1 --pretty=%ad {commit}", capture=True),
    }


# ================= CHECKOUT =================

def checkout_partial(commit):
    valid_paths = []

    for path in TARGET_PATHS:
        # check if file exists in that commit
        cmd = f"git ls-tree -r --name-only {commit} | grep -w {path}"
        try:
            out = subprocess.check_output(cmd, shell=True).decode().strip()
            if out:
                valid_paths.append(path)
        except subprocess.CalledProcessError:
            continue

    if not valid_paths:
        print("❌ no valid paths found in this commit")
        return False

    paths_str = " ".join(valid_paths)
    return safe_run(f"git checkout {commit} -- {paths_str}")

def restore_files(branch):
    paths = " ".join(TARGET_PATHS)
    run(f"git checkout {branch} -- {paths}")


# ================= BUILD =================

def rebuild():
    print("🔨 Rebuilding (cache reset)...")

    cmd = """
    cd build && \
    rm -f CMakeCache.txt && \
    rm -rf CMakeFiles && \
    cmake .. -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm && \
    make -j
    """

    return safe_run(cmd)      



# ================= BENCH =================

def run_benchmark(output_dir):
    print("📊 Running benchmark...")

    log_file = output_dir / "benchmark.log"

    proc = subprocess.Popen(
        ["python3", "benchmark.py", "--preload", "--rebuild", "--mode", "preload-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    with open(log_file, "w") as f:
        for line in proc.stdout:
            print(line, end="")   # live output
            f.write(line)

    proc.wait()

    if not Path("results.md").exists():
        print("⚠️ results.md missing")
        return False

    run(f"cp results.md {output_dir / 'results.md'}")
    return True


# ================= CORE =================

def benchmark_commit(commit, original_branch):
    print(f"\n===== {commit} =====")

    output_dir = RESULTS_DIR / commit

    # clean overwrite
    if output_dir.exists():
        print(f"♻️ Removing old results for {commit}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    metadata = get_commit_metadata(commit)

    # checkout only relevant files
    if not checkout_partial(commit):
        print("❌ partial checkout failed")
        return False

    # rebuild
    if not rebuild():
        print("❌ build failed")
        restore_files(original_branch)
        return False

    # run benchmark
    if not run_benchmark(output_dir):
        print("❌ benchmark failed")

    # save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            **metadata,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2)

    # restore original files
    restore_files(original_branch)

    return True


# ================= INDEX =================

def generate_index():
    index_file = RESULTS_DIR / "INDEX.md"

    with open(index_file, "w") as f:
        f.write("# Benchmark History\n\n")
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir():
                f.write(f"- {d.name}\n")


# ================= MAIN =================

def main():
    print("🚀 Script started")
    RESULTS_DIR.mkdir(exist_ok=True)

    original_branch = get_current_branch()
    print(f"📍 Starting from branch: {original_branch}")

    # ---- ARG PARSING ----
    if "--last" in sys.argv:
        n = int(sys.argv[sys.argv.index("--last") + 1])
        commits = get_commits_last_n(n)

    elif "--range" in sys.argv:
        rng = sys.argv[sys.argv.index("--range") + 1]
        commits = get_commits_range(rng)

    elif "--commits" in sys.argv:
        idx = sys.argv.index("--commits") + 1
        commits = sys.argv[idx:]

    elif "--auto" in sys.argv:
        commits = get_commits_touching_files()

    else:
        print("Usage:")
        print("  --last N")
        print("  --range A..B")
        print("  --commits <list>")
        print("  --auto")
        sys.exit(1)

    print(f"🔍 {len(commits)} commits selected")

    success = 0
    failed = []

    for c in commits:
        try:
            if benchmark_commit(c, original_branch):
                success += 1
            else:
                failed.append(c)
        except Exception as e:
            print(f"💥 error on {c}: {e}")
            failed.append(c)
            restore_files(original_branch)

    generate_index()

    print("\n" + "=" * 80)
    print(f"✅ Successful: {success}/{len(commits)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()