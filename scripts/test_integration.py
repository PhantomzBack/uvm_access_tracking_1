#!/usr/bin/env python3
"""
Phase 5 — Full integration test across all 3 modes with control thread + LD_PRELOAD.

For each mode (0=no-preload, 1=preload-alloc, 2=preload-only):
  1. Builds long_running_test with the mode flag.
  2. Launches it with LD_PRELOAD.
  3. Sends socket commands: STATUS, DISABLE, SNAPSHOT, ENABLE, SNAPSHOT,
     CLEAR, SNAPSHOT, PRELOAD_MANAGED toggle, SHUTDOWN.
  4. Verifies page counts and toggle states.

Usage:
    python3 scripts/test_integration.py
"""

import subprocess
import os
import sys
import socket
import time
import struct
import tempfile
import glob

CUDA_PATH    = "/usr/local/cuda"
CLANG        = "clang++-20"
PASS_PATH    = "./build/UvmTrackingPass.so"
PRELOAD_SO   = "./libMallocIntercept.so"
TEST_BIN     = "./build/long_running_test"

SM_ARCH = subprocess.getoutput(
    "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
)


def compile_test(mode):
    print(f"\n{'='*60}")
    print(f"Building long_running_test for mode {mode}")
    print(f"{'='*60}")
    cmd = [
        CLANG, "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}",
        "-fgpu-rdc", "-O2", "-I./include", "-rdynamic",
        "-DTRACKING_ENABLED", f"-fpass-plugin={PASS_PATH}",
    ]
    if mode != 0:
        cmd.append(f"-DUVM_TRACKING_MODE={mode}")
    cmd += [
        "examples/long_running_test.cu",
        "libMarkAccess.cu", "uvm_control_thread.cu",
        f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
        "-o", TEST_BIN,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("Build FAILED")
        print(r.stderr[-2000:])
        return False
    print("Build OK")
    return True


def send_cmd(sock_path, cmd, timeout=2.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((cmd + "\n").encode())
    resp = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
    except socket.timeout:
        pass
    s.close()
    return resp.decode()


def count_pages(path):
    if not os.path.exists(path):
        return -1
    with open(path, "rb") as f:
        raw = f.read()
    fmt = "<IHHHHIIIQ"
    hdr_sz = struct.calcsize(fmt)
    if len(raw) < hdr_sz:
        return -1
    magic, _, _, _, l3b, _, _, _, num_leaves = struct.unpack_from(fmt, raw)
    if magic != 0x50474C47:
        return -1
    idx_fmt = "<HHQ"
    idx_sz = struct.calcsize(idx_fmt)
    pages = 0
    for k in range(num_leaves):
        l1, l2, off = struct.unpack_from(idx_fmt, raw, hdr_sz + k * idx_sz)
        for w in range(l3b // 8):
            word = struct.unpack_from("<Q", raw, off + w * 8)[0]
            pages += bin(word).count("1")
    return pages


def run_mode_test(mode):
    if not compile_test(mode):
        return False

    # Rebuild wrapper to be safe
    print("\n── Building wrapper ──")
    subprocess.run([
        CLANG, "-shared", "-fPIC", "-O2", "-I./include",
        "libMallocIntercept.cpp", "-o", PRELOAD_SO, "-ldl"
    ], check=True)
    print("Wrapper OK")

    print(f"\n── Starting long_running_test mode={mode} (8 s) ──")
    env = os.environ.copy()
    env["LD_PRELOAD"] = os.path.abspath(PRELOAD_SO)

    proc = subprocess.Popen(
        [TEST_BIN, "8"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    time.sleep(0.5)

    # Find socket
    pid = proc.pid
    sock = f"/tmp/uvm-ctl.{pid}"
    if not os.path.exists(sock):
        socks = glob.glob("/tmp/uvm-ctl.*")
        if socks:
            sock = socks[0]
        else:
            print("FAIL: no control socket found")
            proc.kill()
            return False

    print(f"Connected to {sock}")

    # Collect temp files for cleanup
    temps = []
    try:
        # ── Test 1: STATUS ──
        print("\n── Test 1: STATUS ──")
        resp = send_cmd(sock, "STATUS")
        print(resp)
        assert f"mode: {mode}" in resp, f"STATUS missing expected mode {mode}"
        print("PASS")

        # ── Test 2: DISABLE → work 1 s → SNAPSHOT A ──
        print("\n── Test 2: DISABLE → work 1 s → SNAPSHOT A ──")
        print(send_cmd(sock, "DISABLE"), end="")
        time.sleep(1.2)
        snap_a = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_a)
        print(send_cmd(sock, f"SNAPSHOT {snap_a}"), end="")
        pages_a = count_pages(snap_a)
        print(f"Snapshot A pages: {pages_a}")
        print("PASS (data collected)")

        # ── Test 3: ENABLE → work 2 s → SNAPSHOT B ──
        print("\n── Test 3: ENABLE → work 2 s → SNAPSHOT B ──")
        print(send_cmd(sock, "ENABLE"), end="")
        time.sleep(2.2)
        snap_b = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_b)
        print(send_cmd(sock, f"SNAPSHOT {snap_b}"), end="")
        pages_b = count_pages(snap_b)
        print(f"Snapshot B pages: {pages_b}")
        assert pages_b > 0, "Snapshot B should have pages (tracking was ON)"
        print("PASS")

        # ── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──
        print("\n── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──")
        print(send_cmd(sock, "DISABLE"), end="")
        time.sleep(0.5)
        print(send_cmd(sock, "CLEAR"), end="")
        time.sleep(1.5)
        snap_c = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_c)
        print(send_cmd(sock, f"SNAPSHOT {snap_c}"), end="")
        pages_c = count_pages(snap_c)
        print(f"Snapshot C pages: {pages_c}")
        assert pages_c == 0, f"Snapshot C should be empty after DISABLE+CLEAR, got {pages_c}"
        print("PASS")

        # ── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──
        print("\n── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──")
        print(send_cmd(sock, "ENABLE"), end="")
        time.sleep(2.2)
        snap_d = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_d)
        print(send_cmd(sock, f"SNAPSHOT {snap_d}"), end="")
        pages_d = count_pages(snap_d)
        print(f"Snapshot D pages: {pages_d}")
        assert pages_d > 0, "Snapshot D should have pages after re-enable"
        print("PASS")

        # ── Test 6: PRELOAD_MANAGED toggle ──
        print("\n── Test 6: PRELOAD_MANAGED toggle ──")
        print(send_cmd(sock, "PRELOAD_MANAGED 0"), end="")
        resp = send_cmd(sock, "STATUS")
        assert "preload_managed: 0" in resp, "PRELOAD_MANAGED 0 failed"
        print(send_cmd(sock, "PRELOAD_MANAGED 1"), end="")
        resp = send_cmd(sock, "STATUS")
        assert "preload_managed: 1" in resp, "PRELOAD_MANAGED 1 failed"
        print("PASS")

        # Shutdown
        print("\n── SHUTDOWN ──")
        print(send_cmd(sock, "SHUTDOWN"), end="")
        proc.wait(timeout=5)

    except AssertionError as e:
        print(f"\nFAIL: {e}")
        proc.kill()
        proc.wait(timeout=5)
        return False
    finally:
        for p in temps:
            if os.path.exists(p):
                os.remove(p)

    print(f"\n── Mode {mode}: ALL TESTS PASSED ──")
    return True


def main():
    results = {}
    for mode in (0, 1, 2):
        results[mode] = run_mode_test(mode)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    for mode, ok in results.items():
        label = {0: "no-preload", 1: "preload-alloc", 2: "preload-only"}[mode]
        print(f"  Mode {mode} ({label:12}): {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nAll modes passed.")
        sys.exit(0)
    else:
        print("\nSome modes failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
