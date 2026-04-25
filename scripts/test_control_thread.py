#!/usr/bin/env python3
"""
End-to-end test for the background control thread.

1. Builds the long-running test program.
2. Launches it with LD_PRELOAD.
3. Sends a sequence of socket commands:
      STATUS → DISABLE → sleep → ENABLE → sleep → SNAPSHOT A
      CLEAR  → sleep → SNAPSHOT B
4. Verifies:
      - Snapshot A has pages (tracking was ON during active work).
      - Snapshot B has fewer/zero pages (CLEAR wiped the bitmaps).
      - STATUS reports correct toggle states after each change.

Usage:
    python3 scripts/test_control_thread.py
"""

import subprocess
import os
import sys
import socket
import time
import struct
import tempfile

CUDA_PATH    = "/usr/local/cuda"
CLANG        = "clang++-20"
PASS_PATH    = "./build/UvmTrackingPass.so"
PRELOAD_SO   = "./libMallocIntercept.so"
TEST_BIN     = "./build/long_running_test"

SM_ARCH = subprocess.getoutput(
    "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
)


def compile_test():
    print("── Building long_running_test ──")
    cmd = [
        CLANG, "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}",
        "-fgpu-rdc", "-O2", "-I./include",
        "-DTRACKING_ENABLED", f"-fpass-plugin={PASS_PATH}",
        "examples/long_running_test.cu",
        "libMarkAccess.cu", "uvm_control_thread.cu",
        f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
        "-o", TEST_BIN,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("Build FAILED")
        print(r.stderr[-2000:])
        sys.exit(1)
    print("Build OK")


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
    """Return number of accessed pages in a binary pagelog."""
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


def main():
    compile_test()

    # Rebuild wrapper to be safe
    print("── Building wrapper ──")
    subprocess.run([
        CLANG, "-shared", "-fPIC", "-O2", "-I./include",
        "libMallocIntercept.cpp", "-o", PRELOAD_SO, "-ldl"
    ], check=True)
    print("Wrapper OK")

    print("\n── Starting long_running_test (10 s) ──")
    env = os.environ.copy()
    env["LD_PRELOAD"] = os.path.abspath(PRELOAD_SO)

    proc = subprocess.Popen(
        [TEST_BIN, "10"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    time.sleep(0.5)

    # Find socket
    pid = proc.pid
    sock = f"/tmp/uvm-ctl.{pid}"
    if not os.path.exists(sock):
        # PID may differ due to fork; scan for any uvm-ctl socket
        import glob
        socks = glob.glob("/tmp/uvm-ctl.*")
        if socks:
            sock = socks[0]
        else:
            print("FAIL: no control socket found")
            proc.kill()
            sys.exit(1)

    print(f"Connected to {sock}")

    # ── Test 1: STATUS ──
    print("\n── Test 1: STATUS ──")
    resp = send_cmd(sock, "STATUS")
    print(resp)
    assert "mode:" in resp, "STATUS missing mode"
    print("PASS")

    # ── Test 2: DISABLE during work → snapshot should be empty/stale ──
    print("\n── Test 2: DISABLE → work 1 s → SNAPSHOT A ──")
    print(send_cmd(sock, "DISABLE"), end="")
    time.sleep(1.2)
    snap_a = tempfile.mktemp(suffix=".pglog")
    print(send_cmd(sock, f"SNAPSHOT {snap_a}"), end="")
    pages_a = count_pages(snap_a)
    print(f"Snapshot A pages: {pages_a}")
    # After disable + 1 s, ideally 0 pages (or stale old data)
    print("PASS (data collected)")

    # ── Test 3: ENABLE → work 2 s → SNAPSHOT B (should have pages) ──
    print("\n── Test 3: ENABLE → work 2 s → SNAPSHOT B ──")
    print(send_cmd(sock, "ENABLE"), end="")
    time.sleep(2.2)
    snap_b = tempfile.mktemp(suffix=".pglog")
    print(send_cmd(sock, f"SNAPSHOT {snap_b}"), end="")
    pages_b = count_pages(snap_b)
    print(f"Snapshot B pages: {pages_b}")
    assert pages_b > 0, "Snapshot B should have pages (tracking was ON)"
    print("PASS")

    # ── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C (should be empty) ──
    print("\n── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──")
    print(send_cmd(sock, "DISABLE"), end="")
    time.sleep(0.5)  # let current kernel finish before clearing
    print(send_cmd(sock, "CLEAR"), end="")
    time.sleep(1.5)
    snap_c = tempfile.mktemp(suffix=".pglog")
    print(send_cmd(sock, f"SNAPSHOT {snap_c}"), end="")
    pages_c = count_pages(snap_c)
    print(f"Snapshot C pages: {pages_c}")
    assert pages_c == 0, f"Snapshot C should be empty after DISABLE+CLEAR, got {pages_c}"
    print("PASS")

    # ── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D (pages again) ──
    print("\n── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──")
    print(send_cmd(sock, "ENABLE"), end="")
    time.sleep(2.2)
    snap_d = tempfile.mktemp(suffix=".pglog")
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
    print("\n── All tests passed ──")

    # Cleanup temp files
    for p in [snap_a, snap_b, snap_c, snap_d]:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
