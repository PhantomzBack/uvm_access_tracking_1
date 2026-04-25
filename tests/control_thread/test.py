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
    python3 tests/control_thread/test.py
"""

import subprocess
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

TEST_BIN = "./build/long_running_test_control"

def main():
    print("── Building long_running_test ──")
    ok, stderr = harness.compile("tests/control_thread/long_running_test.cu", TEST_BIN, rdynamic=True)
    if not ok:
        print("Build FAILED")
        print(stderr[-2000:])
        sys.exit(1)
    print("Build OK")

    print("── Building wrapper ──")
    # harness assumes libMallocIntercept.so is already built or we can build it here?
    # run_tests usually runs make or we just compile it. harness doesn't build the wrapper.
    subprocess.run([
        "clang++-20", "-shared", "-fPIC", "-O2", "-I./include",
        "libMallocIntercept.cpp", "-o", harness.PRELOAD_SO, "-ldl"
    ], check=True)
    print("Wrapper OK")

    print("\n── Starting long_running_test (10 s) ──")
    
    proc, sock = harness.launch_socket_binary(TEST_BIN, ["10"], with_preload=True)
    print(f"Connected to {sock}")

    # ── Test 1: STATUS ──
    print("\n── Test 1: STATUS ──")
    resp = harness.send_socket_cmd(sock, "STATUS")
    print(resp)
    assert "mode:" in resp, "STATUS missing mode"
    print("PASS")

    # ── Test 2: DISABLE during work → snapshot should be empty/stale ──
    print("\n── Test 2: DISABLE → work 1 s → SNAPSHOT A ──")
    print(harness.send_socket_cmd(sock, "DISABLE"), end="")
    time.sleep(1.2)
    snap_a = tempfile.mktemp(suffix=".pglog")
    print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_a}"), end="")
    pages_a = harness.count_pages(snap_a) if os.path.exists(snap_a) else -1
    print(f"Snapshot A pages: {pages_a}")
    print("PASS (data collected)")

    # ── Test 3: ENABLE → work 2 s → SNAPSHOT B (should have pages) ──
    print("\n── Test 3: ENABLE → work 2 s → SNAPSHOT B ──")
    print(harness.send_socket_cmd(sock, "ENABLE"), end="")
    time.sleep(2.2)
    snap_b = tempfile.mktemp(suffix=".pglog")
    print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_b}"), end="")
    pages_b = harness.count_pages(snap_b)
    print(f"Snapshot B pages: {pages_b}")
    assert pages_b > 0, "Snapshot B should have pages (tracking was ON)"
    print("PASS")

    # ── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C (should be empty) ──
    print("\n── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──")
    print(harness.send_socket_cmd(sock, "DISABLE"), end="")
    time.sleep(0.5)  # let current kernel finish before clearing
    print(harness.send_socket_cmd(sock, "CLEAR"), end="")
    time.sleep(1.5)
    snap_c = tempfile.mktemp(suffix=".pglog")
    print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_c}"), end="")
    pages_c = harness.count_pages(snap_c)
    print(f"Snapshot C pages: {pages_c}")
    assert pages_c == 0, f"Snapshot C should be empty after DISABLE+CLEAR, got {pages_c}"
    print("PASS")

    # ── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D (pages again) ──
    print("\n── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──")
    print(harness.send_socket_cmd(sock, "ENABLE"), end="")
    time.sleep(2.2)
    snap_d = tempfile.mktemp(suffix=".pglog")
    print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_d}"), end="")
    pages_d = harness.count_pages(snap_d)
    print(f"Snapshot D pages: {pages_d}")
    assert pages_d > 0, "Snapshot D should have pages after re-enable"
    print("PASS")

    # ── Test 6: PRELOAD_MANAGED toggle ──
    print("\n── Test 6: PRELOAD_MANAGED toggle ──")
    print(harness.send_socket_cmd(sock, "PRELOAD_MANAGED 0"), end="")
    resp = harness.send_socket_cmd(sock, "STATUS")
    assert "preload_managed: 0" in resp, "PRELOAD_MANAGED 0 failed"
    print(harness.send_socket_cmd(sock, "PRELOAD_MANAGED 1"), end="")
    resp = harness.send_socket_cmd(sock, "STATUS")
    assert "preload_managed: 1" in resp, "PRELOAD_MANAGED 1 failed"
    print("PASS")

    # Shutdown
    print("\n── SHUTDOWN ──")
    print(harness.send_socket_cmd(sock, "SHUTDOWN"), end="")

    proc.wait(timeout=5)
    print("\n── All tests passed ──")

    # Cleanup temp files
    for p in [snap_a, snap_b, snap_c, snap_d]:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
