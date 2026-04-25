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
    python3 tests/integration/test.py
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

TEST_BIN = "./build/long_running_test_integration"


def run_mode_test(mode):
    print(f"\n{'='*60}")
    print(f"Building long_running_test for mode {mode}")
    print(f"{'='*60}")

    ok, stderr = harness.compile(
        "tests/control_thread/long_running_test.cu", 
        TEST_BIN, 
        instrumented=True, 
        mode=mode, 
        rdynamic=True,
        force=True # Ensure it recompiles for the specific mode
    )
    if not ok:
        print("Build FAILED")
        print(stderr[-2000:])
        return False
    print("Build OK")

    # Rebuild wrapper to be safe
    print("\n── Building wrapper ──")
    harness.subprocess.run([
        harness.CLANG, "-shared", "-fPIC", "-O2", f"-I{os.path.join(harness.ROOT, 'include')}",
        "libMallocIntercept.cpp", "-o", harness.PRELOAD_SO, "-ldl"
    ], check=True, cwd=harness.ROOT)
    print("Wrapper OK")

    print(f"\n── Starting long_running_test mode={mode} (8 s) ──")
    
    try:
        proc, sock = harness.launch_socket_binary(TEST_BIN, ["8"], with_preload=True)
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    print(f"Connected to {sock}")

    # Collect temp files for cleanup
    temps = []
    try:
        # ── Test 1: STATUS ──
        print("\n── Test 1: STATUS ──")
        resp = harness.send_socket_cmd(sock, "STATUS")
        print(resp)
        assert f"mode: {mode}" in resp, f"STATUS missing expected mode {mode}"
        print("PASS")

        # ── Test 2: DISABLE → work 1 s → SNAPSHOT A ──
        print("\n── Test 2: DISABLE → work 1 s → SNAPSHOT A ──")
        print(harness.send_socket_cmd(sock, "DISABLE"), end="")
        time.sleep(1.2)
        snap_a = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_a)
        print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_a}"), end="")
        pages_a = harness.count_pages(snap_a) if os.path.exists(snap_a) else -1
        print(f"Snapshot A pages: {pages_a}")
        print("PASS (data collected)")

        # ── Test 3: ENABLE → work 2 s → SNAPSHOT B ──
        print("\n── Test 3: ENABLE → work 2 s → SNAPSHOT B ──")
        print(harness.send_socket_cmd(sock, "ENABLE"), end="")
        time.sleep(2.2)
        snap_b = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_b)
        print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_b}"), end="")
        pages_b = harness.count_pages(snap_b)
        print(f"Snapshot B pages: {pages_b}")
        assert pages_b > 0, "Snapshot B should have pages (tracking was ON)"
        print("PASS")

        # ── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──
        print("\n── Test 4: DISABLE → wait → CLEAR → work 1 s → SNAPSHOT C ──")
        print(harness.send_socket_cmd(sock, "DISABLE"), end="")
        time.sleep(0.5)
        print(harness.send_socket_cmd(sock, "CLEAR"), end="")
        time.sleep(1.5)
        snap_c = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_c)
        print(harness.send_socket_cmd(sock, f"SNAPSHOT {snap_c}"), end="")
        pages_c = harness.count_pages(snap_c)
        print(f"Snapshot C pages: {pages_c}")
        assert pages_c == 0, f"Snapshot C should be empty after DISABLE+CLEAR, got {pages_c}"
        print("PASS")

        # ── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──
        print("\n── Test 5: RE-ENABLE → work 2 s → SNAPSHOT D ──")
        print(harness.send_socket_cmd(sock, "ENABLE"), end="")
        time.sleep(2.2)
        snap_d = tempfile.mktemp(suffix=".pglog")
        temps.append(snap_d)
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
