import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

SOURCE_FILE = "tests/thread_modes/thread_mode_tests.cu"

MODE_MAP = {
    0: "no-preload",
    1: "preload-alloc",
    2: "preload-only"
}

def compile_app(mode_int):
    mode_str = MODE_MAP.get(mode_int, "no-preload")
    print(f"\n[*] Compiling {SOURCE_FILE} via harness in mode: {mode_str}...")
    
    executable = f"./build/thread_mode_tests_instrumented_{mode_str}"
    ok, stderr = harness.compile(SOURCE_FILE, executable, instrumented=True, mode=mode_int, rdynamic=True)
    if not ok:
        print(f"[!] Compilation failed: {stderr}")
        sys.exit(1)
    return executable

def run_test(name, executable, env_vars, socket_cmds=None, expected_pages=0):
    print(f"\n--- Testing: {name} ---")
    
    # Parse env_vars to pass to harness.run_with_handshake
    with_preload = env_vars.get("LD_PRELOAD") is not None
    
    preload_managed_str = env_vars.get("UVM_PRELOAD_MANAGED")
    preload_managed = None if preload_managed_str is None else (preload_managed_str == "1")

    preload_device_str = env_vars.get("UVM_PRELOAD_DEVICE")
    preload_device = None if preload_device_str is None else (preload_device_str == "1")

    extra_env = {k: v for k, v in env_vars.items() if k not in ["LD_PRELOAD", "UVM_PRELOAD_MANAGED", "UVM_PRELOAD_DEVICE"]}

    pagelog, responses, err = harness.run_with_handshake(
        executable,
        socket_cmds=socket_cmds,
        with_preload=with_preload,
        preload_managed=preload_managed,
        preload_device=preload_device,
        extra_env=extra_env,
        ready_timeout=10,
        run_timeout=20
    )

    if err and "timed out" not in err and "READY" not in err and "no access_log" not in err:
        pass # harness captures stderr

    if not pagelog:
        print(f"Failed to run properly: {err}")
        return

    if socket_cmds:
        for cmd in socket_cmds:
            print(f"    > Socket cmd '{cmd}' response: {responses.get(cmd)}")

    tracked = harness.count_pages(pagelog)
    status = "PASS" if tracked == expected_pages else "FAIL"
    print(f"    [{status}] Expected: {expected_pages} | Tracked: {tracked}")


if __name__ == "__main__":
    
    print("── Building wrapper ──")
    harness.subprocess.run([
        harness.CLANG, "-shared", "-fPIC", "-O2", f"-I{os.path.join(harness.ROOT, 'include')}",
        "libMallocIntercept.cpp", "-o", harness.PRELOAD_SO, "-ldl"
    ], check=True, cwd=harness.ROOT)

    # ---------------------------------------------------------
    # Test 1: No Preload (Mode 0)
    # ---------------------------------------------------------
    mode_int = 0
    mode_str = MODE_MAP[mode_int]
    exe = compile_app(mode_int)
    run_test(
        name="No Preload (Mode 0)",
        executable=exe,
        env_vars={}, 
        expected_pages=2000
    )

    # ---------------------------------------------------------
    # Test 2: Preload Only (Mode 2)
    # ---------------------------------------------------------
    mode_int = 2
    mode_str = MODE_MAP[mode_int]
    exe = compile_app(mode_int)
    run_test(
        name="Preload Only - Both Enabled",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "1", "UVM_PRELOAD_DEVICE": "1"},
        expected_pages=2000
    )
    run_test(
        name="Preload Only - Managed Only",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "1", "UVM_PRELOAD_DEVICE": "0"},
        expected_pages=1000
    )
    run_test(
        name="Preload Only - Device Only",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "1"},
        expected_pages=1000
    )
    run_test(
        name="Preload Only - None Enabled (Defaults to Skip)",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        expected_pages=0
    )

    # ---------------------------------------------------------
    # Test 3: Preload Alloc (Mode 1)
    # ---------------------------------------------------------
    mode_int = 1
    mode_str = MODE_MAP[mode_int]
    exe = compile_app(mode_int)
    run_test(
        name="Preload Alloc - Normal Mode (Alloc on Miss)",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        socket_cmds=["MODE ALLOC"], 
        expected_pages=2000
    )
    run_test(
        name="Preload Alloc - Skip Mode (Ignore on Miss)",
        executable=exe,
        env_vars={"LD_PRELOAD": "libMallocIntercept.so", "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        socket_cmds=["MODE SKIP"], 
        expected_pages=0
    )