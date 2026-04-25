import os
import subprocess
import socket
import struct
import time

# --- Configuration ---
BUILD_SCRIPT = "./build_and_time.sh"
SOURCE_FILE = "examples/thread_mode_tests.cu"
PRELOAD_LIB = "./libMallocIntercept.so"
LOG_FILE = "access_log.bin"

MODE_MAP = {
    0: "no-preload",
    1: "preload-alloc",
    2: "preload-only"
}

def compile_app(mode_int):
    mode_str = MODE_MAP.get(mode_int, "no-preload")
    print(f"\n[*] Compiling {SOURCE_FILE} via build script in mode: {mode_str}...")
    
    cmd = [BUILD_SCRIPT, SOURCE_FILE, "--mode", mode_str]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Compilation failed: {e}")
        exit(1)

def send_socket_command(pid, command):
    sock_path = f"/tmp/uvm-ctl.{pid}"
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(sock_path)
        client.sendall((command + "\n").encode())
        resp = client.recv(1024).decode().strip()
        client.close()
        return resp
    except Exception as e:
        print(f"Socket error: {e}")
        return None

_PAGELOG_MAGIC   = 0x50474C47
_HEADER_FMT      = "<IHHHHIIIQ"
_HEADER_SIZE     = struct.calcsize(_HEADER_FMT)
_INDEX_FMT       = "<HHQ"
_INDEX_SIZE      = struct.calcsize(_INDEX_FMT)

def count_unique_pages(filepath):
    """Count unique 4 KB pages recorded in a structured pagelog bitmap."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file {filepath} not found.")

    with open(filepath, "rb") as f:
        raw = f.read()

    hdr = struct.unpack_from(_HEADER_FMT, raw)
    if hdr[0] != _PAGELOG_MAGIC:
        raise ValueError(f"Bad pagelog magic 0x{hdr[0]:08X} in {filepath}")

    l3_bytes, num_leaves = hdr[4], hdr[8]
    count = 0
    for k in range(num_leaves):
        _, _, data_off = struct.unpack_from(_INDEX_FMT, raw, _HEADER_SIZE + k * _INDEX_SIZE)
        for w in range(l3_bytes // 8):
            word = struct.unpack_from("<Q", raw, data_off + w * 8)[0]
            if word:
                count += bin(word).count("1")
    return count

def run_test(name, mode_str, env_vars, socket_cmds=None, expected_pages=0):
    print(f"\n--- Testing: {name} ---")
    
    # Construct the dynamic executable name based on the mode string
    executable = f"./build/thread_mode_tests_instrumented_{mode_str}"
    
    # Clean up old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    env = os.environ.copy()
    env.update(env_vars)

    if not os.path.exists(executable):
        print(f"[!] Executable {executable} not found. Did compilation fail?")
        return

    proc = subprocess.Popen(
        [executable],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    # Wait for the CUDA app to finish allocating and spawn the socket
    ready_signal = proc.stdout.readline().strip()
    if ready_signal != "READY":
        print(f"Failed to initialize tracking or program crashed early. (Read: '{ready_signal}')")
        proc.kill()
        return

    # Fire optional live commands
    if socket_cmds:
        for cmd in socket_cmds:
            print(f"    > Sending socket cmd: {cmd}")
            send_socket_command(proc.pid, cmd)

    # Resume the CUDA program to run the kernel
    proc.stdin.write("GO\n")
    proc.stdin.flush()
    
    proc.wait()

    # Verify counts
    tracked = count_unique_pages(LOG_FILE)
    status = "PASS" if tracked == expected_pages else "FAIL"
    print(f"    [{status}] Expected: {expected_pages} | Tracked: {tracked}")

if __name__ == "__main__":
    
    # Ensure the bash script is executable
    if os.path.exists(BUILD_SCRIPT) and not os.access(BUILD_SCRIPT, os.X_OK):
        os.chmod(BUILD_SCRIPT, 0o755)

    # ---------------------------------------------------------
    # Test 1: No Preload (Mode 0)
    # ---------------------------------------------------------
    mode_int = 0
    mode_str = MODE_MAP[mode_int]
    compile_app(mode_int)
    run_test(
        name="No Preload (Mode 0)",
        mode_str=mode_str,
        env_vars={}, 
        expected_pages=2000
    )

    # ---------------------------------------------------------
    # Test 2: Preload Only (Mode 2)
    # ---------------------------------------------------------
    mode_int = 2
    mode_str = MODE_MAP[mode_int]
    compile_app(mode_int)
    run_test(
        name="Preload Only - Both Enabled",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "1", "UVM_PRELOAD_DEVICE": "1"},
        expected_pages=2000
    )
    run_test(
        name="Preload Only - Managed Only",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "1", "UVM_PRELOAD_DEVICE": "0"},
        expected_pages=1000
    )
    run_test(
        name="Preload Only - Device Only",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "1"},
        expected_pages=1000
    )
    run_test(
        name="Preload Only - None Enabled (Defaults to Skip)",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        expected_pages=0
    )

    # ---------------------------------------------------------
    # Test 3: Preload Alloc (Mode 1)
    # ---------------------------------------------------------
    mode_int = 1
    mode_str = MODE_MAP[mode_int]
    compile_app(mode_int)
    run_test(
        name="Preload Alloc - Normal Mode (Alloc on Miss)",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        socket_cmds=["MODE ALLOC"], 
        expected_pages=2000
    )
    run_test(
        name="Preload Alloc - Skip Mode (Ignore on Miss)",
        mode_str=mode_str,
        env_vars={"LD_PRELOAD": PRELOAD_LIB, "UVM_PRELOAD_MANAGED": "0", "UVM_PRELOAD_DEVICE": "0"},
        socket_cmds=["MODE SKIP"], 
        expected_pages=0
    )