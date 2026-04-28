#!/usr/bin/env python3
"""
tests/harness.py — shared test utilities for the UVM tracking pass test suite.

Public API
----------
Compilation
    compile(src, out, *, instrumented, mode, rdynamic, force)

One-shot kernel runs
    run_kernel(binary, binary_flags, *, env, timeout)
    run_kernel_ex(binary, binary_flags, *, with_preload, preload_managed,
                  preload_device, extra_env, timeout)

Long-running socket-controlled binaries  (long_running_test style)
    launch_socket_binary(binary, binary_flags, *, with_preload, preload_managed,
                         preload_device, extra_env, init_timeout)
    send_socket_cmd(proc_or_path, command, *, timeout)

READY / GO handshake binaries  (thread_mode_tests style)
    run_with_handshake(binary, binary_flags, *, socket_cmds, with_preload,
                       preload_managed, preload_device, extra_env,
                       ready_timeout, run_timeout)

Pagelog utilities
    parse_pagelog(path)       → set of page addresses
    count_pages(path)         → int
    fingerprint_match(a, b)   → (ok, detail)

Preflight
    skip_if_missing(*paths, message)   exits 0 (SKIP) if any path absent
"""

import glob
import os
import socket as _socket
import struct
import subprocess
import sys
import threading
import time

# ── Project root (this file lives at tests/harness.py) ────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Build artifact paths ───────────────────────────────────────────────────────
PASS_PATH  = os.path.join(ROOT, "build", "UvmTrackingPass.so")
LIB_PATH   = os.path.join(ROOT, "libMarkAccess.cu")
CTL_PATH   = os.path.join(ROOT, "uvm_control_thread.cu")
PRELOAD_SO = os.path.join(ROOT, "libMallocIntercept.so")
CUDA_PATH  = "/usr/local/cuda"
CLANG      = "clang++-20"

try:
    SM_ARCH = subprocess.getoutput(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
    ).strip()
    if not SM_ARCH or not SM_ARCH.isdigit():
        SM_ARCH = "61"
except Exception:
    SM_ARCH = "61"

BASE_FLAGS = [
    "-x", "cuda", f"--cuda-gpu-arch=sm_{SM_ARCH}",
    "-fgpu-rdc", "-O2", f"-I{os.path.join(ROOT, 'include')}",
    f"--cuda-path={CUDA_PATH}", f"-L{CUDA_PATH}/lib64", "-lcudart",
]

# Path where instrumented binaries write their pagelog
ACCESS_LOG = os.path.join(ROOT, "access_log.bin")

# ── Pagelog binary format ──────────────────────────────────────────────────────
_MAGIC      = 0x50474C47
_HEADER_FMT = "<IHHHHIIIQ"
_HEADER_SZ  = struct.calcsize(_HEADER_FMT)
_INDEX_FMT  = "<HHQ"
_INDEX_SZ   = struct.calcsize(_INDEX_FMT)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _stale(src, out):
    if not os.path.exists(out):
        return True
    out_t = os.path.getmtime(out)
    return max(
        os.path.getmtime(src)       if os.path.exists(src)       else 0,
        os.path.getmtime(PASS_PATH) if os.path.exists(PASS_PATH) else 0,
        os.path.getmtime(LIB_PATH)  if os.path.exists(LIB_PATH)  else 0,
    ) > out_t


def compile(src, out, *,
            instrumented=True,
            mode=0,
            rdynamic=False,
            control_thread=True,
            extra_flags=None,
            force=False,
            dry_run=False):
    """
    Compile a CUDA source file, skipping if outputs are up-to-date.

    Parameters
    ----------
    src            : path to .cu source (absolute, or relative to project root)
    out            : output binary path
    instrumented   : inject the UVM tracking pass (default True)
    mode           : tracking mode — 0 no-preload, 1 preload-alloc, 2 preload-only
                     (adds -DUVM_TRACKING_MODE=N for N > 0)
    rdynamic       : add -rdynamic (required for LD_PRELOAD modes)
    control_thread : include control thread (default True; False adds -DUVM_NO_CONTROL_THREAD)
    extra_flags    : list of additional compiler flags to append
    force          : rebuild even when outputs are fresh
    dry_run        : print command without executing

    Returns
    -------
    (ok: bool, stderr: str)
    """
    # Resolve relative paths against ROOT so the mtime check is consistent
    if not os.path.isabs(src):
        src = os.path.join(ROOT, src)
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)

    if not force and not _stale(src, out):
        return True, ""

    flags = list(BASE_FLAGS) + [src]
    if rdynamic:
        flags.append("-rdynamic")

    if instrumented:
        if mode > 0:
            flags.append(f"-DUVM_TRACKING_MODE={mode}")
        if not control_thread:
            flags.append("-DUVM_NO_CONTROL_THREAD")
        flags += [
            "-DTRACKING_ENABLED",
            f"-fpass-plugin={PASS_PATH}",
            LIB_PATH,
        ]
        if control_thread:
            flags.append(CTL_PATH)

    flags += ["-o", out]

    if extra_flags:
        flags.extend(extra_flags)

    if dry_run:
        print(" ".join([CLANG] + flags))
        return True, ""

    r = subprocess.run([CLANG] + flags,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, cwd=ROOT)
    return r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_env(with_preload=False, preload_managed=None, preload_device=None,
              extra_env=None):
    """
    Build an env dict from named flags.

    Parameters
    ----------
    with_preload    : set LD_PRELOAD to libMallocIntercept.so
    preload_managed : True/False → UVM_PRELOAD_MANAGED=1/0  (None = leave unset)
    preload_device  : True/False → UVM_PRELOAD_DEVICE=1/0   (None = leave unset)
    extra_env       : dict of additional variables merged last
    """
    env = os.environ.copy()
    if with_preload:
        env["LD_PRELOAD"] = PRELOAD_SO
    if preload_managed is not None:
        env["UVM_PRELOAD_MANAGED"] = "1" if preload_managed else "0"
    if preload_device is not None:
        env["UVM_PRELOAD_DEVICE"] = "1" if preload_device else "0"
    if extra_env:
        env.update(extra_env)
    return env


def _resolve_binary(binary):
    """Return an absolute path for binary (relative paths resolved against ROOT)."""
    if not os.path.isabs(binary):
        return os.path.join(ROOT, binary)
    return binary


# ═══════════════════════════════════════════════════════════════════════════════
# One-shot kernel runs
# ═══════════════════════════════════════════════════════════════════════════════

def run_kernel(binary, binary_flags=None, *, env=None, timeout=30):
    """
    Run a compiled binary and return the path to the generated pagelog.

    The binary is expected to write access_log.bin in the project root.
    The caller is responsible for removing the returned file when done.

    Parameters
    ----------
    binary       : path to the compiled binary
    binary_flags : list of CLI arguments passed to the binary
                   (e.g. ["0"] to select kernel 0, or ["8"] for 8-second run)
    env          : full environment dict; None inherits the current environment
    timeout      : seconds before the run is killed

    Returns
    -------
    (pagelog_path | None, stderr: str)
    """
    binary = _resolve_binary(binary)
    if os.path.exists(ACCESS_LOG):
        os.remove(ACCESS_LOG)

    cmd = [binary] + (binary_flags or [])
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=env, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, "timed out"

    if r.returncode != 0:
        return None, r.stderr
    if not os.path.exists(ACCESS_LOG):
        return None, "no access_log.bin produced"
    return ACCESS_LOG, r.stderr


def run_kernel_ex(binary, binary_flags=None, *,
                  with_preload=False,
                  preload_managed=None,
                  preload_device=None,
                  extra_env=None,
                  timeout=30):
    """
    Verbose wrapper around run_kernel with named environment flags.

    Parameters
    ----------
    binary          : path to the compiled binary
    binary_flags    : list of CLI arguments passed to the binary
    with_preload    : set LD_PRELOAD=libMallocIntercept.so
    preload_managed : True → UVM_PRELOAD_MANAGED=1
                      False → UVM_PRELOAD_MANAGED=0
                      None → leave unset
    preload_device  : True → UVM_PRELOAD_DEVICE=1
                      False → UVM_PRELOAD_DEVICE=0
                      None → leave unset
    extra_env       : dict of additional environment variables (applied last)
    timeout         : seconds before the run is killed

    Returns
    -------
    (pagelog_path | None, stderr: str)
    """
    env = _make_env(with_preload=with_preload,
                    preload_managed=preload_managed,
                    preload_device=preload_device,
                    extra_env=extra_env)
    return run_kernel(binary, binary_flags, env=env, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# Long-running socket-controlled binaries  (long_running_test style)
# ═══════════════════════════════════════════════════════════════════════════════

def send_socket_cmd(proc_or_path, command, *, timeout=2.0):
    """
    Send a command to the control-thread Unix socket and return the response.

    Parameters
    ----------
    proc_or_path : a Popen object (PID used to locate socket) or the socket
                   path string directly
    command      : command string, e.g. "STATUS", "SNAPSHOT /tmp/snap.pglog"
    timeout      : seconds to wait for a response

    Returns
    -------
    response string (or an error message starting with "<error:")
    """
    if isinstance(proc_or_path, str):
        sock_path = proc_or_path
    else:
        sock_path = f"/tmp/uvm-ctl.{proc_or_path.pid}"

    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((command + "\n").encode())
        resp = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp += chunk
        except _socket.timeout:
            pass
        s.close()
        return resp.decode()
    except Exception as e:
        return f"<error: {e}>"


def launch_socket_binary(binary, binary_flags=None, *,
                         with_preload=False,
                         preload_managed=None,
                         preload_device=None,
                         extra_env=None,
                         init_timeout=3.0):
    """
    Launch a long-running binary that exposes a control-thread Unix socket.

    Waits up to init_timeout seconds for the socket file to appear, falling
    back to a glob scan of /tmp/uvm-ctl.* if the PID-derived path isn't found.

    Parameters
    ----------
    binary        : path to the compiled binary
    binary_flags  : list of CLI arguments (e.g. ["8"] for an 8-second run)
    with_preload  : set LD_PRELOAD
    preload_managed / preload_device : UVM_PRELOAD_* env vars
    extra_env     : additional environment variables
    init_timeout  : seconds to wait for the socket to appear

    Returns
    -------
    (proc: Popen, sock_path: str)
    Raises RuntimeError if the socket never appears.

    Usage
    -----
        proc, sock = launch_socket_binary("build/long_running_test", ["8"],
                                          with_preload=True)
        resp = send_socket_cmd(sock, "STATUS")
        send_socket_cmd(sock, "SHUTDOWN")
        proc.wait(timeout=5)
    """
    binary = _resolve_binary(binary)
    env = _make_env(with_preload=with_preload,
                    preload_managed=preload_managed,
                    preload_device=preload_device,
                    extra_env=extra_env)

    cmd = [binary] + (binary_flags or [])
    proc = subprocess.Popen(cmd, env=env, cwd=ROOT,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    # Poll for the socket file
    pid_sock = f"/tmp/uvm-ctl.{proc.pid}"
    deadline = time.monotonic() + init_timeout
    while time.monotonic() < deadline:
        if os.path.exists(pid_sock):
            return proc, pid_sock
        time.sleep(0.05)

    # Fallback: scan for any uvm-ctl socket (handles fork-based PID shifts)
    socks = glob.glob("/tmp/uvm-ctl.*")
    if socks:
        return proc, socks[0]

    proc.kill()
    proc.wait()
    raise RuntimeError(
        f"socket {pid_sock} never appeared after {init_timeout}s"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# READY / GO handshake binaries  (thread_mode_tests style)
# ═══════════════════════════════════════════════════════════════════════════════

def run_with_handshake(binary, binary_flags=None, *,
                       socket_cmds=None,
                       with_preload=False,
                       preload_managed=None,
                       preload_device=None,
                       extra_env=None,
                       ready_timeout=10,
                       run_timeout=60):
    """
    Run a binary that uses the READY / GO stdin handshake protocol.

    Protocol
    --------
    1. Binary prints "READY" to stdout when tracking is initialised.
    2. (Optional) caller sends socket commands before releasing the binary.
    3. Caller writes "GO\\n" to the binary's stdin.
    4. Binary runs its kernel(s), writes access_log.bin, and exits.

    Parameters
    ----------
    binary          : path to the compiled binary
    binary_flags    : extra CLI arguments
    socket_cmds     : list of command strings sent to the Unix socket before GO
    with_preload    : set LD_PRELOAD
    preload_managed / preload_device : UVM_PRELOAD_* env vars (True/False/None)
    extra_env       : additional environment variables
    ready_timeout   : seconds to wait for the "READY" line
    run_timeout     : seconds for the binary to finish after GO

    Returns
    -------
    (pagelog_path | None, socket_responses: dict[cmd→str], stderr: str)
    """
    binary = _resolve_binary(binary)
    env = _make_env(with_preload=with_preload,
                    preload_managed=preload_managed,
                    preload_device=preload_device,
                    extra_env=extra_env)

    if os.path.exists(ACCESS_LOG):
        os.remove(ACCESS_LOG)

    cmd = [binary] + (binary_flags or [])
    proc = subprocess.Popen(cmd, env=env, cwd=ROOT,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    # Wait for READY using a daemon thread so we can time out cleanly
    _got = []

    def _read():
        _got.append(proc.stdout.readline().strip())

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=ready_timeout)

    if not _got or _got[0] != "READY":
        proc.kill()
        _, stderr = proc.communicate()
        got = _got[0] if _got else "<timeout>"
        return None, {}, f"expected READY, got '{got}'\n{stderr}"

    # Send optional socket commands
    responses = {}
    for cmd_str in (socket_cmds or []):
        time.sleep(0.05)   # brief pause so the socket listener is ready
        responses[cmd_str] = send_socket_cmd(proc, cmd_str)

    # Release the binary
    try:
        proc.stdin.write("GO\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass

    try:
        _, stderr = proc.communicate(timeout=run_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None, responses, "timed out after GO"

    if not os.path.exists(ACCESS_LOG):
        return None, responses, f"no access_log.bin\n{stderr}"
    return ACCESS_LOG, responses, stderr


# ═══════════════════════════════════════════════════════════════════════════════
# Pagelog utilities
# ═══════════════════════════════════════════════════════════════════════════════

def parse_pagelog(path):
    """
    Parse a binary pagelog and return the set of touched 4 KB page addresses.
    """
    with open(path, "rb") as f:
        raw = f.read()
    hdr = struct.unpack_from(_HEADER_FMT, raw)
    if hdr[0] != _MAGIC:
        raise ValueError(f"bad pagelog magic 0x{hdr[0]:08X} in {path}")
    l3_bytes   = hdr[4]
    l1_shift, l2_shift, l3_shift = hdr[5], hdr[6], hdr[7]
    num_leaves = hdr[8]
    pages = set()
    for k in range(num_leaves):
        l1, l2, data_off = struct.unpack_from(_INDEX_FMT, raw,
                                              _HEADER_SZ + k * _INDEX_SZ)
        for w in range(l3_bytes // 8):
            word = struct.unpack_from("<Q", raw, data_off + w * 8)[0]
            if not word:
                continue
            for b in range(64):
                if word & (1 << b):
                    l3_off = w * 64 + b
                    pages.add(
                        (l1 << l1_shift) | (l2 << l2_shift) | (l3_off << l3_shift)
                    )
    return pages


def count_pages(path):
    """Return the number of unique 4 KB pages recorded in a pagelog file."""
    return len(parse_pagelog(path))


def _cluster_sizes(pages):
    if not pages:
        return []
    PAGE = 4096
    sp = sorted(pages)
    sizes, run = [], 1
    for prev, cur in zip(sp, sp[1:]):
        if cur == prev + PAGE:
            run += 1
        else:
            sizes.append(run)
            run = 1
    sizes.append(run)
    return sorted(sizes)


def fingerprint_match(generated, baseline):
    """
    Compare two pagelogs by allocation fingerprint
    (total page count + sorted contiguous-cluster sizes).

    Returns
    -------
    (ok: bool, detail: str)
    """
    try:
        g = parse_pagelog(generated)
        b = parse_pagelog(baseline)
    except Exception as e:
        return False, f"parse error: {e}"

    g_total, g_sizes = len(g), _cluster_sizes(g)
    b_total, b_sizes = len(b), _cluster_sizes(b)

    if g_total != b_total:
        return False, f"page count {g_total} vs baseline {b_total}"
    if g_sizes != b_sizes:
        return False, f"cluster layout differs ({len(g_sizes)} vs {len(b_sizes)} clusters)"
    return True, f"{g_total} pages, {len(g_sizes)} cluster(s)"


# ═══════════════════════════════════════════════════════════════════════════════
# Preflight
# ═══════════════════════════════════════════════════════════════════════════════

def skip_if_missing(*paths, message=None):
    """
    Exit 0 (SKIP) if any of the given paths do not exist.

    Prints "SKIP: <message>" and lists the missing paths, then exits with
    code 0 so the test runner records it as a non-failure.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"SKIP: {message or 'required file(s) missing'}")
        for p in missing:
            print(f"  {p}")
        sys.exit(0)
