#include "uvm_control_thread.h"
#include "common.h"
#include "tracking.h"

// ── Device kernel: clear every L3 bitmap reachable from L1 ───────────────────
// Runs entirely on the GPU so it works for L2 pointers that were allocated
// by device-side malloc() (host cudaMemcpy cannot touch those).
__global__ void uvm_clear_l3s_kernel(void** l1_table)
{
    int l1_idx = blockIdx.x;
    if (l1_idx >= L1_ENTRIES || !l1_table[l1_idx]) return;

    void** l2_table = (void**)l1_table[l1_idx];
    for (int l2_idx = threadIdx.x; l2_idx < L2_ENTRIES; l2_idx += blockDim.x) {
        void* l3 = l2_table[l2_idx];
        if (l3) {
            unsigned long long* bits = (unsigned long long*)l3;
            for (int k = 0; k < L3_BYTES / 8; ++k)
                bits[k] = 0ULL;
        }
    }
}

#ifndef __CUDA_ARCH__

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>
#include <string.h>
#include <cuda_runtime.h>

// ── Globals ──────────────────────────────────────────────────────────────────
int g_uvm_preload_managed = 1;

static std::thread              g_ctl_thread;
static std::atomic<bool>        g_ctl_running{false};
static std::atomic<bool>        g_ctl_shutdown{false};
static std::mutex               g_snapshot_mtx;

// ── Helpers ──────────────────────────────────────────────────────────────────
static inline void*** shadow_l1_ptr(void)
{
    return (void***)g_uvm_shadow_l1;
}

static void drop_range(uintptr_t start, size_t len)
{
    if (!g_uvm_shadow_l1 || len == 0) return;
    uvm_tracking_drop_range(start, len);
}

// Clear all L3 bitmaps to zero while preserving L1/L2 structure.
// Uses a device kernel so it works for L2 pointers allocated by device malloc().
static void do_clear(void)
{
    if (!g_uvm_shadow_l1) return;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[uvm_ctl] do_clear: cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        return;
    }

    uvm_clear_l3s_kernel<<<L1_ENTRIES, 256>>>(g_uvm_shadow_l1);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[uvm_ctl] do_clear: kernel sync failed: %s\n",
                cudaGetErrorString(err));
        return;
    }

    fprintf(stderr, "[uvm_ctl] cleared all L3 bitmaps\n");
}

static void do_snapshot(const char* path)
{
    if (!g_uvm_shadow_l1) {
        fprintf(stderr, "[uvm_ctl] snapshot: shadow_l1 not initialized\n");
        return;
    }
    std::lock_guard<std::mutex> lock(g_snapshot_mtx);

    // Reuse the existing export_binary — it already handles all three modes.
    export_binary(shadow_l1_ptr(), path);
}

static void send_reply(int fd, const std::string& msg)
{
    (void)send(fd, msg.c_str(), msg.length(), MSG_NOSIGNAL);
}

static void process_command(const std::string& cmd, int client_fd)
{
    if (cmd == "ENABLE") {
        uvm_tracking_set_enabled(1);
        send_reply(client_fd, "OK\n");
    }
    else if (cmd == "DISABLE") {
        uvm_tracking_set_enabled(0);
        send_reply(client_fd, "OK\n");
    }
    else if (cmd == "MODE SKIP") {
        uvm_tracking_set_skip_on_miss(1);
        send_reply(client_fd, "OK\n");
    }
    else if (cmd == "MODE ALLOC") {
        uvm_tracking_set_skip_on_miss(0);
        send_reply(client_fd, "OK\n");
    }
    else if (cmd.rfind("PRELOAD_MANAGED ", 0) == 0) {
        try {
            int v = std::stoi(cmd.substr(16));
            g_uvm_preload_managed = v ? 1 : 0;
            send_reply(client_fd, "OK\n");
        } catch (...) {
            send_reply(client_fd, "ERR: invalid value\n");
        }
    }
    else if (cmd.rfind("SNAPSHOT ", 0) == 0) {
        std::string path = cmd.substr(9);
        do_snapshot(path.c_str());
        send_reply(client_fd, "OK\n");
    }
    else if (cmd.rfind("DROP_RANGE ", 0) == 0) {
        std::istringstream iss(cmd.substr(11));
        uintptr_t addr = 0;
        size_t len = 0;
        iss >> std::hex >> addr >> std::dec >> len;
        if (!iss.fail()) {
            drop_range(addr, len);
            send_reply(client_fd, "OK\n");
        } else {
            send_reply(client_fd, "ERR: usage DROP_RANGE <hex-addr> <len>\n");
        }
    }
    else if (cmd == "STATUS") {
        int enabled = 0, skip = 0;
        cudaMemcpyFromSymbol(&enabled, g_tracking_enabled, sizeof(int));
        cudaMemcpyFromSymbol(&skip,    g_skip_on_miss,     sizeof(int));

        std::ostringstream oss;
        oss << "mode: " << UVM_TRACKING_MODE << "\n";
        oss << "preload_managed: " << g_uvm_preload_managed << "\n";
        oss << "tracking_enabled: " << enabled << "\n";
        oss << "skip_on_miss: " << skip << "\n";
        oss << "shadow_l1: " << g_uvm_shadow_l1 << "\n";
        send_reply(client_fd, oss.str());
    }
    else if (cmd == "CLEAR") {
        do_clear();
        send_reply(client_fd, "OK\n");
    }
    else if (cmd == "SHUTDOWN") {
        send_reply(client_fd, "OK\n");
        g_ctl_shutdown = true;
    }
    else {
        send_reply(client_fd, "ERR: unknown command\n");
    }
}

static void control_loop(void)
{
    int pid = getpid();
    std::string sock_path = "/tmp/uvm-ctl." + std::to_string(pid);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "[uvm_ctl] socket() failed: %s\n", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

    unlink(sock_path.c_str());
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[uvm_ctl] bind(%s) failed: %s\n", sock_path.c_str(), strerror(errno));
        close(fd);
        return;
    }
    if (listen(fd, 5) < 0) {
        fprintf(stderr, "[uvm_ctl] listen() failed: %s\n", strerror(errno));
        close(fd);
        unlink(sock_path.c_str());
        return;
    }

    while (!g_ctl_shutdown.load()) {
        struct pollfd pfd = { fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, 500);  // 500 ms timeout
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue;  // timeout — check shutdown flag

        int client = accept(fd, nullptr, nullptr);
        if (client < 0) continue;

        char buf[1024];
        ssize_t n = read(client, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string cmd(buf);
            // trim trailing whitespace
            size_t end = cmd.find_last_not_of(" \r\n\t");
            if (end != std::string::npos)
                cmd.erase(end + 1);
            else
                cmd.clear();
            if (!cmd.empty())
                process_command(cmd, client);
        }
        close(client);
    }

    close(fd);
    unlink(sock_path.c_str());
}

// ── Public API ───────────────────────────────────────────────────────────────
void uvm_start_control_thread(void)
{
    if (g_ctl_running.exchange(true))
        return;  // already running

    // Register a one-shot cleanup object so the thread is joined on exit.
    // std::thread's destructor *must* see join() or detach(); otherwise
    // std::terminate() is called.
    static struct CtlCleanup {
        ~CtlCleanup() { uvm_stop_control_thread(); }
    } cleanup;

    g_ctl_shutdown = false;
    g_ctl_thread = std::thread(control_loop);
}

void uvm_stop_control_thread(void)
{
    if (!g_ctl_running.load())
        return;

    g_ctl_shutdown = true;

    // Wake up poll() by connecting to our own socket
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd >= 0) {
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        std::string sock_path = "/tmp/uvm-ctl." + std::to_string(getpid());
        strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
        connect(fd, (struct sockaddr*)&addr, sizeof(addr));
        close(fd);
    }

    if (g_ctl_thread.joinable())
        g_ctl_thread.join();

    g_ctl_running = false;
}

#endif // __CUDA_ARCH__
