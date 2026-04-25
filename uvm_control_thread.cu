#include "uvm_control_thread.h"
#include "common.h"
#include "tracking.h"

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

struct DeferredL3 { void* ptr; };
static std::vector<DeferredL3>  g_deferred;
static std::mutex               g_deferred_mtx;

// ── Helpers ──────────────────────────────────────────────────────────────────
static inline void*** shadow_l1_ptr(void)
{
    return (void***)g_uvm_shadow_l1;
}

// Mode-2: CPU directly nulls managed L2 entries and queues L3 for deferred free.
// Modes 0/1: reads L2 to host, nulls entries, writes back, queues L3.
static void drop_range(uintptr_t start, size_t len)
{
    if (!g_uvm_shadow_l1 || len == 0) return;

    uintptr_t end = start + len - 1;
    uint32_t l1_s = (uint32_t)((start >> L1_SHIFT) & L1_MASK);
    uint32_t l1_e = (uint32_t)((end   >> L1_SHIFT) & L1_MASK);

#if UVM_TRACKING_MODE == 2
    void*** l1 = shadow_l1_ptr();
    for (uint32_t li = l1_s; li <= l1_e; ++li) {
        if (!l1[li]) continue;
        void** l2 = (void**)l1[li];
        uint32_t l2_s = (li == l1_s) ? (uint32_t)((start >> L2_SHIFT) & L2_MASK) : 0;
        uint32_t l2_e = (li == l1_e) ? (uint32_t)((end   >> L2_SHIFT) & L2_MASK) : (L2_ENTRIES - 1);
        for (uint32_t lj = l2_s; lj <= l2_e; ++lj) {
            if (l2[lj]) {
                std::lock_guard<std::mutex> lock(g_deferred_mtx);
                g_deferred.push_back({l2[lj]});
                l2[lj] = nullptr;
            }
        }
    }
#else
    void** h_l1[L1_ENTRIES];
    cudaError_t err = cudaMemcpy(h_l1, g_uvm_shadow_l1,
                                 L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return;

    for (uint32_t li = l1_s; li <= l1_e; ++li) {
        if (!h_l1[li]) continue;
        void* h_l2[L2_ENTRIES];
        cudaMemcpy(h_l2, h_l1[li], L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);

        uint32_t l2_s = (li == l1_s) ? (uint32_t)((start >> L2_SHIFT) & L2_MASK) : 0;
        uint32_t l2_e = (li == l1_e) ? (uint32_t)((end   >> L2_SHIFT) & L2_MASK) : (L2_ENTRIES - 1);

        for (uint32_t lj = l2_s; lj <= l2_e; ++lj) {
            if (h_l2[lj]) {
                std::lock_guard<std::mutex> lock(g_deferred_mtx);
                g_deferred.push_back({h_l2[lj]});
                h_l2[lj] = nullptr;
            }
        }
        cudaMemcpy(h_l1[li], h_l2, L2_ENTRIES * sizeof(void*), cudaMemcpyHostToDevice);
    }
#endif
}

// Walk the current page table and write a binary pagelog.
// For modes 0/1 this uses the existing export_binary (copy kernels + cudaMemcpy).
// For mode 2 it walks managed L1/L2 directly.
// Clear all L3 bitmaps to zero while preserving L1/L2 structure.
static void do_clear(void)
{
    if (!g_uvm_shadow_l1) return;

    std::vector<void**> h_l1(L1_ENTRIES);
    cudaMemcpy(h_l1.data(), g_uvm_shadow_l1,
               L1_ENTRIES * sizeof(void**), cudaMemcpyDeviceToHost);

    for (int i = 0; i < L1_ENTRIES; i++) {
        if (!h_l1[i]) continue;
        std::vector<void*> h_l2(L2_ENTRIES);
        cudaMemcpy(h_l2.data(), h_l1[i],
                   L2_ENTRIES * sizeof(void*), cudaMemcpyDeviceToHost);
        for (int j = 0; j < L2_ENTRIES; j++) {
            if (h_l2[j]) {
                cudaMemset(h_l2[j], 0, L3_BYTES);
            }
        }
    }
    cudaDeviceSynchronize();
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

    // Process deferred free queue (freed now that GPU is synced inside export_binary)
    std::lock_guard<std::mutex> df_lock(g_deferred_mtx);
    for (auto& d : g_deferred) {
        cudaFree(d.ptr);
    }
    g_deferred.clear();
}

static void send_reply(int fd, const std::string& msg)
{
    (void)write(fd, msg.c_str(), msg.length());
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
        {
            std::lock_guard<std::mutex> lock(g_deferred_mtx);
            oss << "deferred_l3: " << g_deferred.size() << "\n";
        }
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
