// ═══════════════════════════════════════════════════════════════════════════════
// LD_PRELOAD wrapper: cudaMallocManaged / cudaMalloc intercept
//
// After every successful allocation we eagerly pre-populate the shadow page
// table (L1→L2→L3) on the host side so that GPU threads never have to call
// device-side malloc() when they hit MarkAccess / BatchMarkAccess.
//
// Build:
//   clang++-20 -shared -fPIC -O2 libMallocIntercept.cpp -o libMallocIntercept.so -ldl
//
// Usage:
//   LD_PRELOAD=./libMallocIntercept.so ./my_cuda_app
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <mutex>

// ── Minimal CUDA runtime forward declarations ────────────────────────────────
typedef int cudaError_t;
#define cudaSuccess 0

typedef cudaError_t (*cudaMallocManaged_t)(void**, size_t, unsigned int);
typedef cudaError_t (*cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*cudaFree_t)(void*);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);

// ── Symbol pointers resolved lazily ──────────────────────────────────────────
static cudaMallocManaged_t    real_cudaMallocManaged    = nullptr;
static cudaMalloc_t           real_cudaMalloc           = nullptr;
static cudaFree_t             real_cudaFree             = nullptr;
static cudaDeviceSynchronize_t real_cudaDeviceSynchronize = nullptr;

static void (*preload_range_fn)(uintptr_t, size_t) = nullptr;
static void**  g_shadow_l1_ptr                   = nullptr;

// Control-thread symbols (resolved lazily from the main executable)
static void (*start_ctl_fn)(void) = nullptr;
static int*  g_preload_managed_ptr = nullptr;

static std::mutex              g_mutex;

struct PendingAlloc {
    void*  ptr;
    size_t size;
    bool   is_managed;
};
static std::vector<PendingAlloc> g_pending;

static bool                    g_init_done = false;
static thread_local bool       g_in_preload = false;

// ── Helper: resolve real CUDA functions ──────────────────────────────────────
static void ensure_init()
{
    if (g_init_done) return;

    real_cudaMallocManaged =
        (cudaMallocManaged_t)dlsym(RTLD_NEXT, "cudaMallocManaged");
    real_cudaMalloc =
        (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaFree =
        (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    real_cudaDeviceSynchronize =
        (cudaDeviceSynchronize_t)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");

    preload_range_fn =
        (void (*)(uintptr_t, size_t))dlsym(RTLD_DEFAULT,
                                           "uvm_tracking_preload_range");
    g_shadow_l1_ptr =
        (void**)dlsym(RTLD_DEFAULT, "g_uvm_shadow_l1");

    // Resolve control-thread symbols (may be absent if binary lacks uvm_control_thread)
    start_ctl_fn =
        (void (*)(void))dlsym(RTLD_DEFAULT, "uvm_start_control_thread");
    g_preload_managed_ptr =
        (int*)dlsym(RTLD_DEFAULT, "g_uvm_preload_managed");

    g_init_done = true;

    // Spawn control thread if available (idempotent)
    if (start_ctl_fn) start_ctl_fn();
}

// ── Helper: is the shadow L1 table already initialised? ──────────────────────
static bool shadow_l1_ready()
{
    if (!g_shadow_l1_ptr) return false;
    return *g_shadow_l1_ptr != nullptr;
}

// ── Helper: should we preload a managed allocation? ──────────────────────────
static bool should_preload_managed()
{
    if (!g_preload_managed_ptr) return true;  // default: yes
    return *g_preload_managed_ptr != 0;
}

// ── Helper: process any pending allocations ──────────────────────────────────
static void process_pending()
{
    if (!preload_range_fn) return;
    if (!shadow_l1_ready()) return;

    std::vector<PendingAlloc> to_process;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_pending.empty()) return;
        to_process.swap(g_pending);
    }

    for (auto& a : to_process) {
        if (a.is_managed && !should_preload_managed()) continue;
        g_in_preload = true;
        preload_range_fn((uintptr_t)a.ptr, a.size);
        g_in_preload = false;
    }
}

// ── Helper: register a new allocation ────────────────────────────────────────
static void register_alloc(void* ptr, size_t size, bool is_managed)
{
    if (!ptr || size == 0) return;
    if (g_in_preload) return;          // prevent recursion

    // Skip managed pre-population if toggled off
    if (is_managed && !should_preload_managed()) return;

    // If the page table is already up, pre-populate immediately.
    if (preload_range_fn && shadow_l1_ready()) {
        g_in_preload = true;
        preload_range_fn((uintptr_t)ptr, size);
        g_in_preload = false;
        return;
    }

    // Otherwise queue it for later (e.g. after init_tracking runs).
    std::lock_guard<std::mutex> lock(g_mutex);
    g_pending.push_back({ptr, size, is_managed});
}

// ── Intercepted cudaMallocManaged ────────────────────────────────────────────
extern "C" cudaError_t cudaMallocManaged(void** ptr, size_t size,
                                           unsigned int flags)
{
    ensure_init();
    cudaError_t err = real_cudaMallocManaged(ptr, size, flags);
    if (err == cudaSuccess) {
        register_alloc(*ptr, size, true);
    }
    return err;
}

// ── Intercepted cudaMalloc ───────────────────────────────────────────────────
extern "C" cudaError_t cudaMalloc(void** ptr, size_t size)
{
    ensure_init();
    cudaError_t err = real_cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        register_alloc(*ptr, size, false);
    }
    return err;
}

// ── Intercepted cudaFree ─────────────────────────────────────────────────────
extern "C" cudaError_t cudaFree(void* ptr)
{
    ensure_init();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto it = g_pending.begin(); it != g_pending.end(); ) {
            if (it->ptr == ptr)
                it = g_pending.erase(it);
            else
                ++it;
        }
    }
    return real_cudaFree(ptr);
}

// ── Intercepted cudaDeviceSynchronize ────────────────────────────────────────
// init_tracking ends with cudaDeviceSynchronize(), so this is a convenient
// trigger to flush any allocations that were made before the shadow L1 was
// initialised.
extern "C" cudaError_t cudaDeviceSynchronize(void)
{
    ensure_init();
    process_pending();
    return real_cudaDeviceSynchronize();
}
