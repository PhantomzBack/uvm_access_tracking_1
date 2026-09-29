// ═══════════════════════════════════════════════════════════════════════════════
// LD_PRELOAD wrapper: cudaMallocManaged / cudaMalloc & Kernel Launch Intercept
//
// 1. Eager shadow page table pre-population on host allocation (cudaMalloc/Managed)
// 2. Dual-variant kernel dispatch: dynamically switches between native (0% overhead)
//    and instrumented (__uvm_tracked_*) kernels at launch time.
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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// ── Minimal CUDA runtime types ────────────────────────────────────────────────
typedef int cudaError_t;
#define cudaSuccess 0

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};
typedef void* cudaStream_t;

typedef struct cudaLaunchConfig_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    void *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;

// ── Minimal CUDA Driver API types ─────────────────────────────────────────────
typedef void* CUfunction;
typedef void* CUmodule;
typedef void* CUstream;
typedef int   CUresult;

typedef struct CUlaunchConfig_st {
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

// ── CUDA Runtime Function Pointers ────────────────────────────────────────────
typedef cudaError_t (*cudaMallocManaged_t)(void**, size_t, unsigned int);
typedef cudaError_t (*cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*cudaFree_t)(void*);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);

typedef void (*cudaRegisterFunction_t)(
    void **fatCubinHandle,
    const char *hostFun,
    const char *deviceFun,
    const char *deviceName,
    int thread_limit,
    void *tid,
    void *bid,
    void *bDim,
    void *gDim,
    int *wSize
);

typedef cudaError_t (*cudaLaunchKernel_t)(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
);

typedef cudaError_t (*cudaLaunchKernelExC_t)(
    const cudaLaunchConfig_t *config,
    const void *func,
    void **args
);

// ── CUDA Driver API Function Pointers ─────────────────────────────────────────
typedef CUresult (*cuLaunchKernel_t)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
);

typedef CUresult (*cuLaunchKernelEx_t)(
    const CUlaunchConfig *config,
    CUfunction f,
    void **kernelParams,
    void **extra
);

typedef CUresult (*cuModuleGetFunction_t)(
    CUfunction *hfunc,
    CUmodule hmod,
    const char *name
);

// ── Real symbol pointers resolved lazily ──────────────────────────────────────
static cudaMallocManaged_t      real_cudaMallocManaged      = nullptr;
static cudaMalloc_t             real_cudaMalloc             = nullptr;
static cudaFree_t               real_cudaFree               = nullptr;
static cudaDeviceSynchronize_t  real_cudaDeviceSynchronize  = nullptr;
static cudaRegisterFunction_t   real___cudaRegisterFunction = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel       = nullptr;
static cudaLaunchKernelExC_t    real_cudaLaunchKernelExC    = nullptr;

static cuLaunchKernel_t         real_cuLaunchKernel         = nullptr;
static cuLaunchKernelEx_t       real_cuLaunchKernelEx       = nullptr;
static cuModuleGetFunction_t    real_cuModuleGetFunction    = nullptr;

static void (*preload_range_fn)(uintptr_t, size_t) = nullptr;
static void (*drop_range_fn)(uintptr_t, size_t)   = nullptr;
static void**  g_shadow_l1_ptr                   = nullptr;

// Control-thread symbols
static void (*start_ctl_fn)(void)                = nullptr;
static int*  g_preload_managed_ptr               = nullptr;
static int*  g_uvm_tracking_enabled_host_ptr     = nullptr;

static std::recursive_mutex     g_mutex;
static std::mutex               g_kernel_mutex;

// ── Kernel Mapping Tables ─────────────────────────────────────────────────────
// CUDART: host stub (native) -> synthetic host stub (tracked)
static std::unordered_map<const void*, const void*> g_runtime_map;
static std::unordered_map<const void*, std::string> g_runtime_names;
static std::vector<void*> g_dummy_stubs;

// Driver API: CUfunction (native) -> CUfunction (tracked)
static std::unordered_map<CUfunction, CUfunction> g_driver_map;

struct PendingAlloc {
    void*  ptr;
    size_t size;
    bool   is_managed;
};
static std::vector<PendingAlloc> g_pending;

struct ActiveAlloc {
    void*  ptr;
    size_t size;
};
static std::vector<ActiveAlloc>  g_active;

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
    real___cudaRegisterFunction =
        (cudaRegisterFunction_t)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    real_cudaLaunchKernel =
        (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    real_cudaLaunchKernelExC =
        (cudaLaunchKernelExC_t)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");

    real_cuLaunchKernel =
        (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    real_cuLaunchKernelEx =
        (cuLaunchKernelEx_t)dlsym(RTLD_NEXT, "cuLaunchKernelEx");
    real_cuModuleGetFunction =
        (cuModuleGetFunction_t)dlsym(RTLD_NEXT, "cuModuleGetFunction");

    preload_range_fn =
        (void (*)(uintptr_t, size_t))dlsym(RTLD_DEFAULT,
                                           "uvm_tracking_preload_range");
    drop_range_fn =
        (void (*)(uintptr_t, size_t))dlsym(RTLD_DEFAULT,
                                           "uvm_tracking_drop_range");
    g_shadow_l1_ptr =
        (void**)dlsym(RTLD_DEFAULT, "g_uvm_shadow_l1");

    // Resolve control-thread symbols (may be absent if binary lacks uvm_control_thread)
    start_ctl_fn =
        (void (*)(void))dlsym(RTLD_DEFAULT, "uvm_start_control_thread");
    g_preload_managed_ptr =
        (int*)dlsym(RTLD_DEFAULT, "g_uvm_preload_managed");
    g_uvm_tracking_enabled_host_ptr =
        (int*)dlsym(RTLD_DEFAULT, "g_uvm_tracking_enabled_host");

    g_init_done = true;

    // Spawn control thread if available (idempotent)
    const char* no_ctl = getenv("UVM_NO_CONTROL_THREAD");
    if (start_ctl_fn && (!no_ctl || no_ctl[0] != '1'))
        start_ctl_fn();
}

// ── Helper: is tracking active on host side? ─────────────────────────────────
static inline bool is_tracking_enabled_host()
{
    // If tracking runtime is not linked into the current binary, tracking is never enabled
    if (!preload_range_fn || !g_shadow_l1_ptr)
        return false;
    if (g_uvm_tracking_enabled_host_ptr)
        return *g_uvm_tracking_enabled_host_ptr != 0;
    const char* env_dis = getenv("UVM_TRACKING_DISABLED");
    return !(env_dis && env_dis[0] == '1');
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

// ── Helper: should we preload a device (cudaMalloc) allocation? ──────────────
static bool should_preload_device()
{
    const char* v = getenv("UVM_PRELOAD_DEVICE");
    return !(v && v[0] == '0');  // default: yes
}

// ── Helper: process any pending allocations ──────────────────────────────────
static void process_pending()
{
    if (!preload_range_fn) return;
    if (!shadow_l1_ready()) return;

    std::vector<PendingAlloc> to_process;
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (g_pending.empty()) return;
        to_process.swap(g_pending);
    }

    for (auto& a : to_process) {
        if (a.is_managed && !should_preload_managed()) continue;
        if (!a.is_managed && !should_preload_device()) continue;
        g_in_preload = true;
        preload_range_fn((uintptr_t)a.ptr, a.size);
        g_in_preload = false;
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_active.push_back({a.ptr, a.size});
    }
}

// ── Helper: register a new allocation ────────────────────────────────────────
static void register_alloc(void* ptr, size_t size, bool is_managed)
{
    if (!ptr || size == 0) return;
    if (g_in_preload) return;          // prevent recursion

    // Skip managed pre-population if toggled off
    if (is_managed && !should_preload_managed()) return;
    // Skip device pre-population if toggled off
    if (!is_managed && !should_preload_device()) return;

    // If the page table is already up, pre-populate immediately.
    if (preload_range_fn && shadow_l1_ready()) {
        g_in_preload = true;
        preload_range_fn((uintptr_t)ptr, size);
        g_in_preload = false;
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_active.push_back({ptr, size});
        return;
    }

    // Otherwise queue it for later (e.g. after init_tracking runs).
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    g_pending.push_back({ptr, size, is_managed});
}

static bool is_runtime_kernel(const std::string& name) {
    if (name.find("copy_l2_to_host") != std::string::npos) return true;
    if (name.find("copy_l3_to_host") != std::string::npos) return true;
    if (name.find("uvm_clear_l3s_kernel") != std::string::npos) return true;
    if (name.find("MarkAccess") != std::string::npos) return true;
    if (name.find("BatchMarkAccess") != std::string::npos) return true;
    return false;
}

static std::vector<std::string> g_tracked_names;

// ── Intercepted __cudaRegisterFunction ───────────────────────────────────────
extern "C" void __cudaRegisterFunction(
    void        **fatCubinHandle,
    const char   *hostFun,
    const char   *deviceFun,
    const char   *deviceName,
    int           thread_limit,
    void         *tid,
    void         *bid,
    void         *bDim,
    void         *gDim,
    int          *wSize
) {
    ensure_init();
    if (real___cudaRegisterFunction) {
        real___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                   thread_limit, tid, bid, bDim, gDim, wSize);
    }

    if (!deviceFun || !hostFun) return;

    std::string dev_name(deviceFun);
    // If this registration is already for a tracked kernel or runtime helper, skip
    if (dev_name.rfind("__uvm_tracked_", 0) == 0 || is_runtime_kernel(dev_name)) return;

    void* tracked_stub = (void*)(new char[64]);
    const char* tracked_dev_str = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_tracked_names.push_back("__uvm_tracked_" + dev_name);
        tracked_dev_str = g_tracked_names.back().c_str();

        g_dummy_stubs.push_back(tracked_stub);
        g_runtime_map[(const void*)hostFun] = (const void*)tracked_stub;
        g_runtime_names[(const void*)hostFun] = dev_name;
    }

    // Register synthetic stub with tracked device function in CUDA runtime
    if (real___cudaRegisterFunction) {
        real___cudaRegisterFunction(fatCubinHandle, (const char*)tracked_stub,
                                   tracked_dev_str, deviceName,
                                   thread_limit, tid, bid, bDim, gDim, wSize);
    }
}

// ── Intercepted cudaLaunchKernel ─────────────────────────────────────────────
extern "C" cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    ensure_init();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cudaLaunchKernel) {
            return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
        }
        return cudaSuccess;
    }

    const void* launch_func = func;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_runtime_map.find(func);
        if (it != g_runtime_map.end()) {
            launch_func = it->second;
        }
    }
    if (real_cudaLaunchKernel) {
        cudaError_t err = real_cudaLaunchKernel(launch_func, gridDim, blockDim, args, sharedMem, stream);
        if (err != cudaSuccess && launch_func != func) {
            return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
        }
        return err;
    }
    return cudaSuccess;
}

// ── Intercepted cudaLaunchKernelExC ──────────────────────────────────────────
extern "C" cudaError_t cudaLaunchKernelExC(
    const cudaLaunchConfig_t *config,
    const void *func,
    void **args
) {
    ensure_init();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cudaLaunchKernelExC) {
            return real_cudaLaunchKernelExC(config, func, args);
        }
        return cudaSuccess;
    }

    const void* launch_func = func;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_runtime_map.find(func);
        if (it != g_runtime_map.end()) {
            launch_func = it->second;
        }
    }
    if (real_cudaLaunchKernelExC) {
        cudaError_t err = real_cudaLaunchKernelExC(config, launch_func, args);
        if (err != cudaSuccess && launch_func != func) {
            return real_cudaLaunchKernelExC(config, func, args);
        }
        return err;
    }
    return cudaSuccess;
}

// ── Intercepted cuModuleGetFunction ──────────────────────────────────────────
extern "C" CUresult cuModuleGetFunction(
    CUfunction *hfunc,
    CUmodule hmod,
    const char *name
) {
    ensure_init();
    if (!real_cuModuleGetFunction) return 0;
    CUresult res = real_cuModuleGetFunction(hfunc, hmod, name);
    if (res == 0 && hfunc && *hfunc && name) {
        std::string sname(name);
        if (sname.rfind("__uvm_tracked_", 0) != 0) {
            std::string tracked_name = "__uvm_tracked_" + sname;
            CUfunction tracked_func = nullptr;
            CUresult tr_res = real_cuModuleGetFunction(&tracked_func, hmod, tracked_name.c_str());
            if (tr_res == 0 && tracked_func) {
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                g_driver_map[*hfunc] = tracked_func;
            }
        }
    }
    return res;
}

// ── Intercepted cuLaunchKernel ───────────────────────────────────────────────
extern "C" CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    ensure_init();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cuLaunchKernel) {
            return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                       blockDimX, blockDimY, blockDimZ,
                                       sharedMemBytes, hStream, kernelParams, extra);
        }
        return 0;
    }

    CUfunction launch_f = f;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_map.find(f);
        if (it != g_driver_map.end()) {
            launch_f = it->second;
        }
    }
    if (real_cuLaunchKernel) {
        return real_cuLaunchKernel(launch_f, gridDimX, gridDimY, gridDimZ,
                                   blockDimX, blockDimY, blockDimZ,
                                   sharedMemBytes, hStream, kernelParams, extra);
    }
    return 0;
}

// ── Intercepted cuLaunchKernelEx ─────────────────────────────────────────────
extern "C" CUresult cuLaunchKernelEx(
    const CUlaunchConfig *config,
    CUfunction f,
    void **kernelParams,
    void **extra
) {
    ensure_init();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cuLaunchKernelEx) {
            return real_cuLaunchKernelEx(config, f, kernelParams, extra);
        }
        return 0;
    }

    CUfunction launch_f = f;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_map.find(f);
        if (it != g_driver_map.end()) {
            launch_f = it->second;
        }
    }
    if (real_cuLaunchKernelEx) {
        return real_cuLaunchKernelEx(config, launch_f, kernelParams, extra);
    }
    return 0;
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
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        // Remove from pending queue (was never preloaded)
        for (auto it = g_pending.begin(); it != g_pending.end(); ) {
            if (it->ptr == ptr)
                it = g_pending.erase(it);
            else
                ++it;
        }
        // If we are freeing the L1 table itself, null the shadow pointer
        // so that later drop_range calls bail out early.
        if (g_shadow_l1_ptr && *g_shadow_l1_ptr == ptr) {
            *g_shadow_l1_ptr = nullptr;
        }
        // Drop from shadow page table if it was preloaded
        for (auto it = g_active.begin(); it != g_active.end(); ) {
            if (it->ptr == ptr) {
                if (drop_range_fn)
                    drop_range_fn((uintptr_t)it->ptr, it->size);
                it = g_active.erase(it);
            } else {
                ++it;
            }
        }
    }
    return real_cudaFree(ptr);
}

// ── Intercepted cudaDeviceSynchronize ────────────────────────────────────────
extern "C" cudaError_t cudaDeviceSynchronize(void)
{
    ensure_init();
    process_pending();
    return real_cudaDeviceSynchronize();
}
