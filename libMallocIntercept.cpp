// ═══════════════════════════════════════════════════════════════════════════════
// LD_PRELOAD wrapper: cudaMallocManaged / cudaMalloc & Kernel Launch Intercept
//
// 1. Eager shadow page table pre-population on host allocation (cudaMalloc/Managed)
// 2. Policy-driven dual-variant kernel dispatch:
//    - PREDICT_HOST: Predicts memory footprint on host from args in O(1), marks shadow
//      page table / prefetches, and runs clean kernel with 0.0% GPU overhead.
//    - INSTRUMENT_GPU: Switches to instrumented (__uvm_tracked_*) kernel dynamically.
//    - SKIP_TRACKING / DISABLED: Runs clean kernel directly.
//
// Build:
//   clang++-20 -shared -fPIC -O2 -I./include libMallocIntercept.cpp -o libMallocIntercept.so -ldl
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
#include "include/uvm_policy.h"

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

static void (*preload_range_fn)(uintptr_t, size_t)         = nullptr;
static void (*drop_range_fn)(uintptr_t, size_t)            = nullptr;
static void (*mark_predicted_range_fn)(uintptr_t, size_t, void*) = nullptr;
static void**  g_shadow_l1_ptr                             = nullptr;

// Control-thread symbols
static void (*start_ctl_fn)(void)                          = nullptr;
static int*  g_preload_managed_ptr                         = nullptr;
static int*  g_uvm_tracking_enabled_host_ptr               = nullptr;

static std::recursive_mutex     g_mutex;
static std::mutex               g_kernel_mutex;

// ── Kernel Mapping & Policy Tables ────────────────────────────────────────────
static std::unordered_map<const void*, const void*>             g_runtime_map;
static std::unordered_map<const void*, std::string>             g_runtime_names;
static std::unordered_map<const void*, KernelPolicyDescriptor>  g_policy_map;
static std::vector<void*>                                       g_dummy_stubs;

// Driver API: CUfunction (native) -> CUfunction (tracked)
static std::unordered_map<CUfunction, CUfunction>              g_driver_map;
static std::unordered_map<CUfunction, KernelPolicyDescriptor>  g_driver_policy_map;

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

// ── Dynamic Bytecode Execution Engine ───────────────────────────────────────
struct KernelBytecodePolicy {
    std::string name;
    uint16_t flags = 0;
    uint16_t num_ranges = 0;
    std::vector<uint8_t> bytecode;
};

static std::unordered_map<std::string, KernelBytecodePolicy> g_metadata_registry;

static void load_metadata_cache() {
    static bool loaded = false;
    if (loaded) return;
    loaded = true;

    const char* paths[] = {".uvm_metadata.cache", "/tmp/.uvm_metadata.cache"};
    for (const char* path : paths) {
        FILE* fp = fopen(path, "rb");
        if (!fp) continue;

        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        if (file_size < (long)sizeof(UvmKernelMetadataHeader)) {
            fclose(fp);
            continue;
        }

        std::vector<uint8_t> buf(file_size);
        if (fread(buf.data(), 1, file_size, fp) == (size_t)file_size) {
            size_t off = 0;
            while (off + sizeof(UvmKernelMetadataHeader) <= buf.size()) {
                const auto* hdr = reinterpret_cast<const UvmKernelMetadataHeader*>(buf.data() + off);
                if (hdr->magic != UVM_METADATA_MAGIC) {
                    off++;
                    continue;
                }
                size_t table_end = off + hdr->table_size;
                if (table_end > buf.size()) break;
                off += sizeof(UvmKernelMetadataHeader);

                for (uint32_t k = 0; k < hdr->num_kernels && off + sizeof(UvmKernelEntryHeader) <= table_end; ++k) {
                    const auto* entry = reinterpret_cast<const UvmKernelEntryHeader*>(buf.data() + off);
                    off += sizeof(UvmKernelEntryHeader);
                    if (off + entry->name_len + 1 + entry->bytecode_len > table_end) break;

                    std::string kname(reinterpret_cast<const char*>(buf.data() + off), entry->name_len);
                    off += entry->name_len + 1; // skip string and null

                    std::vector<uint8_t> bc(buf.data() + off, buf.data() + off + entry->bytecode_len);
                    off += entry->bytecode_len;

                    KernelBytecodePolicy pol;
                    pol.name = kname;
                    pol.flags = entry->flags;
                    pol.num_ranges = entry->num_ranges;
                    pol.bytecode = std::move(bc);
                    g_metadata_registry[kname] = pol;
                }
            }
        }
        fclose(fp);
    }
}

static bool evaluate_bytecode(
    const uint8_t* bc, size_t bc_len,
    void** args,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    const std::vector<ActiveAlloc>& active_allocs,
    std::vector<std::pair<uintptr_t, size_t>>& out_ranges
) {
    if (!bc || bc_len == 0) return false;
    uint64_t stack[64];
    int sp = 0;

    auto push = [&](uint64_t val) {
        if (sp < 64) stack[sp++] = val;
    };
    auto pop = [&]() -> uint64_t {
        return (sp > 0) ? stack[--sp] : 0;
    };

    size_t ip = 0;
    while (ip < bc_len) {
        uint8_t op = bc[ip++];
        if (op == (uint8_t)UvmOpcode::OP_END) break;

        switch ((UvmOpcode)op) {
        case UvmOpcode::OP_CONST_U64: {
            if (ip + sizeof(uint64_t) > bc_len) return false;
            uint64_t v = 0;
            memcpy(&v, bc + ip, sizeof(uint64_t));
            ip += sizeof(uint64_t);
            push(v);
            break;
        }
        case UvmOpcode::OP_CONST_U32: {
            if (ip + sizeof(uint32_t) > bc_len) return false;
            uint32_t v = 0;
            memcpy(&v, bc + ip, sizeof(uint32_t));
            ip += sizeof(uint32_t);
            push(v);
            break;
        }
        case UvmOpcode::OP_ARG_U8: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push(*(uint8_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_I8: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push((int64_t)*(int8_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_U16: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push(*(uint16_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_I16: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push((int64_t)*(int16_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_U32: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push(*(uint32_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_I32: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push((int64_t)*(int32_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_U64: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push(*(uint64_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_I64: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push((int64_t)*(int64_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_ARG_PTR: {
            uint8_t idx = bc[ip++];
            if (args && args[idx]) push(*(uintptr_t*)args[idx]); else push(0);
            break;
        }
        case UvmOpcode::OP_GRID_DIM_X: push(gx); break;
        case UvmOpcode::OP_GRID_DIM_Y: push(gy); break;
        case UvmOpcode::OP_GRID_DIM_Z: push(gz); break;
        case UvmOpcode::OP_BLOCK_DIM_X: push(bx); break;
        case UvmOpcode::OP_BLOCK_DIM_Y: push(by); break;
        case UvmOpcode::OP_BLOCK_DIM_Z: push(bz); break;

        case UvmOpcode::OP_ADD: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a + b);
            break;
        }
        case UvmOpcode::OP_SUB: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a - b);
            break;
        }
        case UvmOpcode::OP_MUL: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a * b);
            break;
        }
        case UvmOpcode::OP_DIV: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(b != 0 ? (a / b) : 0);
            break;
        }
        case UvmOpcode::OP_SHL: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a << b);
            break;
        }
        case UvmOpcode::OP_MAX: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a > b ? a : b);
            break;
        }
        case UvmOpcode::OP_MIN: {
            uint64_t b = pop();
            uint64_t a = pop();
            push(a < b ? a : b);
            break;
        }
        case UvmOpcode::OP_EMIT_RANGE: {
            uint64_t size = pop();
            uint64_t base = pop();
            if (base != 0 && size > 0) {
                out_ranges.push_back({(uintptr_t)base, (size_t)size});
            }
            break;
        }
        case UvmOpcode::OP_EMIT_ARG_RANGE: {
            uint8_t arg_idx = bc[ip++];
            uint64_t size = pop();
            if (args && args[arg_idx]) {
                uintptr_t base = *(uintptr_t*)args[arg_idx];
                if (base != 0 && size > 0) {
                    out_ranges.push_back({base, (size_t)size});
                }
            }
            break;
        }
        case UvmOpcode::OP_SCAN_ACTIVE_ALLOCS: {
            uint8_t arg_idx = bc[ip++];
            if (args) {
                void* scan_buf = (arg_idx != 0xFF && args[arg_idx]) ? args[arg_idx] : args[0];
                if (scan_buf) {
                    uint8_t* byte_scan = (uint8_t*)scan_buf;
                    for (size_t offset = 0; offset + sizeof(uintptr_t) <= 512; offset += 4) {
                        uintptr_t cand = 0;
                        memcpy(&cand, byte_scan + offset, sizeof(uintptr_t));
                        if (cand < 0x10000) continue;
                        for (const auto& alloc : active_allocs) {
                            uintptr_t base = (uintptr_t)alloc.ptr;
                            if (cand >= base && cand < base + alloc.size) {
                                bool exists = false;
                                for (auto& r : out_ranges) if (r.first == base) { exists = true; break; }
                                if (!exists) out_ranges.push_back({base, alloc.size});
                            }
                        }
                    }
                }
            }
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

static inline void predict_and_mark_ranges(
    const KernelPolicyDescriptor& desc,
    void** args,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    void* stream
) {
    std::vector<std::pair<uintptr_t, size_t>> ranges;
    if (!desc.bytecode.empty()) {
        std::vector<ActiveAlloc> active_copy;
        {
            std::lock_guard<std::recursive_mutex> lock(g_mutex);
            active_copy = g_active;
        }
        evaluate_bytecode(desc.bytecode.data(), desc.bytecode.size(), args,
                          gx, gy, gz, bx, by, bz, active_copy, ranges);
    } else if (desc.predictor_fn) {
        desc.predictor_fn(args, gx, gy, gz, bx, by, bz, ranges);
    }
    const char* dbg = getenv("UVM_DEBUG_PREDICT");
    if (dbg && dbg[0] == '1') {
        fprintf(stderr, "[DEBUG PREDICT] Kernel %s emitted %zu ranges (active allocs: %zu)\n",
                desc.kernel_name.c_str(), ranges.size(), g_active.size());
        for (const auto& r : ranges) {
            fprintf(stderr, "   -> Range: base=%p, size=%zu (%zu pages)\n",
                    (void*)r.first, r.second, (r.second + 4095) / 4096);
        }
    }
    if (mark_predicted_range_fn) {
        for (const auto& r : ranges) {
            mark_predicted_range_fn(r.first, r.second, stream);
        }
    }
}

// ── Helper: resolve real CUDA functions ──────────────────────────────────────
static void ensure_init()
{
    if (g_init_done) return;

    load_metadata_cache();

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
    mark_predicted_range_fn =
        (void (*)(uintptr_t, size_t, void*))dlsym(RTLD_DEFAULT,
                                                  "uvm_mark_predicted_range");
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
    if (name.find("uvm_mark_predicted_range_kernel") != std::string::npos) return true;
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

        // Configure policy using compiler-generated bytecode metadata
        KernelPolicyDescriptor desc;
        desc.kernel_name = dev_name;
        desc.clean_func = hostFun;
        desc.instrumented_func = tracked_stub;
        desc.enabled = true;

        auto it = g_metadata_registry.find(dev_name);
        if (it != g_metadata_registry.end()) {
            desc.bytecode = it->second.bytecode;
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else if (dev_name.find("cutlass") != std::string::npos || dev_name.find("Gemm") != std::string::npos) {
            desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else {
            desc.strategy = TrackingStrategy::INSTRUMENT_GPU;
        }
        g_policy_map[(const void*)hostFun] = desc;
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
    process_pending();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cudaLaunchKernel) {
            return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
        }
        return cudaSuccess;
    }

    KernelPolicyDescriptor desc;
    bool has_policy = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_policy_map.find(func);
        if (it != g_policy_map.end()) {
            desc = it->second;
            has_policy = true;
        }
    }

    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    bool force_gpu = (force_gpu_env && force_gpu_env[0] == '1');

    if (has_policy && desc.enabled && !force_gpu) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cudaLaunchKernel ? real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream) : cudaSuccess;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            // Predict memory footprint dynamically from bytecode on host and mark shadow table
            predict_and_mark_ranges(desc, args, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, stream);
            // Launch CLEAN kernel with 0.0% GPU overhead!
            return real_cudaLaunchKernel ? real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream) : cudaSuccess;
        }
    }

    // Fallback: INSTRUMENT_GPU
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
    process_pending();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cudaLaunchKernelExC) {
            return real_cudaLaunchKernelExC(config, func, args);
        }
        return cudaSuccess;
    }

    KernelPolicyDescriptor desc;
    bool has_policy = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_policy_map.find(func);
        if (it != g_policy_map.end()) {
            desc = it->second;
            has_policy = true;
        }
    }

    if (has_policy && desc.enabled) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cudaLaunchKernelExC ? real_cudaLaunchKernelExC(config, func, args) : cudaSuccess;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, args, config->gridDim.x, config->gridDim.y, config->gridDim.z,
                                   config->blockDim.x, config->blockDim.y, config->blockDim.z, config->stream);
            return real_cudaLaunchKernelExC ? real_cudaLaunchKernelExC(config, func, args) : cudaSuccess;
        }
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

                KernelPolicyDescriptor desc;
                desc.kernel_name = sname;
                desc.enabled = true;

                auto it = g_metadata_registry.find(sname);
                if (it != g_metadata_registry.end()) {
                    desc.bytecode = it->second.bytecode;
                    desc.strategy = TrackingStrategy::PREDICT_HOST;
                } else if (sname.find("cutlass") != std::string::npos || sname.find("Gemm") != std::string::npos) {
                    desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
                    desc.strategy = TrackingStrategy::PREDICT_HOST;
                } else {
                    desc.strategy = TrackingStrategy::INSTRUMENT_GPU;
                }
                g_driver_policy_map[*hfunc] = desc;
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
    process_pending();
    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        if (real_cuLaunchKernel) {
            return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                       blockDimX, blockDimY, blockDimZ,
                                       sharedMemBytes, hStream, kernelParams, extra);
        }
        return 0;
    }

    KernelPolicyDescriptor desc;
    bool has_policy = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_policy_map.find(f);
        if (it != g_driver_policy_map.end()) {
            desc = it->second;
            has_policy = true;
        }
    }

    if (has_policy && desc.enabled) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cuLaunchKernel ? real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                                             blockDimX, blockDimY, blockDimZ,
                                                             sharedMemBytes, hStream, kernelParams, extra) : 0;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, kernelParams, gridDimX, gridDimY, gridDimZ,
                                   blockDimX, blockDimY, blockDimZ, (void*)hStream);
            return real_cuLaunchKernel ? real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                                             blockDimX, blockDimY, blockDimZ,
                                                             sharedMemBytes, hStream, kernelParams, extra) : 0;
        }
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

    KernelPolicyDescriptor desc;
    bool has_policy = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_policy_map.find(f);
        if (it != g_driver_policy_map.end()) {
            desc = it->second;
            has_policy = true;
        }
    }

    if (has_policy && desc.enabled) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cuLaunchKernelEx ? real_cuLaunchKernelEx(config, f, kernelParams, extra) : 0;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, kernelParams, config->gridDimX, config->gridDimY, config->gridDimZ,
                                   config->blockDimX, config->blockDimY, config->blockDimZ, (void*)config->hStream);
            return real_cuLaunchKernelEx ? real_cuLaunchKernelEx(config, f, kernelParams, extra) : 0;
        }
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

// ── Intercepted cuBLAS API (Host-Side Prediction for Closed-Source BLAS) ──────
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef void* cublasHandle_t;
typedef cublasStatus_t (*cublasSgemm_v2_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasGetStream_v2_t)(cublasHandle_t, cudaStream_t*);

static cublasSgemm_v2_t real_cublasSgemm_v2 = nullptr;
static cublasGetStream_v2_t real_cublasGetStream_v2 = nullptr;

extern "C" cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
) {
    ensure_init();
    process_pending();
    fprintf(stderr, "[DEBUG] cublasSgemm_v2 intercepted: A=%p, B=%p, C=%p, m=%d, n=%d, k=%d\n", A, B, C, m, n, k);
    if (!real_cublasSgemm_v2) {
        real_cublasSgemm_v2 = (cublasSgemm_v2_t)dlsym(RTLD_NEXT, "cublasSgemm_v2");
        if (!real_cublasSgemm_v2) real_cublasSgemm_v2 = (cublasSgemm_v2_t)dlsym(RTLD_NEXT, "cublasSgemm");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)dlsym(RTLD_NEXT, "cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)dlsym(RTLD_NEXT, "cublasGetStream");
    }

    if (is_tracking_enabled_host() && mark_predicted_range_fn) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        if (A && m > 0 && k > 0) mark_predicted_range_fn((uintptr_t)A, (size_t)m * k * sizeof(float), stream);
        if (B && k > 0 && n > 0) mark_predicted_range_fn((uintptr_t)B, (size_t)k * n * sizeof(float), stream);
        if (C && m > 0 && n > 0) mark_predicted_range_fn((uintptr_t)C, (size_t)m * n * sizeof(float), stream);
    }

    if (real_cublasSgemm_v2) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return 0;
}

extern "C" cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
) {
    return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
