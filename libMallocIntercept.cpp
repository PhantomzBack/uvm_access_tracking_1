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
#include <unistd.h>
#include <signal.h>
#include <sys/uio.h>
#include <execinfo.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cxxabi.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "include/uvm_policy.h"

// ── Minimal CUDA runtime types ────────────────────────────────────────────────
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidResourceHandle 400

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

typedef cudaError_t (*cudaMemPrefetchAsync_t)(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
typedef cudaError_t (*cudaMemAdvise_t)(const void *devPtr, size_t count, int advice, int device);
typedef cudaError_t (*cudaStreamCreateWithFlags_t)(cudaStream_t *pStream, unsigned int flags);
typedef cudaError_t (*cudaGetDevice_t)(int *device);
typedef cudaError_t (*cudaStreamIsCapturing_t)(cudaStream_t stream, int *pCaptureStatus);

typedef void* cudaEvent_t;
typedef cudaError_t (*cudaEventCreateWithFlags_t)(cudaEvent_t *event, unsigned int flags);
typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t event, cudaStream_t stream);
typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t event);

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

typedef unsigned long long CUdeviceptr;
typedef CUresult (*cuMemPrefetchAsync_t)(uintptr_t devPtr, size_t count, int dstDevice, CUstream hStream);
typedef CUresult (*cuMemAdvise_t)(uintptr_t devPtr, size_t count, int advice, int device);
typedef CUresult (*cuMemAllocManaged_t)(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
typedef CUresult (*cuMemFree_t)(CUdeviceptr dptr);

// ── Real symbol pointers resolved lazily ──────────────────────────────────────
static cudaMallocManaged_t      real_cudaMallocManaged      = nullptr;
static cudaMalloc_t             real_cudaMalloc             = nullptr;
static cudaFree_t               real_cudaFree               = nullptr;
static cudaDeviceSynchronize_t  real_cudaDeviceSynchronize  = nullptr;
static cudaRegisterFunction_t   real___cudaRegisterFunction = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel       = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel     = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel_ptsz= nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel_ptsz  = nullptr;
static cudaLaunchKernelExC_t    real_cudaLaunchKernelExC    = nullptr;

// Multi-version libcudart support (PyTorch: libcudart.so.12, vLLM ops: libcudart.so.13)
static cudaRegisterFunction_t   real___cudaRegisterFunction_v12 = nullptr;
static cudaRegisterFunction_t   real___cudaRegisterFunction_v13 = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel_v12       = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel_v13       = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel_v12     = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel_v13     = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel_ptsz_v12 = nullptr;
static cudaLaunchKernel_t       real___cudaLaunchKernel_ptsz_v13 = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel_ptsz_v12  = nullptr;
static cudaLaunchKernel_t       real_cudaLaunchKernel_ptsz_v13  = nullptr;
static cudaLaunchKernelExC_t    real_cudaLaunchKernelExC_v12    = nullptr;
static cudaLaunchKernelExC_t    real_cudaLaunchKernelExC_v13    = nullptr;
static cudaError_t (*real_gle_v12)(void)                        = nullptr;
static cudaError_t (*real_gle_v13)(void)                        = nullptr;
static cudaError_t (*real_ple_v12)(void)                        = nullptr;
static cudaError_t (*real_ple_v13)(void)                        = nullptr;

typedef cudaError_t (*cudaMemAdvise_v12_t)(const void *devPtr, size_t count, int advice, int device);
typedef cudaError_t (*cudaMemAdvise_v13_t)(const void *devPtr, size_t count, int advice, uint64_t location);
static cudaMemAdvise_v12_t real_cudaMemAdvise_v12               = nullptr;
static cudaMemAdvise_v13_t real_cudaMemAdvise_v13               = nullptr;

static cudaMemPrefetchAsync_t      real_cudaMemPrefetchAsync      = nullptr;
static cudaMemAdvise_t             real_cudaMemAdvise             = nullptr;
static cudaStreamCreateWithFlags_t real_cudaStreamCreateWithFlags = nullptr;
static cudaGetDevice_t             real_cudaGetDevice             = nullptr;
static cudaStreamIsCapturing_t     real_cudaStreamIsCapturing     = nullptr;

static cudaEventCreateWithFlags_t  real_cudaEventCreateWithFlags  = nullptr;
static cudaEventRecord_t           real_cudaEventRecord           = nullptr;
static cudaStreamWaitEvent_t       real_cudaStreamWaitEvent       = nullptr;
static cudaEventDestroy_t          real_cudaEventDestroy          = nullptr;

static cuLaunchKernel_t         real_cuLaunchKernel         = nullptr;
static cuLaunchKernelEx_t       real_cuLaunchKernelEx       = nullptr;
static cuModuleGetFunction_t    real_cuModuleGetFunction    = nullptr;
static cuMemPrefetchAsync_t     real_cuMemPrefetchAsync     = nullptr;
static cuMemAdvise_t            real_cuMemAdvise            = nullptr;
static cuMemAllocManaged_t      real_cuMemAllocManaged      = nullptr;
static cuMemFree_t              real_cuMemFree              = nullptr;

static void (*preload_range_fn)(uintptr_t, size_t)         = nullptr;
static void (*drop_range_fn)(uintptr_t, size_t)            = nullptr;
static void (*mark_predicted_range_fn)(uintptr_t, size_t, void*) = nullptr;
static void**  g_shadow_l1_ptr                             = nullptr;

// Control-thread symbols
static void (*start_ctl_fn)(void)                          = nullptr;
static int*  g_preload_managed_ptr                         = nullptr;
static int*  g_uvm_tracking_enabled_host_ptr               = nullptr;

static inline void check_cuda_err(const char* name) {
    if (!getenv("UVM_DEBUG_PREDICT")) return;
    cudaError_t err12 = real_gle_v12 ? real_gle_v12() : cudaSuccess;
    cudaError_t err13 = real_gle_v13 ? real_gle_v13() : cudaSuccess;
    cudaError_t err = (err12 != cudaSuccess) ? err12 : err13;
    if (err != cudaSuccess) {
        fprintf(stderr, "[FOUND DIRTY STATE] Function %s left cudaError %d on context!\n", name, (int)err);
    }
}

static std::recursive_mutex     g_mutex;
static std::mutex               g_kernel_mutex;

// ── Kernel Mapping & Policy Tables (Construct-on-First-Use) ───────────────────
static inline std::unordered_map<const void*, const void*>& get_runtime_map() {
    static auto* m = new std::unordered_map<const void*, const void*>();
    return *m;
}
#define g_runtime_map (get_runtime_map())

static inline std::unordered_map<const void*, std::string>& get_runtime_names() {
    static auto* m = new std::unordered_map<const void*, std::string>();
    return *m;
}
#define g_runtime_names (get_runtime_names())

static inline std::unordered_map<const void*, KernelPolicyDescriptor>& get_policy_map() {
    static auto* m = new std::unordered_map<const void*, KernelPolicyDescriptor>();
    return *m;
}
#define g_policy_map (get_policy_map())

static inline std::unordered_set<const void*>& get_cuda13_funcs() {
    static auto* s = new std::unordered_set<const void*>();
    return *s;
}
#define g_cuda13_funcs (get_cuda13_funcs())

static inline std::vector<void*>& get_dummy_stubs() {
    static auto* v = new std::vector<void*>();
    return *v;
}
#define g_dummy_stubs (get_dummy_stubs())

// Driver API: CUfunction (native) -> CUfunction (tracked)
static inline std::unordered_map<CUfunction, CUfunction>& get_driver_map() {
    static auto* m = new std::unordered_map<CUfunction, CUfunction>();
    return *m;
}
#define g_driver_map (get_driver_map())

static inline std::unordered_map<CUfunction, KernelPolicyDescriptor>& get_driver_policy_map() {
    static auto* m = new std::unordered_map<CUfunction, KernelPolicyDescriptor>();
    return *m;
}
#define g_driver_policy_map (get_driver_policy_map())

struct PendingAlloc {
    void*  ptr;
    size_t size;
    bool   is_managed;
};
static inline std::vector<PendingAlloc>& get_pending() {
    static auto* v = new std::vector<PendingAlloc>();
    return *v;
}
#define g_pending (get_pending())

struct ActiveAlloc {
    void*  ptr;
    size_t size;
    bool   preferred_cpu = false;
};
static inline std::vector<ActiveAlloc>& get_active() {
    static auto* v = new std::vector<ActiveAlloc>();
    return *v;
}
#define g_active (get_active())

static bool                    g_init_done = false;
static thread_local bool       g_in_preload = false;

// ── Dynamic Bytecode Execution Engine ───────────────────────────────────────
struct KernelBytecodePolicy {
    std::string name;
    uint16_t flags = 0;
    uint16_t num_ranges = 0;
    std::vector<uint8_t> bytecode;
};

static inline std::unordered_map<std::string, KernelBytecodePolicy>& get_metadata_registry() {
    static auto* reg = new std::unordered_map<std::string, KernelBytecodePolicy>();
    return *reg;
}
#define g_metadata_registry (get_metadata_registry())

// Normalize mangled CUDA kernel symbols across nvcc and clang++:
// 1. Strip `_ZL` internal linkage prefix -> `_Z`
// 2. Strip nvcc anonymous namespace `_ZN<N>_INTERNAL_<chars>` inside `IXadL...E` -> `IXadL_ZL`
// 3. Canonicalize Itanium substitution tokens `S[0-9A-Z]+_` -> `S_` when `IXadL` is present
static std::string normalize_kernel_symbol(const std::string& raw) {
    std::string s = raw;
    if (s.rfind("__uvm_tracked_", 0) == 0) {
        s = s.substr(14);
    }
    if (s.rfind("_ZL", 0) == 0) {
        s = "_Z" + s.substr(3);
    }
    size_t ix = s.find("IXadL_ZN");
    if (ix != std::string::npos) {
        size_t p = ix + 8; // after "IXadL_ZN"
        size_t len_val = 0;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            len_val = len_val * 10 + (s[p] - '0');
            p++;
        }
        if (len_val > 0 && p + len_val < s.size() && s.compare(p, 10, "_INTERNAL_") == 0) {
            size_t fn_pos = p + len_val;
            // Find matching 'E' closing _ZN...E before template args
            size_t fn_len = 0;
            size_t fp = fn_pos;
            while (fp < s.size() && s[fp] >= '0' && s[fp] <= '9') {
                fn_len = fn_len * 10 + (s[fp] - '0');
                fp++;
            }
            size_t close_e = fp + fn_len;
            if (close_e < s.size() && s[close_e] == 'E') {
                s = s.substr(0, ix) + "IXadL_ZL" + s.substr(fn_pos, close_e - fn_pos) + s.substr(close_e + 1);
            }
        }
    }
    if (s.find("IXadL_ZL") != std::string::npos) {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == 'S' && i + 2 < s.size()) {
                size_t j = i + 1;
                while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || (s[j] >= 'A' && s[j] <= 'Z'))) {
                    j++;
                }
                if (j > i + 1 && j < s.size() && s[j] == '_') {
                    out += "S_";
                    i = j;
                    continue;
                }
            }
            out += s[i];
        }
        s = std::move(out);
    }
    return s;
}

static void load_metadata_cache(bool force = false) {
    static bool loaded = false;
    if (loaded && !force) return;
    loaded = true;

    const char* paths[] = {
        ".uvm_metadata.cache",
        "/tmp/.uvm_metadata.cache",
        "/home/arush/data/ravi/uvm_access_tracking_1/.uvm_metadata.cache"
    };
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
                if (table_end > buf.size() || hdr->table_size < sizeof(UvmKernelMetadataHeader)) {
                    off += 4;
                    continue;
                }
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
                    std::string norm_name = normalize_kernel_symbol(kname);
                    if (norm_name != kname) {
                        g_metadata_registry[norm_name] = pol;
                    }
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
        case UvmOpcode::OP_EMIT_RANGE:
        case UvmOpcode::OP_EMIT_ARG_RANGE:
        case UvmOpcode::OP_EMIT_STRUCT_ARG_RANGE: {
            uintptr_t base = 0;
            uint64_t size = 0;
            if ((UvmOpcode)op == UvmOpcode::OP_EMIT_RANGE) {
                size = pop();
                base = (uintptr_t)pop();
            } else if ((UvmOpcode)op == UvmOpcode::OP_EMIT_ARG_RANGE) {
                uint8_t arg_idx = bc[ip++];
                size = pop();
                if (args && args[arg_idx]) {
                    base = *(uintptr_t*)args[arg_idx];
                }
            } else {
                uint8_t arg_idx = bc[ip++];
                uint8_t byte_off = bc[ip++];
                size = pop();
                if (args && args[arg_idx]) {
                    memcpy(&base, (const uint8_t*)args[arg_idx] + byte_off, sizeof(uintptr_t));
                }
            }
            if (base != 0 && size > 0) {
                if (active_allocs.empty()) {
                    out_ranges.push_back({base, (size_t)size});
                } else {
                    bool any_preferred_cpu = false;
                    for (const auto& a : active_allocs) {
                        if (a.preferred_cpu) { any_preferred_cpu = true; break; }
                    }
                    for (const auto& alloc : active_allocs) {
                        uintptr_t a_start = (uintptr_t)alloc.ptr;
                        uintptr_t a_end = a_start + alloc.size;
                        if (base >= a_start && base < a_end) {
                            if (any_preferred_cpu && !alloc.preferred_cpu) {
                                break;
                            }
                            size_t clamped = std::min((size_t)size, (size_t)(a_end - base));
                            out_ranges.push_back({base, clamped});
                            break;
                        }
                    }
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

// ── Helper: count kernel arguments from demangled device function name ───────
static int count_kernel_args(const std::string& mangled_name) {
    if (mangled_name.empty()) return -1;
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
    if (!demangled || status != 0) {
        if (demangled) free(demangled);
        return -1;
    }
    std::string s(demangled);
    free(demangled);

    size_t open_paren = s.rfind('(');
    size_t close_paren = s.rfind(')');
    if (open_paren == std::string::npos || close_paren == std::string::npos || close_paren <= open_paren) {
        return -1;
    }
    std::string params = s.substr(open_paren + 1, close_paren - open_paren - 1);
    while (!params.empty() && isspace(params.front())) params.erase(params.begin());
    while (!params.empty() && isspace(params.back())) params.pop_back();
    if (params.empty() || params == "void") return 0;

    int count = 1;
    int template_depth = 0;
    for (char c : params) {
        if (c == '<') template_depth++;
        else if (c == '>') template_depth--;
        else if (c == ',' && template_depth == 0) count++;
    }
    return count;
}

// ── Helper: scan pointer arguments against active managed allocations ────────
static inline bool safe_read_uintptr(const void* addr, uintptr_t* out) {
    if (!addr) return false;
    uintptr_t u = (uintptr_t)addr;
    if (u < 0x10000 || u > 0x00007fffffffffffull) return false;
    struct iovec local_iov = { out, sizeof(uintptr_t) };
    struct iovec remote_iov = { const_cast<void*>(addr), sizeof(uintptr_t) };
    return process_vm_readv(getpid(), &local_iov, 1, &remote_iov, 1, 0) == sizeof(uintptr_t);
}

static inline size_t get_prefetch_chunk_limit() {
    static pid_t cached_pid = 0;
    static size_t chunk_limit = 32ULL * 1024ULL * 1024ULL; // Default: 32 MB
    pid_t cur_pid = getpid();
    if (cached_pid != cur_pid) {
        cached_pid = cur_pid;
        const char* env = getenv("UVM_PREFETCH_CHUNK_MB");
        if (env) {
            long long mb = atoll(env);
            if (mb <= 0) {
                chunk_limit = 0; // 0 = full alloc
            } else {
                chunk_limit = (size_t)mb * 1024ULL * 1024ULL;
            }
        } else {
            chunk_limit = 32ULL * 1024ULL * 1024ULL;
        }
    }
    return chunk_limit;
}

static inline void add_clamped_range(
    uintptr_t cand,
    uintptr_t base,
    size_t alloc_size,
    size_t chunk_limit,
    std::vector<std::pair<uintptr_t, size_t>>& out_ranges
) {
    uintptr_t start = base;
    size_t sz = alloc_size;

    if (chunk_limit > 0) {
        start = cand & ~0x1fffffULL; // Align down to 2MB boundary
        if (start < base) start = base;
        uintptr_t alloc_end = base + alloc_size;
        size_t rem = (alloc_end > start) ? (alloc_end - start) : 0;
        sz = std::min(rem, chunk_limit);
    }
    if (sz == 0) return;

    for (auto& r : out_ranges) {
        if (r.first == start) {
            r.second = std::max(r.second, sz);
            return;
        }
        if (start >= r.first && start + sz <= r.first + r.second) {
            return;
        }
        if (r.first >= start && r.first + r.second <= start + sz) {
            r.first = start;
            r.second = sz;
            return;
        }
    }
    out_ranges.push_back({start, sz});
}

static inline void scan_pointers_against_active(
    void** args,
    int max_args,
    std::vector<std::pair<uintptr_t, size_t>>& out_ranges
) {
    if (!args) return;
    std::vector<ActiveAlloc> active_copy;
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        active_copy = g_active;
    }
    if (active_copy.empty()) return;

    size_t chunk_limit = get_prefetch_chunk_limit();

    int limit = (max_args > 0) ? max_args : 16;
    if (limit <= 0) return;
    for (int i = 0; i < limit; ++i) {
        uintptr_t arg_slot = 0;
        if (!safe_read_uintptr((const void*)(args + i), &arg_slot)) {
            break;
        }
        if (arg_slot == 0) continue;

        // Direct candidate check (in case args was pointer array directly)
        for (const auto& alloc : active_copy) {
            uintptr_t base = (uintptr_t)alloc.ptr;
            if (arg_slot >= base && arg_slot < base + alloc.size) {
                add_clamped_range(arg_slot, base, alloc.size, chunk_limit, out_ranges);
            }
        }

        // Dereferenced candidate check (standard CUDA args[i] is pointer to argument variable)
        uintptr_t cand = 0;
        if (safe_read_uintptr((const void*)arg_slot, &cand)) {
            for (const auto& alloc : active_copy) {
                uintptr_t base = (uintptr_t)alloc.ptr;
                if (cand >= base && cand < base + alloc.size) {
                    add_clamped_range(cand, base, alloc.size, chunk_limit, out_ranges);
                }
            }
        }
    }
}

// ── Proactive Prefetcher (Record & Replay Engine) ───────────────────────────
struct MemoryRange {
    uintptr_t base;
    size_t size;
};

struct KernelLaunchEvent {
    uint64_t signature;
    std::string name;
    std::vector<MemoryRange> ranges;
    cudaEvent_t completion_event = nullptr;
};

class ProactivePrefetcher {
public:
    enum State {
        STATE_DISABLED,
        STATE_SYNC,       // Synchronous prefetch directly on kernel launch stream
        STATE_RECORDING,  // Proactive prefetch: recording execution trace
        STATE_REPLAYING   // Proactive prefetch: replaying lookahead
    };

private:
    std::mutex m_mutex;
    State m_state = STATE_DISABLED;
    std::vector<KernelLaunchEvent> m_sequence;
    size_t m_current_step = 0;
    size_t m_cycle_start = 0;
    size_t m_cycle_length = 0;
    int m_horizon = 4;
    int m_evict_lag = 4;
    bool m_proactive_evict = false;
    int m_device_id = 0;
    cudaStream_t m_prefetch_stream = nullptr;
    bool m_stream_initialized = false;
    uint64_t m_total_prefetches = 0;
    uint64_t m_total_prefetched_bytes = 0;
    uint64_t m_total_evictions = 0;
    bool m_debug = false;
    pid_t m_pid = 0;

    uint64_t compute_signature(const void* func_id, const std::string& name, const std::vector<MemoryRange>& ranges) {
        uint64_t h = (uint64_t)func_id ^ 0x9e3779b97f4a7c15ULL;
        if (func_id == nullptr) {
            for (char c : name) h = (h * 131) + (uint8_t)c;
        }
        for (const auto& r : ranges) {
            h ^= (r.base + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        }
        return h;
    }

    void init_stream() {
        pid_t cur_pid = getpid();
        if (m_pid != cur_pid) {
            m_pid = cur_pid;
            m_stream_initialized = false;
            m_prefetch_stream = nullptr;
            m_sequence.clear();
            init();
            m_current_step = 0;
            m_cycle_length = 0;
        }
        if (m_stream_initialized) return;
        m_stream_initialized = true;
        if (real_cudaGetDevice) {
            real_cudaGetDevice(&m_device_id);
        }
        if (m_state != STATE_SYNC && real_cudaStreamCreateWithFlags) {
            cudaError_t s_err = real_cudaStreamCreateWithFlags(&m_prefetch_stream, 0x01 /* cudaStreamNonBlocking */);
            if (s_err != cudaSuccess) {
                m_prefetch_stream = nullptr;
                m_stream_initialized = false;
            }
        }
    }

public:
    void init() {
        const char* p_en = getenv("UVM_PREFETCH_ENABLE");
        const char* p_sync = getenv("UVM_PREFETCH_SYNC");
        const char* p_mode = getenv("UVM_PREFETCH_MODE");

        bool sync_mode = (p_sync && p_sync[0] == '1') ||
                         (p_mode && (strcasecmp(p_mode, "sync") == 0 || strcasecmp(p_mode, "synchronous") == 0));

        if (sync_mode) {
            m_state = STATE_SYNC;
        } else if (p_en && p_en[0] == '1') {
            m_state = STATE_RECORDING;
        } else {
            m_state = STATE_DISABLED;
            return;
        }

        const char* horiz_str = getenv("UVM_PREFETCH_HORIZON");
        if (horiz_str) {
            int h = atoi(horiz_str);
            if (h > 0) m_horizon = h;
        }

        const char* evict_str = getenv("UVM_PREFETCH_EVICT");
        if (evict_str && evict_str[0] == '1') {
            m_proactive_evict = true;
        }

        const char* lag_str = getenv("UVM_PREFETCH_EVICT_LAG");
        if (lag_str && atoi(lag_str) > 0) {
            m_evict_lag = atoi(lag_str);
        } else {
            m_evict_lag = m_horizon;
        }

        const char* dbg = getenv("UVM_DEBUG_PREFETCH");
        if (dbg && dbg[0] == '1') {
            m_debug = true;
        }

        size_t chunk_mb = get_prefetch_chunk_limit() / (1024 * 1024);
        if (m_state == STATE_SYNC) {
            fprintf(stderr, "[UVM_PREFETCH] Initialized in SYNCHRONOUS mode: ChunkLimit=%zu MB (prefetch directly on kernel launch stream)\n",
                    chunk_mb);
        } else {
            fprintf(stderr, "[UVM_PREFETCH] Initialized in PROACTIVE mode: Horizon=%d, EvictLag=%d, ChunkLimit=%zu MB, ProactiveEvict=%d, State=RECORDING\n",
                    m_horizon, m_evict_lag, chunk_mb, (int)m_proactive_evict);
        }
    }

    bool is_enabled() const {
        return m_state != STATE_DISABLED;
    }

    void on_launch(const void* func_id, const std::string& name, const std::vector<std::pair<uintptr_t, size_t>>& in_ranges, cudaStream_t stream = nullptr) {
        if (m_state == STATE_DISABLED) return;
        if (in_ranges.empty()) return;

        std::lock_guard<std::mutex> lock(m_mutex);
        init_stream();

        std::vector<MemoryRange> ranges;
        {
            std::lock_guard<std::recursive_mutex> lock_active(g_mutex);
            if (g_active.empty()) return;

            static const char* host_only_env = getenv("UVM_PREFETCH_HOST_ONLY");
            bool host_only = (host_only_env && host_only_env[0] == '1');
            if (!host_only && (!host_only_env || host_only_env[0] != '0')) {
                // Auto-detect: if any active allocation is tagged preferred_cpu (oversubscribed offload),
                // filter prefetching to host-resident memory only!
                for (const auto& a : g_active) {
                    if (a.preferred_cpu) {
                        host_only = true;
                        break;
                    }
                }
            }

            for (const auto& r : in_ranges) {
                if (!r.first || r.second == 0) continue;
                for (const auto& alloc : g_active) {
                    uintptr_t base = (uintptr_t)alloc.ptr;
                    if (r.first >= base && r.first < base + alloc.size) {
                        if (host_only && !alloc.preferred_cpu) {
                            // Buffer is already resident in GPU VRAM - skip redundant prefetch call
                            break;
                        }
                        bool exists = false;
                        for (const auto& existing : ranges) {
                            if (existing.base == r.first) { exists = true; break; }
                        }
                        if (!exists) {
                            uintptr_t alloc_end = base + alloc.size;
                            size_t max_len = (alloc_end > r.first) ? (alloc_end - r.first) : 0;
                            size_t clamped_sz = std::min(r.second, max_len);
                            if (clamped_sz > 0) {
                                ranges.push_back({r.first, clamped_sz});
                            }
                        }
                        break;
                    }
                }
            }
        }
        if (ranges.empty()) return;

        // ── SYNCHRONOUS MODE: Prefetch directly on the kernel's launch stream ──
        if (m_state == STATE_SYNC) {
            if (stream && real_cudaStreamIsCapturing) {
                int capture_status = 0;
                if (real_cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess && capture_status != 0) {
                    return; // Avoid corrupting stream capture graph!
                }
            }
            int dev = m_device_id;
            if (real_cudaGetDevice) {
                real_cudaGetDevice(&dev);
            }
            if (real_cudaMemPrefetchAsync) {
                for (const auto& r : ranges) {
                    if (r.base && r.size > 0) {
                        cudaError_t err = real_cudaMemPrefetchAsync((const void*)r.base, r.size, dev, stream);
                        if (err == cudaSuccess) {
                            m_total_prefetches++;
                            m_total_prefetched_bytes += r.size;
                            if (m_debug) {
                                fprintf(stderr, "[UVM_PREFETCH_SYNC] prefetch SUCCESS for %s @ %p (%zu bytes, dev=%d, stream=%p)\n",
                                        name.c_str(), (void*)r.base, r.size, dev, (void*)stream);
                            }
                        } else if (m_debug) {
                            fprintf(stderr, "[UVM_PREFETCH_SYNC] prefetch failed for %s @ %p (%zu bytes, dev=%d, stream=%p): error %d\n",
                                    name.c_str(), (void*)r.base, r.size, dev, (void*)stream, (int)err);
                        }
                    }
                }
            }
            return;
        }

        uint64_t sig = compute_signature(func_id, name, ranges);

        if (m_state == STATE_RECORDING) {
            // Check if sig has appeared in m_sequence (identifying the repeat of a loop)
            bool cycle_found = false;
            for (size_t i = 0; i < m_sequence.size(); ++i) {
                if (m_sequence[i].signature == sig) {
                    size_t candidate_len = m_sequence.size() - i;
                    static const char* min_cyc_env = getenv("UVM_MIN_CYCLE");
                    size_t min_cycle = (min_cyc_env && atoi(min_cyc_env) > 0) ? (size_t)atoi(min_cyc_env) : 8;
                    if (candidate_len >= min_cycle) {
                        m_cycle_start = i;
                        m_cycle_length = candidate_len;
                        m_state = STATE_REPLAYING;
                        m_current_step = 0;
                        cycle_found = true;
                        fprintf(stderr, "[UVM_PREFETCH] >>> Cycle detected! Repeating sequence: %zu launches (offset: %zu) <<<\n",
                                m_cycle_length, m_cycle_start);
                        fprintf(stderr, "[UVM_PREFETCH] >>> Switching to REPLAYING with Lookahead Horizon=%d <<<\n", m_horizon);

                        // Prime the lookahead pipeline for the first 'horizon' steps
                        for (int h = 0; h < m_horizon && h < (int)m_cycle_length; ++h) {
                            issue_prefetch(m_sequence[m_cycle_start + h]);
                        }
                        break;
                    }
                }
            }
            if (!cycle_found) {
                m_sequence.push_back({sig, name, ranges, nullptr});
                if (m_debug) {
                    fprintf(stderr, "[UVM_PREFETCH] [Record #%zu] %s (sig=0x%llx, %zu ranges)\n",
                            m_sequence.size() - 1, name.c_str(), (unsigned long long)sig, ranges.size());
                }
                return;
            }
        }

        if (m_state == STATE_REPLAYING && m_cycle_length > 0) {
            if (m_proactive_evict && (uintptr_t)stream > 2 && real_cudaEventRecord) {
                size_t curr_idx = m_cycle_start + m_current_step;
                auto& curr_event = m_sequence[curr_idx];
                if (!curr_event.completion_event && real_cudaEventCreateWithFlags) {
                    real_cudaEventCreateWithFlags(&curr_event.completion_event, 0x02 /* cudaEventDisableTiming */);
                }
                if (curr_event.completion_event) {
                    real_cudaEventRecord(curr_event.completion_event, stream);
                }
            }

            size_t target_idx = m_cycle_start + ((m_current_step + m_horizon) % m_cycle_length);
            const auto& target_event = m_sequence[target_idx];

            if (m_debug) {
                fprintf(stderr, "[UVM_PREFETCH] [Step #%zu] %s -> Lookahead #%zu: %s (%zu ranges)\n",
                        m_current_step, name.c_str(), target_idx - m_cycle_start, target_event.name.c_str(), target_event.ranges.size());
            }

            issue_prefetch(target_event);

            if (m_proactive_evict) {
                size_t evict_idx = m_cycle_start + ((m_current_step - m_evict_lag + m_cycle_length) % m_cycle_length);
                issue_evict(m_sequence[evict_idx]);
            }

            m_current_step = (m_current_step + 1) % m_cycle_length;
        }
    }

private:
    void issue_prefetch(const KernelLaunchEvent& ev) {
        if (!real_cudaMemPrefetchAsync) return;
        for (const auto& r : ev.ranges) {
            if (r.base && r.size > 0) {
                cudaError_t err = real_cudaMemPrefetchAsync((const void*)r.base, r.size, m_device_id, m_prefetch_stream);
                if (err == cudaSuccess) {
                    m_total_prefetches++;
                    m_total_prefetched_bytes += r.size;
                } else if (m_debug) {
                    fprintf(stderr, "[UVM_PREFETCH] prefetch failed for %p (%zu bytes): error %d\n",
                            (void*)r.base, r.size, (int)err);
                }
            }
        }
    }

    void issue_evict(const KernelLaunchEvent& ev) {
        if (!real_cudaMemPrefetchAsync) return;
        if (ev.completion_event && real_cudaStreamWaitEvent) {
            real_cudaStreamWaitEvent(m_prefetch_stream, ev.completion_event, 0);
        }
        static const char* evict_all = getenv("UVM_PREFETCH_EVICT_ALL");
        bool force_all = (evict_all && evict_all[0] == '1');

        for (const auto& r : ev.ranges) {
            if (r.base && r.size > 0) {
                bool should_evict = force_all;
                if (!should_evict) {
                    std::lock_guard<std::recursive_mutex> lock(g_mutex);
                    for (const auto& alloc : g_active) {
                        uintptr_t base = (uintptr_t)alloc.ptr;
                        if (r.base >= base && r.base < base + alloc.size) {
                            if (alloc.preferred_cpu) {
                                should_evict = true;
                            }
                            break;
                        }
                    }
                }
                if (should_evict) {
                    cudaError_t err = real_cudaMemPrefetchAsync((const void*)r.base, r.size, -1 /* cudaCpuDeviceId */, m_prefetch_stream);
                    if (err == cudaSuccess) {
                        m_total_evictions++;
                    } else if (m_debug) {
                        fprintf(stderr, "[UVM_PREFETCH] evict failed for %p (%zu bytes): error %d\n",
                                (void*)r.base, r.size, (int)err);
                    }
                }
            }
        }
    }

public:
    void print_stats() {
        if (m_state == STATE_DISABLED) return;
        if (m_state == STATE_SYNC) {
            fprintf(stderr, "\n==================== UVM Synchronous Prefetcher Summary ====================\n");
            fprintf(stderr, "Mode:                  SYNCHRONOUS (pre-kernel launch on compute stream)\n");
            fprintf(stderr, "Total Prefetches:      %llu calls\n", (unsigned long long)m_total_prefetches);
            fprintf(stderr, "Total Data Prefetched: %.2f MB\n", (double)m_total_prefetched_bytes / (1024.0 * 1024.0));
            fprintf(stderr, "============================================================================\n\n");
            return;
        }
        fprintf(stderr, "\n==================== UVM Proactive Prefetcher Summary ====================\n");
        fprintf(stderr, "State:                 %s\n", m_state == STATE_REPLAYING ? "REPLAYING" : "RECORDING");
        fprintf(stderr, "Discovered Cycle:      %zu kernel launches (start offset: %zu)\n", m_cycle_length, m_cycle_start);
        fprintf(stderr, "Lookahead Horizon:     %d steps\n", m_horizon);
        fprintf(stderr, "Total Prefetches:      %llu calls\n", (unsigned long long)m_total_prefetches);
        fprintf(stderr, "Total Data Prefetched: %.2f MB\n", (double)m_total_prefetched_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "Total Evictions:       %llu calls\n", (unsigned long long)m_total_evictions);
        fprintf(stderr, "==========================================================================\n\n");
    }
};

static inline ProactivePrefetcher& get_prefetcher() {
    static auto* p = new ProactivePrefetcher();
    return *p;
}
#define g_prefetcher (get_prefetcher())

struct StaticTemporalLaunchRecord {
    std::string kernel_name;
    uint32_t kernel_id;
    std::vector<std::pair<uintptr_t, size_t>> ranges;
    uint64_t timestamp_ns;
};

static inline std::vector<StaticTemporalLaunchRecord>& get_static_temporal_launches() {
    static auto* v = new std::vector<StaticTemporalLaunchRecord>();
    return *v;
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
    static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
    if (st_env && st_env[0] == '1') {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        std::string tracked_name = "__uvm_tracked_" + desc.kernel_name;
        uint32_t kid = static_cast<uint32_t>(std::hash<std::string>{}(tracked_name) & 0x7FFFFFFF);
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        get_static_temporal_launches().push_back({desc.kernel_name, kid, ranges, now_ns});
    }
    g_prefetcher.on_launch(desc.clean_func ? desc.clean_func : (const void*)&desc, desc.kernel_name, ranges, (cudaStream_t)stream);
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
    real___cudaLaunchKernel =
        (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "__cudaLaunchKernel");
    real___cudaLaunchKernel_ptsz =
        (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz");
    real_cudaLaunchKernelExC =
        (cudaLaunchKernelExC_t)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");

    // Open both CUDA 12 and CUDA 13 runtimes globally if present
    void* h_cudart13 = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_GLOBAL);
    void* h_cudart12 = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);

    real___cudaRegisterFunction_v12 =
        (cudaRegisterFunction_t)dlvsym(RTLD_DEFAULT, "__cudaRegisterFunction", "libcudart.so.12");
    real___cudaRegisterFunction_v13 =
        (cudaRegisterFunction_t)dlvsym(RTLD_DEFAULT, "__cudaRegisterFunction", "libcudart.so.13");
    if (!real___cudaRegisterFunction_v12 && h_cudart12)
        real___cudaRegisterFunction_v12 = (cudaRegisterFunction_t)dlsym(h_cudart12, "__cudaRegisterFunction");
    if (!real___cudaRegisterFunction_v13 && h_cudart13)
        real___cudaRegisterFunction_v13 = (cudaRegisterFunction_t)dlsym(h_cudart13, "__cudaRegisterFunction");

    real_cudaLaunchKernel_v12 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernel", "libcudart.so.12");
    real_cudaLaunchKernel_v13 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernel", "libcudart.so.13");
    if (!real_cudaLaunchKernel_v12 && h_cudart12)
        real_cudaLaunchKernel_v12 = (cudaLaunchKernel_t)dlsym(h_cudart12, "cudaLaunchKernel");
    if (!real_cudaLaunchKernel_v13 && h_cudart13)
        real_cudaLaunchKernel_v13 = (cudaLaunchKernel_t)dlsym(h_cudart13, "cudaLaunchKernel");

    real___cudaLaunchKernel_v12 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "__cudaLaunchKernel", "libcudart.so.12");
    real___cudaLaunchKernel_v13 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "__cudaLaunchKernel", "libcudart.so.13");
    if (!real___cudaLaunchKernel_v12 && h_cudart12)
        real___cudaLaunchKernel_v12 = (cudaLaunchKernel_t)dlsym(h_cudart12, "__cudaLaunchKernel");
    if (!real___cudaLaunchKernel_v13 && h_cudart13)
        real___cudaLaunchKernel_v13 = (cudaLaunchKernel_t)dlsym(h_cudart13, "__cudaLaunchKernel");

    real___cudaLaunchKernel_ptsz_v12 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "__cudaLaunchKernel_ptsz", "libcudart.so.12");
    real___cudaLaunchKernel_ptsz_v13 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "__cudaLaunchKernel_ptsz", "libcudart.so.13");
    if (!real___cudaLaunchKernel_ptsz_v12 && h_cudart12)
        real___cudaLaunchKernel_ptsz_v12 = (cudaLaunchKernel_t)dlsym(h_cudart12, "__cudaLaunchKernel_ptsz");
    if (!real___cudaLaunchKernel_ptsz_v13 && h_cudart13)
        real___cudaLaunchKernel_ptsz_v13 = (cudaLaunchKernel_t)dlsym(h_cudart13, "__cudaLaunchKernel_ptsz");

    real_cudaLaunchKernel_ptsz_v12 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernel_ptsz", "libcudart.so.12");
    real_cudaLaunchKernel_ptsz_v13 =
        (cudaLaunchKernel_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernel_ptsz", "libcudart.so.13");
    if (!real_cudaLaunchKernel_ptsz_v12 && h_cudart12)
        real_cudaLaunchKernel_ptsz_v12 = (cudaLaunchKernel_t)dlsym(h_cudart12, "cudaLaunchKernel_ptsz");
    if (!real_cudaLaunchKernel_ptsz_v13 && h_cudart13)
        real_cudaLaunchKernel_ptsz_v13 = (cudaLaunchKernel_t)dlsym(h_cudart13, "cudaLaunchKernel_ptsz");

    real_cudaLaunchKernelExC_v12 =
        (cudaLaunchKernelExC_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernelExC", "libcudart.so.12");
    real_cudaLaunchKernelExC_v13 =
        (cudaLaunchKernelExC_t)dlvsym(RTLD_DEFAULT, "cudaLaunchKernelExC", "libcudart.so.13");
    if (!real_cudaLaunchKernelExC_v12 && h_cudart12)
        real_cudaLaunchKernelExC_v12 = (cudaLaunchKernelExC_t)dlsym(h_cudart12, "cudaLaunchKernelExC");
    if (!real_cudaLaunchKernelExC_v13 && h_cudart13)
        real_cudaLaunchKernelExC_v13 = (cudaLaunchKernelExC_t)dlsym(h_cudart13, "cudaLaunchKernelExC");

    real_gle_v12 = (cudaError_t (*)(void))dlvsym(RTLD_DEFAULT, "cudaGetLastError", "libcudart.so.12");
    real_gle_v13 = (cudaError_t (*)(void))dlvsym(RTLD_DEFAULT, "cudaGetLastError", "libcudart.so.13");
    if (!real_gle_v12 && h_cudart12) real_gle_v12 = (cudaError_t (*)(void))dlsym(h_cudart12, "cudaGetLastError");
    if (!real_gle_v13 && h_cudart13) real_gle_v13 = (cudaError_t (*)(void))dlsym(h_cudart13, "cudaGetLastError");

    real_ple_v12 = (cudaError_t (*)(void))dlvsym(RTLD_DEFAULT, "cudaPeekAtLastError", "libcudart.so.12");
    real_ple_v13 = (cudaError_t (*)(void))dlvsym(RTLD_DEFAULT, "cudaPeekAtLastError", "libcudart.so.13");
    if (!real_ple_v12 && h_cudart12) real_ple_v12 = (cudaError_t (*)(void))dlsym(h_cudart12, "cudaPeekAtLastError");
    if (!real_ple_v13 && h_cudart13) real_ple_v13 = (cudaError_t (*)(void))dlsym(h_cudart13, "cudaPeekAtLastError");

    real_cudaMemAdvise_v12 =
        (cudaMemAdvise_v12_t)dlvsym(RTLD_DEFAULT, "cudaMemAdvise", "libcudart.so.12");
    real_cudaMemAdvise_v13 =
        (cudaMemAdvise_v13_t)dlvsym(RTLD_DEFAULT, "cudaMemAdvise", "libcudart.so.13");
    if (!real_cudaMemAdvise_v12 && h_cudart12)
        real_cudaMemAdvise_v12 = (cudaMemAdvise_v12_t)dlsym(h_cudart12, "cudaMemAdvise");
    if (!real_cudaMemAdvise_v13 && h_cudart13)
        real_cudaMemAdvise_v13 = (cudaMemAdvise_v13_t)dlsym(h_cudart13, "cudaMemAdvise");

    real_cudaMemPrefetchAsync =
        (cudaMemPrefetchAsync_t)dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");
    real_cudaMemAdvise =
        (cudaMemAdvise_t)dlsym(RTLD_NEXT, "cudaMemAdvise");
    real_cudaStreamCreateWithFlags =
        (cudaStreamCreateWithFlags_t)dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags");
    real_cudaGetDevice =
        (cudaGetDevice_t)dlsym(RTLD_NEXT, "cudaGetDevice");
    real_cudaStreamIsCapturing =
        (cudaStreamIsCapturing_t)dlsym(RTLD_NEXT, "cudaStreamIsCapturing");

    real_cudaEventCreateWithFlags =
        (cudaEventCreateWithFlags_t)dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");
    real_cudaEventRecord =
        (cudaEventRecord_t)dlsym(RTLD_NEXT, "cudaEventRecord");
    real_cudaStreamWaitEvent =
        (cudaStreamWaitEvent_t)dlsym(RTLD_NEXT, "cudaStreamWaitEvent");
    real_cudaEventDestroy =
        (cudaEventDestroy_t)dlsym(RTLD_NEXT, "cudaEventDestroy");

    if (!real_cudaMemPrefetchAsync || !real___cudaLaunchKernel || !real_cudaEventRecord) {
        void* h_cudart = h_cudart12 ? h_cudart12 : (h_cudart13 ? h_cudart13 : dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL));
        if (!h_cudart) h_cudart = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        if (!h_cudart) h_cudart = dlopen("libcudart.so.11.0", RTLD_LAZY | RTLD_GLOBAL);
        if (h_cudart) {
            if (!real_cudaLaunchKernel)
                real_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(h_cudart, "cudaLaunchKernel");
            if (!real___cudaLaunchKernel)
                real___cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(h_cudart, "__cudaLaunchKernel");
            if (!real___cudaLaunchKernel_ptsz)
                real___cudaLaunchKernel_ptsz = (cudaLaunchKernel_t)dlsym(h_cudart, "__cudaLaunchKernel_ptsz");
            if (!real_cudaMemPrefetchAsync)
                real_cudaMemPrefetchAsync = (cudaMemPrefetchAsync_t)dlsym(h_cudart, "cudaMemPrefetchAsync");
            if (!real_cudaMemAdvise)
                real_cudaMemAdvise = (cudaMemAdvise_t)dlsym(h_cudart, "cudaMemAdvise");
            if (!real_cudaStreamCreateWithFlags)
                real_cudaStreamCreateWithFlags = (cudaStreamCreateWithFlags_t)dlsym(h_cudart, "cudaStreamCreateWithFlags");
            if (!real_cudaGetDevice)
                real_cudaGetDevice = (cudaGetDevice_t)dlsym(h_cudart, "cudaGetDevice");
            if (!real_cudaStreamIsCapturing)
                real_cudaStreamIsCapturing = (cudaStreamIsCapturing_t)dlsym(h_cudart, "cudaStreamIsCapturing");
            if (!real_cudaEventCreateWithFlags)
                real_cudaEventCreateWithFlags = (cudaEventCreateWithFlags_t)dlsym(h_cudart, "cudaEventCreateWithFlags");
            if (!real_cudaEventRecord)
                real_cudaEventRecord = (cudaEventRecord_t)dlsym(h_cudart, "cudaEventRecord");
            if (!real_cudaStreamWaitEvent)
                real_cudaStreamWaitEvent = (cudaStreamWaitEvent_t)dlsym(h_cudart, "cudaStreamWaitEvent");
            if (!real_cudaEventDestroy)
                real_cudaEventDestroy = (cudaEventDestroy_t)dlsym(h_cudart, "cudaEventDestroy");
        }
    }

    real_cuLaunchKernel =
        (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    real_cuLaunchKernelEx =
        (cuLaunchKernelEx_t)dlsym(RTLD_NEXT, "cuLaunchKernelEx");
    real_cuModuleGetFunction =
        (cuModuleGetFunction_t)dlsym(RTLD_NEXT, "cuModuleGetFunction");
    real_cuMemAllocManaged =
        (cuMemAllocManaged_t)dlsym(RTLD_NEXT, "cuMemAllocManaged");
    real_cuMemFree =
        (cuMemFree_t)dlsym(RTLD_NEXT, "cuMemFree_v2");
    if (!real_cuMemFree)
        real_cuMemFree = (cuMemFree_t)dlsym(RTLD_NEXT, "cuMemFree");

    if (!real_cuModuleGetFunction || !real_cuLaunchKernel || !real_cuMemAllocManaged) {
        void* h_cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (!h_cuda) h_cuda = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
        if (h_cuda) {
            if (!real_cuModuleGetFunction)
                real_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(h_cuda, "cuModuleGetFunction");
            if (!real_cuLaunchKernel)
                real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(h_cuda, "cuLaunchKernel");
            if (!real_cuLaunchKernelEx)
                real_cuLaunchKernelEx = (cuLaunchKernelEx_t)dlsym(h_cuda, "cuLaunchKernelEx");
            if (!real_cuMemAllocManaged)
                real_cuMemAllocManaged = (cuMemAllocManaged_t)dlsym(h_cuda, "cuMemAllocManaged");
            if (!real_cuMemFree)
                real_cuMemFree = (cuMemFree_t)dlsym(h_cuda, "cuMemFree_v2");
            if (!real_cuMemFree)
                real_cuMemFree = (cuMemFree_t)dlsym(h_cuda, "cuMemFree");
        }
    }

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

    // Initialize proactive prefetcher
    g_prefetcher.init();

    // Spawn control thread if available (idempotent)
    const char* no_ctl = getenv("UVM_NO_CONTROL_THREAD");
    if (start_ctl_fn && (!no_ctl || no_ctl[0] != '1'))
        start_ctl_fn();
}

// ── Helper: is tracking active on host side? ─────────────────────────────────
static inline bool is_tracking_enabled_host()
{
    const char* tw = getenv("UVM_STATIC_TRIPWIRE");
    if (tw && tw[0] == '1')
        return true;

    const char* dbg = getenv("UVM_DEBUG_PREDICT");
    if (dbg && dbg[0] == '1')
        return true;

    if (g_prefetcher.is_enabled())
        return true;

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
    }
}

// ── Helper: register a new allocation ────────────────────────────────────────
static void register_alloc(void* ptr, size_t size, bool is_managed)
{
    if (!ptr || size == 0) return;
    if (g_in_preload) return;          // prevent recursion

    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        bool is_large_weight_pool = (size >= 512ULL * 1024ULL * 1024ULL);
        g_active.push_back({ptr, size, is_managed ? is_large_weight_pool : false});
        if (getenv("UVM_DEBUG_PREFETCH")) {
            fprintf(stderr, "[UVM_INTERCEPT pid=%d] register_alloc (%s): ptr=%p, size=%zu, preferred_cpu=%d (total active: %zu)\n",
                    getpid(), is_managed ? "managed" : "device", ptr, size, (int)is_large_weight_pool, g_active.size());
        }
    }

    // Skip managed pre-population if toggled off
    if (is_managed && !should_preload_managed()) return;
    // Skip device pre-population if toggled off
    if (!is_managed && !should_preload_device()) return;

    // If the page table is already up, pre-populate immediately.
    if (preload_range_fn && shadow_l1_ready()) {
        g_in_preload = true;
        preload_range_fn((uintptr_t)ptr, size);
        g_in_preload = false;
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

static inline std::vector<std::string>& get_tracked_names() {
    static auto* v = new std::vector<std::string>();
    return *v;
}
#define g_tracked_names (get_tracked_names())

// ── Intercepted __cudaRegisterFunction ───────────────────────────────────────
#if 1
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

    // Determine whether caller / hostFun belongs to a CUDA 13 module (vLLM extensions)
    Dl_info caller_info;
    bool is_v13 = false;
    if (dladdr(__builtin_return_address(0), &caller_info) && caller_info.dli_fname) {
        if (strstr(caller_info.dli_fname, "vllm") && !strstr(caller_info.dli_fname, "flashmla")) {
            is_v13 = true;
        }
    }
    if (!is_v13 && dladdr(hostFun, &caller_info) && caller_info.dli_fname) {
        if (strstr(caller_info.dli_fname, "vllm") && !strstr(caller_info.dli_fname, "flashmla")) {
            is_v13 = true;
        }
    }

    if (is_v13) {
        {
            std::lock_guard<std::mutex> lock(g_kernel_mutex);
            g_cuda13_funcs.insert((const void*)hostFun);
        }
        if (real___cudaRegisterFunction_v13) {
            real___cudaRegisterFunction_v13(fatCubinHandle, hostFun, deviceFun, deviceName,
                                           thread_limit, tid, bid, bDim, gDim, wSize);
        } else if (real___cudaRegisterFunction) {
            real___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                       thread_limit, tid, bid, bDim, gDim, wSize);
        }
    } else {
        if (real___cudaRegisterFunction_v12) {
            real___cudaRegisterFunction_v12(fatCubinHandle, hostFun, deviceFun, deviceName,
                                           thread_limit, tid, bid, bDim, gDim, wSize);
        } else if (real___cudaRegisterFunction) {
            real___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                       thread_limit, tid, bid, bDim, gDim, wSize);
        }
    }

    if (!deviceFun || !hostFun) return;

    std::string dev_name(deviceFun);
    if (getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[REGISTER] dev_name=%s, hostFun=%p, v13=%d\n", dev_name.c_str(), hostFun, (int)is_v13);
    }
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
        desc.num_args = count_kernel_args(dev_name);

        auto it = g_metadata_registry.find(dev_name);
        if (it == g_metadata_registry.end()) {
            it = g_metadata_registry.find(normalize_kernel_symbol(dev_name));
        }
        if (it != g_metadata_registry.end()) {
            desc.bytecode = it->second.bytecode;
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else if (dev_name.find("cutlass") != std::string::npos || dev_name.find("Gemm") != std::string::npos) {
            desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else {
            desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        }
        g_policy_map[(const void*)hostFun] = desc;
    }

    // Register synthetic stub with tracked device function in CUDA runtime only if GPU instrumentation or static tripwire is active
    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    static const char* static_tw_env = getenv("UVM_STATIC_TRIPWIRE");
    if ((force_gpu_env && force_gpu_env[0] == '1') || (static_tw_env && static_tw_env[0] == '1')) {
        if (is_v13 && real___cudaRegisterFunction_v13) {
            real___cudaRegisterFunction_v13(fatCubinHandle, (const char*)tracked_stub,
                                           tracked_dev_str, deviceName,
                                           thread_limit, tid, bid, bDim, gDim, wSize);
        } else if (!is_v13 && real___cudaRegisterFunction_v12) {
            real___cudaRegisterFunction_v12(fatCubinHandle, (const char*)tracked_stub,
                                           tracked_dev_str, deviceName,
                                           thread_limit, tid, bid, bDim, gDim, wSize);
        }
    }
}
#endif

// ── Common CUDA launch handler ──────────────────────────────────────────────
static inline cudaError_t handle_cuda_launch(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream,
    cudaLaunchKernel_t real_fn
) {
    ensure_init();
    process_pending();

    bool is_v13 = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        is_v13 = (g_cuda13_funcs.count(func) > 0);
    }
    if (!is_v13) {
        Dl_info dlinf;
        if (dladdr(func, &dlinf) && dlinf.dli_fname) {
            if (strstr(dlinf.dli_fname, "vllm") && !strstr(dlinf.dli_fname, "flashmla")) {
                is_v13 = true;
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                g_cuda13_funcs.insert(func);
            }
        }
    }

    cudaLaunchKernel_t primary_fn = nullptr;
    cudaLaunchKernel_t secondary_fn = nullptr;
    if (real_fn == real___cudaLaunchKernel_ptsz || real_fn == real_cudaLaunchKernel_ptsz) {
        primary_fn = is_v13 ? (real___cudaLaunchKernel_ptsz_v13 ? real___cudaLaunchKernel_ptsz_v13 : real_cudaLaunchKernel_ptsz_v13)
                            : (real___cudaLaunchKernel_ptsz_v12 ? real___cudaLaunchKernel_ptsz_v12 : real_cudaLaunchKernel_ptsz_v12);
        secondary_fn = is_v13 ? (real___cudaLaunchKernel_ptsz_v12 ? real___cudaLaunchKernel_ptsz_v12 : real_cudaLaunchKernel_ptsz_v12)
                              : (real___cudaLaunchKernel_ptsz_v13 ? real___cudaLaunchKernel_ptsz_v13 : real_cudaLaunchKernel_ptsz_v13);
    } else {
        primary_fn = is_v13 ? (real___cudaLaunchKernel_v13 ? real___cudaLaunchKernel_v13 : real_cudaLaunchKernel_v13)
                            : (real___cudaLaunchKernel_v12 ? real___cudaLaunchKernel_v12 : real_cudaLaunchKernel_v12);
        secondary_fn = is_v13 ? (real___cudaLaunchKernel_v12 ? real___cudaLaunchKernel_v12 : real_cudaLaunchKernel_v12)
                              : (real___cudaLaunchKernel_v13 ? real___cudaLaunchKernel_v13 : real_cudaLaunchKernel_v13);
    }
    if (!primary_fn) primary_fn = real_fn;

    auto do_launch = [&](const void* kfunc) -> cudaError_t {
        cudaError_t err = primary_fn ? primary_fn(kfunc, gridDim, blockDim, args, sharedMem, stream) : cudaSuccess;
        if (err == cudaErrorInvalidResourceHandle && secondary_fn && secondary_fn != primary_fn) {
            if (is_v13 && real_gle_v13) real_gle_v13();
            else if (!is_v13 && real_gle_v12) real_gle_v12();
            err = secondary_fn(kfunc, gridDim, blockDim, args, sharedMem, stream);
        }
        return err;
    };

    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        cudaError_t err = do_launch(func);
        if (err != cudaSuccess && getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[EARLY ERROR] do_launch for %p (stream=%p, v13=%d) returned %d!\n", func, stream, (int)is_v13, (int)err);
        }
        return err;
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
    if (getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[LAUNCH] func=%p, has_policy=%d, name=%s\n", func, (int)has_policy, desc.kernel_name.c_str());
    }

    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    bool force_gpu = (force_gpu_env && force_gpu_env[0] == '1');

    if (has_policy && desc.enabled && !force_gpu) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return do_launch(func);
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, args, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, stream);
            static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
            if (st_env && st_env[0] == '1' && desc.instrumented_func) {
                static bool has_device_runtime = (dlsym(RTLD_DEFAULT, "uvm_temporal_copy_entries") != nullptr);
                if (has_device_runtime) {
                    cudaError_t tw_err = do_launch(desc.instrumented_func);
                    if (tw_err == cudaSuccess) return tw_err;
                    if (real_gle_v12) real_gle_v12();
                    if (real_gle_v13) real_gle_v13();
                }
            }
            return do_launch(func);
        }
    }

    // Fallback: scan pointer arguments against active managed allocations
    std::vector<std::pair<uintptr_t, size_t>> scan_ranges;
    scan_pointers_against_active(args, desc.num_args, scan_ranges);
    if (!scan_ranges.empty()) {
        g_prefetcher.on_launch(func, desc.kernel_name.empty() ? "cudaKernel" : desc.kernel_name, scan_ranges, stream);
    }

    const void* launch_func = func;
    if (force_gpu) {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_runtime_map.find(func);
        if (it != g_runtime_map.end()) {
            launch_func = it->second;
        }
    }
    cudaError_t err = do_launch(launch_func);
    if (err != cudaSuccess) {
        if (getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[INTERCEPT ERROR] do_launch for %s (%p, launch=%p, stream=%p) returned error %d!\n",
                    desc.kernel_name.c_str(), func, launch_func, stream, (int)err);
        }
    }
    if (err != cudaSuccess && launch_func != func) {
        if (getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[DEBUG] Launching tracked %p failed (code %d), falling back to clean %p\n",
                    launch_func, (int)err, func);
        }
        err = do_launch(func);
    }
    check_cuda_err("handle_cuda_launch");
    return err;
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
    return handle_cuda_launch(func, gridDim, blockDim, args, sharedMem, stream,
                              real_cudaLaunchKernel ? real_cudaLaunchKernel : real___cudaLaunchKernel);
}

// ── Intercepted __cudaLaunchKernel (compiler default <<<>>> in CUDA 12/13) ───
extern "C" cudaError_t __cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    return handle_cuda_launch(func, gridDim, blockDim, args, sharedMem, stream,
                              real___cudaLaunchKernel ? real___cudaLaunchKernel : real_cudaLaunchKernel);
}

// ── Intercepted __cudaLaunchKernel_ptsz ───────────────────────────────────────
extern "C" cudaError_t __cudaLaunchKernel_ptsz(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    return handle_cuda_launch(func, gridDim, blockDim, args, sharedMem, stream,
                              real___cudaLaunchKernel_ptsz ? real___cudaLaunchKernel_ptsz : (real___cudaLaunchKernel ? real___cudaLaunchKernel : real_cudaLaunchKernel));
}

extern "C" cudaError_t cudaLaunchKernel_ptsz(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    static auto real_clk_ptsz = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel_ptsz");
    return handle_cuda_launch(func, gridDim, blockDim, args, sharedMem, stream,
                              real_clk_ptsz ? real_clk_ptsz : (real___cudaLaunchKernel_ptsz ? real___cudaLaunchKernel_ptsz : real_cudaLaunchKernel));
}

// ── Intercepted cudaLaunchKernelExC ──────────────────────────────────────────
extern "C" cudaError_t cudaLaunchKernelExC(
    const cudaLaunchConfig_t *config,
    const void *func,
    void **args
) {
    ensure_init();
    process_pending();

    bool is_v13 = false;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        is_v13 = (g_cuda13_funcs.count(func) > 0);
    }
    if (!is_v13) {
        Dl_info dlinf;
        if (dladdr(func, &dlinf) && dlinf.dli_fname) {
            if (strstr(dlinf.dli_fname, "vllm") && !strstr(dlinf.dli_fname, "flashmla")) {
                is_v13 = true;
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                g_cuda13_funcs.insert(func);
            }
        }
    }

    cudaLaunchKernelExC_t primary_fn = is_v13 ? (real_cudaLaunchKernelExC_v13 ? real_cudaLaunchKernelExC_v13 : real_cudaLaunchKernelExC)
                                             : (real_cudaLaunchKernelExC_v12 ? real_cudaLaunchKernelExC_v12 : real_cudaLaunchKernelExC);
    cudaLaunchKernelExC_t secondary_fn = is_v13 ? (real_cudaLaunchKernelExC_v12 ? real_cudaLaunchKernelExC_v12 : real_cudaLaunchKernelExC)
                                               : (real_cudaLaunchKernelExC_v13 ? real_cudaLaunchKernelExC_v13 : real_cudaLaunchKernelExC);

    auto do_launch_exc = [&](const void* kfunc) -> cudaError_t {
        cudaError_t err = primary_fn ? primary_fn(config, kfunc, args) : cudaSuccess;
        if (err == cudaErrorInvalidResourceHandle && secondary_fn && secondary_fn != primary_fn) {
            if (is_v13 && real_gle_v13) real_gle_v13();
            else if (!is_v13 && real_gle_v12) real_gle_v12();
            err = secondary_fn(config, kfunc, args);
        }
        return err;
    };

    if (__builtin_expect(!is_tracking_enabled_host(), 1)) {
        cudaError_t err = do_launch_exc(func);
        if (err != cudaSuccess && getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[EARLY ERROR] do_launch_exc for %p (stream=%p, v13=%d) returned %d!\n",
                    func, config ? config->stream : nullptr, (int)is_v13, (int)err);
        }
        return err;
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
    if (getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[LAUNCH] func=%p, has_policy=%d, name=%s\n", func, (int)has_policy, desc.kernel_name.c_str());
    }

    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    bool force_gpu = (force_gpu_env && force_gpu_env[0] == '1');

    if (has_policy && desc.enabled && !force_gpu) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return do_launch_exc(func);
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, args, config->gridDim.x, config->gridDim.y, config->gridDim.z,
                                   config->blockDim.x, config->blockDim.y, config->blockDim.z, config->stream);
            const void* target_f = func;
            static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
            if (st_env && st_env[0] == '1') {
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                auto it = g_runtime_map.find(func);
                if (it != g_runtime_map.end()) target_f = it->second;
            }
            return do_launch_exc(target_f);
        }
    }

    // Fallback: INSTRUMENT_GPU
    std::vector<std::pair<uintptr_t, size_t>> scan_ranges_exc;
    scan_pointers_against_active(args, desc.num_args, scan_ranges_exc);
    if (!scan_ranges_exc.empty()) {
        g_prefetcher.on_launch(func, desc.kernel_name.empty() ? "cudaKernelExC" : desc.kernel_name, scan_ranges_exc, config->stream);
    }

    const void* launch_func = func;
    if (force_gpu) {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_runtime_map.find(func);
        if (it != g_runtime_map.end()) {
            launch_func = it->second;
        }
    }
    cudaError_t err = do_launch_exc(launch_func);
    if (err != cudaSuccess) {
        if (getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[INTERCEPT ERROR] do_launch_exc for %s returned error %d!\n",
                    desc.kernel_name.c_str(), (int)err);
        }
    }
    if (err != cudaSuccess && launch_func != func) {
        err = do_launch_exc(func);
    }
    check_cuda_err("cudaLaunchKernelExC");
    return err;
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
        std::string clean_name = sname;
        if (clean_name.rfind("__uvm_tracked_", 0) == 0) {
            clean_name = clean_name.substr(14);
        }

        std::string tracked_name = "__uvm_tracked_" + clean_name;
        CUfunction tracked_func = *hfunc;
        if (sname.rfind("__uvm_tracked_", 0) != 0) {
            CUfunction tr_func = nullptr;
            CUresult tr_res = real_cuModuleGetFunction(&tr_func, hmod, tracked_name.c_str());
            if (tr_res == 0 && tr_func) {
                tracked_func = tr_func;
            }
        }

        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_driver_map[*hfunc] = tracked_func;

        KernelPolicyDescriptor desc;
        desc.kernel_name = clean_name;
        desc.enabled = true;
        desc.num_args = count_kernel_args(clean_name);

        auto it = g_metadata_registry.find(clean_name);
        if (it == g_metadata_registry.end()) {
            it = g_metadata_registry.find(normalize_kernel_symbol(clean_name));
        }
        if (it == g_metadata_registry.end()) {
            load_metadata_cache(true);
            it = g_metadata_registry.find(clean_name);
            if (it == g_metadata_registry.end()) {
                it = g_metadata_registry.find(normalize_kernel_symbol(clean_name));
            }
        }
        if (it != g_metadata_registry.end()) {
            desc.bytecode = it->second.bytecode;
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else if (clean_name.find("cutlass") != std::string::npos || clean_name.find("Gemm") != std::string::npos) {
            desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        } else {
            desc.bytecode = { (uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS, 0xFF, (uint8_t)UvmOpcode::OP_END };
            desc.strategy = TrackingStrategy::PREDICT_HOST;
        }
        g_driver_policy_map[*hfunc] = desc;

        if (getenv("UVM_DEBUG_PREDICT")) {
            fprintf(stderr, "[DEBUG PREDICT] cuModuleGetFunction name=%s, clean=%s, in_registry=%d, bc_len=%zu\n",
                    name, clean_name.c_str(), (int)(it != g_metadata_registry.end()), desc.bytecode.size());
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
            CUresult res = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                               blockDimX, blockDimY, blockDimZ,
                                               sharedMemBytes, hStream, kernelParams, extra);
            if (res != 0) {
                fprintf(stderr, "[EARLY ERROR] real_cuLaunchKernel for %p (hStream=%p) returned %d!\n", f, hStream, (int)res);
            }
            return res;
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
    if (getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[DEBUG PREDICT] cuLaunchKernel f=%p, has_policy=%d, kernel=%s, kparams=%p, extra=%p\n",
                f, has_policy, has_policy ? desc.kernel_name.c_str() : "none", kernelParams, extra);
    }

    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    bool force_gpu = (force_gpu_env && force_gpu_env[0] == '1');

    if (has_policy && desc.enabled && !force_gpu) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cuLaunchKernel ? real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                                             blockDimX, blockDimY, blockDimZ,
                                                             sharedMemBytes, hStream, kernelParams, extra) : 0;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, kernelParams, gridDimX, gridDimY, gridDimZ,
                                   blockDimX, blockDimY, blockDimZ, (void*)hStream);
            CUfunction target_f = f;
            static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
            if (st_env && st_env[0] == '1') {
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                auto it = g_driver_map.find(f);
                if (it != g_driver_map.end()) target_f = it->second;
            }
            return real_cuLaunchKernel ? real_cuLaunchKernel(target_f, gridDimX, gridDimY, gridDimZ,
                                                             blockDimX, blockDimY, blockDimZ,
                                                             sharedMemBytes, hStream, kernelParams, extra) : 0;
        }
    }

    // Fallback: INSTRUMENT_GPU or uninstrumented
    std::vector<std::pair<uintptr_t, size_t>> scan_ranges_cu;
    scan_pointers_against_active(kernelParams, desc.num_args, scan_ranges_cu);
    if (!scan_ranges_cu.empty()) {
        g_prefetcher.on_launch(f, desc.kernel_name.empty() ? "cuLaunchKernel" : desc.kernel_name, scan_ranges_cu, (cudaStream_t)hStream);
    }

    CUfunction launch_f = f;
    if (force_gpu) {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_map.find(f);
        if (it != g_driver_map.end()) {
            launch_f = it->second;
        }
    }
    if (real_cuLaunchKernel) {
        CUresult res = real_cuLaunchKernel(launch_f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, hStream, kernelParams, extra);
        if (res != 0) {
            fprintf(stderr, "[INTERCEPT ERROR] real_cuLaunchKernel for %s (hStream=%p) returned error %d!\n",
                    desc.kernel_name.c_str(), hStream, (int)res);
        }
        return res;
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
            CUresult res = real_cuLaunchKernelEx(config, f, kernelParams, extra);
            if (res != 0) {
                fprintf(stderr, "[EARLY ERROR] real_cuLaunchKernelEx for %p (hStream=%p) returned %d!\n",
                        f, config ? config->hStream : nullptr, (int)res);
            }
            return res;
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

    static const char* force_gpu_env = getenv("UVM_FORCE_GPU_INSTRUMENTATION");
    bool force_gpu = (force_gpu_env && force_gpu_env[0] == '1');

    if (has_policy && desc.enabled && !force_gpu) {
        if (desc.strategy == TrackingStrategy::SKIP_TRACKING) {
            return real_cuLaunchKernelEx ? real_cuLaunchKernelEx(config, f, kernelParams, extra) : 0;
        }
        if ((desc.strategy == TrackingStrategy::PREDICT_HOST || desc.strategy == TrackingStrategy::AUTO_DETECT) &&
            (!desc.bytecode.empty() || desc.predictor_fn)) {
            predict_and_mark_ranges(desc, kernelParams, config->gridDimX, config->gridDimY, config->gridDimZ,
                                   config->blockDimX, config->blockDimY, config->blockDimZ, (void*)config->hStream);
            CUfunction target_f = f;
            static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
            if (st_env && st_env[0] == '1') {
                std::lock_guard<std::mutex> lock(g_kernel_mutex);
                auto it = g_driver_map.find(f);
                if (it != g_driver_map.end()) target_f = it->second;
            }
            return real_cuLaunchKernelEx ? real_cuLaunchKernelEx(config, target_f, kernelParams, extra) : 0;
        }
    }

    // Fallback: INSTRUMENT_GPU or uninstrumented
    std::vector<std::pair<uintptr_t, size_t>> scan_ranges_cuex;
    scan_pointers_against_active(kernelParams, desc.num_args, scan_ranges_cuex);
    if (!scan_ranges_cuex.empty()) {
        g_prefetcher.on_launch(f, desc.kernel_name.empty() ? "cuLaunchKernelEx" : desc.kernel_name, scan_ranges_cuex, (cudaStream_t)config->hStream);
    }

    CUfunction launch_f = f;
    if (force_gpu) {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_driver_map.find(f);
        if (it != g_driver_map.end()) {
            launch_f = it->second;
        }
    }
    if (real_cuLaunchKernelEx) {
        CUresult res = real_cuLaunchKernelEx(config, launch_f, kernelParams, extra);
        if (res != 0) {
            fprintf(stderr, "[INTERCEPT ERROR] real_cuLaunchKernelEx for %s (hStream=%p) returned error %d!\n",
                    desc.kernel_name.c_str(), config ? config->hStream : nullptr, (int)res);
        }
        return res;
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
    check_cuda_err("cudaMallocManaged");
    return err;
}

// ── Intercepted cudaMalloc ───────────────────────────────────────────────────
extern "C" cudaError_t cudaMalloc(void** ptr, size_t size)
{
    ensure_init();
    static const char* force_managed = getenv("UVM_FORCE_MANAGED");
    if (force_managed && force_managed[0] == '1' && real_cudaMallocManaged) {
        cudaError_t err = real_cudaMallocManaged(ptr, size, 1 /* cudaMemAttachGlobal */);
        if (err == cudaSuccess) {
            static const char* pref_cpu = getenv("UVM_FORCE_PREFER_CPU");
            bool set_cpu = (pref_cpu && pref_cpu[0] == '1');
            if (set_cpu && real_cudaMemAdvise) {
                real_cudaMemAdvise(*ptr, size, 3 /* cudaMemAdviseSetPreferredLocation */, -1 /* cudaCpuDeviceId */);
            }
            register_alloc(*ptr, size, true);
            if (set_cpu) {
                std::lock_guard<std::recursive_mutex> lock(g_mutex);
                for (auto& a : g_active) {
                    if (a.ptr == *ptr) a.preferred_cpu = true;
                }
            }
        }
        check_cuda_err("cudaMalloc(forced_managed)");
        return err;
    }
    cudaError_t err = real_cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        register_alloc(*ptr, size, false);
    }
    check_cuda_err("cudaMalloc");
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
    cudaError_t err = real_cudaFree(ptr);
    check_cuda_err("cudaFree");
    return err;
}

// ── Intercepted cuMemAllocManaged ────────────────────────────────────────────
extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    ensure_init();
    if (!real_cuMemAllocManaged) return (CUresult)1;
    CUresult res = real_cuMemAllocManaged(dptr, bytesize, flags);
    if (res == 0 && dptr && *dptr) {
        register_alloc((void*)(uintptr_t)*dptr, bytesize, true);
    }
    if (getenv("UVM_DEBUG_PREFETCH")) {
        fprintf(stderr, "[INTERCEPT HIT] cuMemAllocManaged: res=%d, dptr=%p, bytesize=%zu\n",
                (int)res, dptr ? (void*)*dptr : nullptr, bytesize);
    }
    check_cuda_err("cuMemAllocManaged");
    return res;
}

// ── Intercepted cuMemFree / cuMemFree_v2 ─────────────────────────────────────
extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    ensure_init();
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        for (auto it = g_active.begin(); it != g_active.end(); ) {
            if ((uintptr_t)it->ptr == (uintptr_t)dptr) {
                if (drop_range_fn)
                    drop_range_fn((uintptr_t)it->ptr, it->size);
                it = g_active.erase(it);
            } else {
                ++it;
            }
        }
    }
    CUresult res = real_cuMemFree ? real_cuMemFree(dptr) : 0;
    check_cuda_err("cuMemFree_v2");
    return res;
}

extern "C" CUresult cuMemFree(CUdeviceptr dptr)
{
    return cuMemFree_v2(dptr);
}

// ── Intercepted cudaMemAdvise & cuMemAdvise ──────────────────────────────────
extern "C" cudaError_t cudaMemAdvise(const void *devPtr, size_t count, int advice, uint64_t loc_or_dev) {
    ensure_init();

    Dl_info caller_info;
    bool is_v13 = false;
    if (dladdr(__builtin_return_address(0), &caller_info) && caller_info.dli_fname) {
        if (strstr(caller_info.dli_fname, "vllm") && !strstr(caller_info.dli_fname, "flashmla")) {
            is_v13 = true;
        }
    }

    bool is_cpu = false;
    if (is_v13) {
        uint32_t loc_type = (uint32_t)(loc_or_dev & 0xFFFFFFFF);
        if (advice == 3 /* cudaMemAdviseSetPreferredLocation */ && loc_type == 1 /* cudaMemLocationTypeHost */) {
            is_cpu = true;
        }
    } else {
        if (advice == 3 && (int)loc_or_dev == -1 /* cudaCpuDeviceId */) {
            is_cpu = true;
        }
    }

    if (is_cpu) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        uintptr_t target = (uintptr_t)devPtr;
        for (auto& alloc : g_active) {
            uintptr_t base = (uintptr_t)alloc.ptr;
            if (target >= base && target < base + alloc.size) {
                alloc.preferred_cpu = true;
                if (getenv("UVM_DEBUG_PREFETCH")) {
                    fprintf(stderr, "[UVM_INTERCEPT] Marked allocation %p (size=%zu) preferred_cpu=true\n", alloc.ptr, alloc.size);
                }
            }
        }
    }

    cudaError_t err = cudaSuccess;
    if (is_v13 && real_cudaMemAdvise_v13) {
        err = real_cudaMemAdvise_v13(devPtr, count, advice, loc_or_dev);
    } else if (real_cudaMemAdvise_v12) {
        err = real_cudaMemAdvise_v12(devPtr, count, advice, (int)loc_or_dev);
    } else if (real_cudaMemAdvise) {
        err = real_cudaMemAdvise(devPtr, count, advice, (int)loc_or_dev);
    }
    check_cuda_err("cudaMemAdvise");
    return err;
}

extern "C" CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, int advice, int device) {
    ensure_init();
    if (advice == 3 /* CU_MEM_ADVISE_SET_PREFERRED_LOCATION */ && device == -1 /* CU_DEVICE_CPU */) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        uintptr_t target = (uintptr_t)devPtr;
        for (auto& alloc : g_active) {
            uintptr_t base = (uintptr_t)alloc.ptr;
            if (target >= base && target < base + alloc.size) {
                alloc.preferred_cpu = true;
                if (getenv("UVM_DEBUG_PREFETCH")) {
                    fprintf(stderr, "[UVM_INTERCEPT] Marked cu allocation %p (size=%zu) preferred_cpu=true\n", alloc.ptr, alloc.size);
                }
            }
        }
    }
    CUresult res = real_cuMemAdvise ? real_cuMemAdvise(devPtr, count, advice, device) : (CUresult)0;
    check_cuda_err("cuMemAdvise");
    return res;
}

// ── Intercepted cudaDeviceSynchronize ────────────────────────────────────────
extern "C" cudaError_t cudaDeviceSynchronize(void)
{
    ensure_init();
    process_pending();
    return real_cudaDeviceSynchronize();
}

// ── Intercepted cudaGetLastError / cudaPeekAtLastError ───────────────────────
extern "C" cudaError_t cudaGetLastError(void) {
    ensure_init();
    cudaError_t err12 = real_gle_v12 ? real_gle_v12() : cudaSuccess;
    cudaError_t err13 = real_gle_v13 ? real_gle_v13() : cudaSuccess;
    cudaError_t err = (err12 != cudaSuccess) ? err12 : err13;
    if (err != cudaSuccess && getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[CUDA_ERROR_INTERCEPT] cudaGetLastError returned %d!\n", (int)err);
    }
    return err;
}

extern "C" cudaError_t cudaPeekAtLastError(void) {
    ensure_init();
    cudaError_t err12 = real_ple_v12 ? real_ple_v12() : cudaSuccess;
    cudaError_t err13 = real_ple_v13 ? real_ple_v13() : cudaSuccess;
    cudaError_t err = (err12 != cudaSuccess) ? err12 : err13;
    if (err != cudaSuccess && getenv("UVM_DEBUG_PREDICT")) {
        fprintf(stderr, "[CUDA_ERROR_INTERCEPT] cudaPeekAtLastError returned %d!\n", (int)err);
    }
    return err;
}

// ── Intercepted cuBLAS / cuBLASLt API (Host-Side Prediction for Closed-Source BLAS) ─
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef void* cublasHandle_t;
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;
typedef void* cublasLtMatmulAlgo_t;

typedef enum {
    CUDA_R_16F  = 2,
    CUDA_C_16F  = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F  = 0,
    CUDA_C_32F  = 4,
    CUDA_R_64F  = 1,
    CUDA_C_64F  = 5,
    CUDA_R_4I   = 16,
    CUDA_C_4I   = 17,
    CUDA_R_4U   = 18,
    CUDA_C_4U   = 19,
    CUDA_R_8I   = 3,
    CUDA_C_8I   = 7,
    CUDA_R_8U   = 8,
    CUDA_C_8U   = 9,
    CUDA_R_16I  = 20,
    CUDA_C_16I  = 21,
    CUDA_R_16U  = 22,
    CUDA_C_16U  = 23,
    CUDA_R_32I  = 10,
    CUDA_C_32I  = 11,
    CUDA_R_32U  = 12,
    CUDA_C_32U  = 13,
    CUDA_R_64I  = 24,
    CUDA_C_64I  = 25,
    CUDA_R_64U  = 26,
    CUDA_C_64U  = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29
} cudaDataType;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73
} cublasComputeType_t;

typedef int cublasGemmAlgo_t;

typedef enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7
} cublasLtMatrixLayoutAttribute_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_TRANSC = 5,
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13
} cublasLtMatmulDescAttributes_t;

static inline size_t sizeof_cuda_datatype(int dtype) {
    switch (dtype) {
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_16I:
        case CUDA_R_16U:
            return 2;
        case CUDA_C_16F:
        case CUDA_C_16BF:
        case CUDA_C_16I:
        case CUDA_C_16U:
            return 4;
        case CUDA_R_32F:
        case CUDA_R_32I:
        case CUDA_R_32U:
            return 4;
        case CUDA_C_32F:
        case CUDA_C_32I:
        case CUDA_C_32U:
            return 8;
        case CUDA_R_64F:
        case CUDA_R_64I:
        case CUDA_R_64U:
            return 8;
        case CUDA_C_64F:
        case CUDA_C_64I:
        case CUDA_C_64U:
            return 16;
        case CUDA_R_8I:
        case CUDA_R_8U:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
            return 1;
        case CUDA_C_8I:
        case CUDA_C_8U:
            return 2;
        case CUDA_R_4I:
        case CUDA_R_4U:
            return 1;
        default:
            return 2;
    }
}

// Function pointer signatures
typedef cublasStatus_t (*cublasGetStream_v2_t)(cublasHandle_t, cudaStream_t*);
typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const void*, const void*, cudaDataType, int, const void*, cudaDataType, int,
                                        const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
typedef cublasStatus_t (*cublasGemmStridedBatchedEx_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                                       const void*, const void*, cudaDataType, int, long long int,
                                                       const void*, cudaDataType, int, long long int, const void*,
                                                       void*, cudaDataType, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
typedef cublasStatus_t (*cublasLtMatmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*,
                                           const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
                                           const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
                                           const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutGetAttribute_t)(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void*, size_t, size_t*);
typedef cublasStatus_t (*cublasLtMatmulDescGetAttribute_t)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void*, size_t, size_t*);
typedef cublasStatus_t (*cublasSgemm_v2_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                          const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDgemm_v2_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                          const double*, const double*, int, const double*, int, const double*, double*, int);
typedef cublasStatus_t (*cublasSgemv_v2_t)(cublasHandle_t, cublasOperation_t, int, int,
                                          const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDgemv_v2_t)(cublasHandle_t, cublasOperation_t, int, int,
                                          const double*, const double*, int, const double*, int, const double*, double*, int);

static cublasGetStream_v2_t                 real_cublasGetStream_v2               = nullptr;
static cublasGemmEx_t                       real_cublasGemmEx                     = nullptr;
static cublasGemmStridedBatchedEx_t         real_cublasGemmStridedBatchedEx       = nullptr;
static cublasLtMatmul_t                     real_cublasLtMatmul                   = nullptr;
static cublasLtMatrixLayoutGetAttribute_t   real_cublasLtMatrixLayoutGetAttribute = nullptr;
static cublasLtMatmulDescGetAttribute_t     real_cublasLtMatmulDescGetAttribute   = nullptr;
static cublasSgemm_v2_t                     real_cublasSgemm_v2                   = nullptr;
static cublasDgemm_v2_t                     real_cublasDgemm_v2                   = nullptr;
static cublasSgemv_v2_t                     real_cublasSgemv_v2                   = nullptr;
static cublasDgemv_v2_t                     real_cublasDgemv_v2                   = nullptr;

static void* get_cublas_symbol(const char* name) {
    void* sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        static void* h_cublas = nullptr;
        if (!h_cublas) {
            int rt_ver = 0;
            auto rt_ver_fn = (cudaError_t (*)(int*))dlsym(RTLD_DEFAULT, "cudaRuntimeGetVersion");
            if (rt_ver_fn) rt_ver_fn(&rt_ver);

            const char* cublas_libs_12[] = {
                "libcublas.so.12",
                "libcublas.so",
                "libcublas.so.13",
                nullptr
            };
            const char* cublas_libs_13[] = {
                "libcublas.so.13",
                "libcublas.so",
                "libcublas.so.12",
                nullptr
            };
            const char** cublas_libs = (rt_ver >= 13000) ? cublas_libs_13 : cublas_libs_12;
            for (int i = 0; cublas_libs[i]; ++i) {
                h_cublas = dlopen(cublas_libs[i], RTLD_LAZY | RTLD_GLOBAL);
                if (h_cublas) break;
            }
        }
        if (h_cublas) sym = dlsym(h_cublas, name);
    }
    return sym;
}

static void* get_cublaslt_symbol(const char* name) {
    void* sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        static void* h_cublaslt = nullptr;
        if (!h_cublaslt) {
            int rt_ver = 0;
            auto rt_ver_fn = (cudaError_t (*)(int*))dlsym(RTLD_DEFAULT, "cudaRuntimeGetVersion");
            if (rt_ver_fn) rt_ver_fn(&rt_ver);

            const char* cublaslt_libs_12[] = {
                "libcublasLt.so.12",
                "libcublasLt.so",
                "libcublasLt.so.13",
                nullptr
            };
            const char* cublaslt_libs_13[] = {
                "libcublasLt.so.13",
                "libcublasLt.so",
                "libcublasLt.so.12",
                nullptr
            };
            const char** cublaslt_libs = (rt_ver >= 13000) ? cublaslt_libs_13 : cublaslt_libs_12;
            for (int i = 0; cublaslt_libs[i]; ++i) {
                h_cublaslt = dlopen(cublaslt_libs[i], RTLD_LAZY | RTLD_GLOBAL);
                if (h_cublaslt) break;
            }
        }
        if (h_cublaslt) sym = dlsym(h_cublaslt, name);
    }
    return sym;
}

static inline void cublas_record_range(
    const char* op_name,
    const char* tensor_name,
    const void* ptr,
    size_t size,
    cudaStream_t stream
) {
    if (!ptr || size == 0) return;
    const char* dbg = getenv("UVM_DEBUG_PREDICT");
    if (dbg && dbg[0] == '1') {
        fprintf(stderr, "[DEBUG PREDICT] %s: %s at %p, size=%zu bytes (%zu pages)\n",
                op_name, tensor_name, ptr, size, (size + 4095) / 4096);
    }
    if (mark_predicted_range_fn) {
        mark_predicted_range_fn((uintptr_t)ptr, size, stream);
    }
}

static inline size_t get_cublaslt_matrix_size(cublasLtMatrixLayout_t layout) {
    if (!layout || !real_cublasLtMatrixLayoutGetAttribute) return 0;

    uint32_t dtype = CUDA_R_16F;
    uint64_t rows = 0;
    uint64_t cols = 0;
    int64_t ld = 0;
    int32_t batch_count = 1;
    int64_t batch_stride = 0;
    int32_t order = 0;

    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtype, sizeof(dtype), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride, sizeof(batch_stride), nullptr);
    real_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order), nullptr);

    size_t elem_size = sizeof_cuda_datatype(dtype);
    if (rows == 0 || cols == 0) return 0;

    size_t num_elements = 0;
    if (order == 1 /* CUBLASLT_ORDER_ROW */) {
        num_elements = (ld > 0) ? ((rows > 1 ? (rows - 1) * (size_t)ld : 0) + cols) : (rows * cols);
    } else {
        num_elements = (ld > 0) ? ((cols > 1 ? (cols - 1) * (size_t)ld : 0) + rows) : (rows * cols);
    }

    size_t total_elements = num_elements;
    if (batch_count > 1) {
        if (batch_stride > 0) {
            total_elements = (size_t)(batch_count - 1) * batch_stride + num_elements;
        } else {
            total_elements = (size_t)batch_count * num_elements;
        }
    }
    return total_elements * elem_size;
}

static inline void get_gemm_dimensions(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    size_t elem_size_a,
    size_t elem_size_b,
    size_t elem_size_c,
    size_t* out_size_a,
    size_t* out_size_b,
    size_t* out_size_c
) {
    if (m <= 0 || n <= 0 || k <= 0) {
        if (out_size_a) *out_size_a = 0;
        if (out_size_b) *out_size_b = 0;
        if (out_size_c) *out_size_c = 0;
        return;
    }

    size_t actual_lda = (lda > 0) ? lda : ((transa == CUBLAS_OP_N) ? m : k);
    size_t actual_ldb = (ldb > 0) ? ldb : ((transb == CUBLAS_OP_N) ? k : n);
    size_t actual_ldc = (ldc > 0) ? ldc : m;

    size_t elems_a = 0;
    if (transa == CUBLAS_OP_N) {
        elems_a = (k > 1 ? (size_t)(k - 1) * actual_lda : 0) + m;
    } else {
        elems_a = (m > 1 ? (size_t)(m - 1) * actual_lda : 0) + k;
    }

    size_t elems_b = 0;
    if (transb == CUBLAS_OP_N) {
        elems_b = (n > 1 ? (size_t)(n - 1) * actual_ldb : 0) + k;
    } else {
        elems_b = (k > 1 ? (size_t)(k - 1) * actual_ldb : 0) + n;
    }

    size_t elems_c = (n > 1 ? (size_t)(n - 1) * actual_ldc : 0) + m;

    if (out_size_a) *out_size_a = elems_a * elem_size_a;
    if (out_size_b) *out_size_b = elems_b * elem_size_b;
    if (out_size_c) *out_size_c = elems_c * elem_size_c;
}

static inline void get_gemv_dimensions(
    cublasOperation_t trans,
    int m, int n,
    int lda, int incx, int incy,
    size_t elem_size,
    size_t* out_size_a,
    size_t* out_size_x,
    size_t* out_size_y
) {
    if (m <= 0 || n <= 0) {
        if (out_size_a) *out_size_a = 0;
        if (out_size_x) *out_size_x = 0;
        if (out_size_y) *out_size_y = 0;
        return;
    }

    size_t actual_lda = lda > 0 ? lda : m;
    size_t elems_a = (n > 1 ? (size_t)(n - 1) * actual_lda : 0) + m;

    size_t stride_x = (incx != 0) ? (size_t)std::abs(incx) : 1;
    size_t stride_y = (incy != 0) ? (size_t)std::abs(incy) : 1;

    size_t len_x = (trans == CUBLAS_OP_N) ? n : m;
    size_t len_y = (trans == CUBLAS_OP_N) ? m : n;

    size_t elems_x = (len_x > 1 ? (size_t)(len_x - 1) * stride_x : 0) + 1;
    size_t elems_y = (len_y > 1 ? (size_t)(len_y - 1) * stride_y : 0) + 1;

    if (out_size_a) *out_size_a = elems_a * elem_size;
    if (out_size_x) *out_size_x = elems_x * elem_size;
    if (out_size_y) *out_size_y = elems_y * elem_size;
}

// ── Intercepted: cublasLtMatmul ──────────────────────────────────────────────
extern "C" cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream
) {
    ensure_init();
    process_pending();

    if (!real_cublasLtMatmul) {
        real_cublasLtMatmul = (cublasLtMatmul_t)get_cublaslt_symbol("cublasLtMatmul");
    }
    if (!real_cublasLtMatrixLayoutGetAttribute) {
        real_cublasLtMatrixLayoutGetAttribute = (cublasLtMatrixLayoutGetAttribute_t)get_cublaslt_symbol("cublasLtMatrixLayoutGetAttribute");
    }
    if (!real_cublasLtMatmulDescGetAttribute) {
        real_cublasLtMatmulDescGetAttribute = (cublasLtMatmulDescGetAttribute_t)get_cublaslt_symbol("cublasLtMatmulDescGetAttribute");
    }

    if (is_tracking_enabled_host()) {
        std::vector<std::pair<uintptr_t, size_t>> cublas_ranges;
        if (A && Adesc) {
            size_t size_a = get_cublaslt_matrix_size(Adesc);
            cublas_record_range("cublasLtMatmul", "A", A, size_a, stream);
            if (size_a > 0) cublas_ranges.push_back({(uintptr_t)A, size_a});
        }
        if (B && Bdesc) {
            size_t size_b = get_cublaslt_matrix_size(Bdesc);
            cublas_record_range("cublasLtMatmul", "B", B, size_b, stream);
            if (size_b > 0) cublas_ranges.push_back({(uintptr_t)B, size_b});
        }
        if (C && Cdesc && C != D) {
            size_t size_c = get_cublaslt_matrix_size(Cdesc);
            cublas_record_range("cublasLtMatmul", "C", C, size_c, stream);
            if (size_c > 0) cublas_ranges.push_back({(uintptr_t)C, size_c});
        }
        if (D && Ddesc) {
            size_t size_d = get_cublaslt_matrix_size(Ddesc);
            cublas_record_range("cublasLtMatmul", "D", D, size_d, stream);
            if (size_d > 0) cublas_ranges.push_back({(uintptr_t)D, size_d});
        }
        if (workspace && workspaceSizeInBytes > 0) {
            cublas_record_range("cublasLtMatmul", "Workspace", workspace, workspaceSizeInBytes, stream);
        }
        if (real_cublasLtMatmulDescGetAttribute && computeDesc && Ddesc) {
            void* bias_ptr = nullptr;
            real_cublasLtMatmulDescGetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr), nullptr);
            if (bias_ptr) {
                uint64_t rows_d = 0;
                uint32_t dtype_d = CUDA_R_16F;
                if (real_cublasLtMatrixLayoutGetAttribute) {
                    real_cublasLtMatrixLayoutGetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows_d, sizeof(rows_d), nullptr);
                    real_cublasLtMatrixLayoutGetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtype_d, sizeof(dtype_d), nullptr);
                }
                if (rows_d > 0) {
                    size_t bias_size = rows_d * sizeof_cuda_datatype(dtype_d);
                    cublas_record_range("cublasLtMatmul", "Bias", bias_ptr, bias_size, stream);
                    cublas_ranges.push_back({(uintptr_t)bias_ptr, bias_size});
                }
            }
        }
        if (!cublas_ranges.empty()) {
            g_prefetcher.on_launch((const void*)computeDesc, "cublasLtMatmul", cublas_ranges, stream);
        }
    }

    if (real_cublasLtMatmul) {
        cublasStatus_t st = real_cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[INTERCEPT ERROR] real_cublasLtMatmul returned status %d!\n", (int)st);
        }
        return st;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Intercepted: cublasGemmEx ────────────────────────────────────────────────
extern "C" cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* A,
    cudaDataType Atype,
    int lda,
    const void* B,
    cudaDataType Btype,
    int ldb,
    const void* beta,
    void* C,
    cudaDataType Ctype,
    int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
) {
    ensure_init();
    process_pending();

    if (!real_cublasGemmEx) {
        real_cublasGemmEx = (cublasGemmEx_t)get_cublas_symbol("cublasGemmEx");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_b = 0, size_c = 0;
        get_gemm_dimensions(transa, transb, m, n, k, lda, ldb, ldc,
                            sizeof_cuda_datatype(Atype),
                            sizeof_cuda_datatype(Btype),
                            sizeof_cuda_datatype(Ctype),
                            &size_a, &size_b, &size_c);
        cublas_record_range("cublasGemmEx", "A", A, size_a, stream);
        cublas_record_range("cublasGemmEx", "B", B, size_b, stream);
        cublas_record_range("cublasGemmEx", "C", C, size_c, stream);

        std::vector<std::pair<uintptr_t, size_t>> gemm_ranges;
        if (A && size_a > 0) gemm_ranges.push_back({(uintptr_t)A, size_a});
        if (B && size_b > 0) gemm_ranges.push_back({(uintptr_t)B, size_b});
        if (C && size_c > 0) gemm_ranges.push_back({(uintptr_t)C, size_c});
        if (!gemm_ranges.empty()) {
            g_prefetcher.on_launch((const void*)real_cublasGemmEx, "cublasGemmEx", gemm_ranges, stream);
        }
    }

    if (real_cublasGemmEx) {
        return real_cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Intercepted: cublasGemmStridedBatchedEx ──────────────────────────────────
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    const void* B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    const void* beta,
    void* C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
) {
    ensure_init();
    process_pending();

    if (!real_cublasGemmStridedBatchedEx) {
        real_cublasGemmStridedBatchedEx = (cublasGemmStridedBatchedEx_t)get_cublas_symbol("cublasGemmStridedBatchedEx");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_b = 0, size_c = 0;
        get_gemm_dimensions(transa, transb, m, n, k, lda, ldb, ldc,
                            sizeof_cuda_datatype(Atype),
                            sizeof_cuda_datatype(Btype),
                            sizeof_cuda_datatype(Ctype),
                            &size_a, &size_b, &size_c);
        if (batchCount > 1) {
            if (strideA != 0) size_a = (size_t)(batchCount - 1) * std::abs(strideA) * sizeof_cuda_datatype(Atype) + size_a;
            else size_a *= batchCount;
            if (strideB != 0) size_b = (size_t)(batchCount - 1) * std::abs(strideB) * sizeof_cuda_datatype(Btype) + size_b;
            else size_b *= batchCount;
            if (strideC != 0) size_c = (size_t)(batchCount - 1) * std::abs(strideC) * sizeof_cuda_datatype(Ctype) + size_c;
            else size_c *= batchCount;
        }
        cublas_record_range("cublasGemmStridedBatchedEx", "A", A, size_a, stream);
        cublas_record_range("cublasGemmStridedBatchedEx", "B", B, size_b, stream);
        cublas_record_range("cublasGemmStridedBatchedEx", "C", C, size_c, stream);
    }

    if (real_cublasGemmStridedBatchedEx) {
        return real_cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Intercepted: cublasSgemm_v2 / cublasSgemm ────────────────────────────────
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

    if (!real_cublasSgemm_v2) {
        real_cublasSgemm_v2 = (cublasSgemm_v2_t)get_cublas_symbol("cublasSgemm_v2");
        if (!real_cublasSgemm_v2) real_cublasSgemm_v2 = (cublasSgemm_v2_t)get_cublas_symbol("cublasSgemm");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_b = 0, size_c = 0;
        get_gemm_dimensions(transa, transb, m, n, k, lda, ldb, ldc,
                            sizeof(float), sizeof(float), sizeof(float),
                            &size_a, &size_b, &size_c);
        cublas_record_range("cublasSgemm_v2", "A", A, size_a, stream);
        cublas_record_range("cublasSgemm_v2", "B", B, size_b, stream);
        cublas_record_range("cublasSgemm_v2", "C", C, size_c, stream);
    }

    if (real_cublasSgemm_v2) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
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

// ── Intercepted: cublasDgemm_v2 / cublasDgemm ────────────────────────────────
extern "C" cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc
) {
    ensure_init();
    process_pending();

    if (!real_cublasDgemm_v2) {
        real_cublasDgemm_v2 = (cublasDgemm_v2_t)get_cublas_symbol("cublasDgemm_v2");
        if (!real_cublasDgemm_v2) real_cublasDgemm_v2 = (cublasDgemm_v2_t)get_cublas_symbol("cublasDgemm");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_b = 0, size_c = 0;
        get_gemm_dimensions(transa, transb, m, n, k, lda, ldb, ldc,
                            sizeof(double), sizeof(double), sizeof(double),
                            &size_a, &size_b, &size_c);
        cublas_record_range("cublasDgemm_v2", "A", A, size_a, stream);
        cublas_record_range("cublasDgemm_v2", "B", B, size_b, stream);
        cublas_record_range("cublasDgemm_v2", "C", C, size_c, stream);
    }

    if (real_cublasDgemm_v2) {
        return real_cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t cublasDgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc
) {
    return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ── Intercepted: cublasSgemv_v2 / cublasSgemv ────────────────────────────────
extern "C" cublasStatus_t cublasSgemv_v2(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy
) {
    ensure_init();
    process_pending();

    if (!real_cublasSgemv_v2) {
        real_cublasSgemv_v2 = (cublasSgemv_v2_t)get_cublas_symbol("cublasSgemv_v2");
        if (!real_cublasSgemv_v2) real_cublasSgemv_v2 = (cublasSgemv_v2_t)get_cublas_symbol("cublasSgemv");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_x = 0, size_y = 0;
        get_gemv_dimensions(trans, m, n, lda, incx, incy, sizeof(float), &size_a, &size_x, &size_y);
        cublas_record_range("cublasSgemv_v2", "A", A, size_a, stream);
        cublas_record_range("cublasSgemv_v2", "x", x, size_x, stream);
        cublas_record_range("cublasSgemv_v2", "y", y, size_y, stream);
    }

    if (real_cublasSgemv_v2) {
        return real_cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t cublasSgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy
) {
    return cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// ── Intercepted: cublasDgemv_v2 / cublasDgemv ────────────────────────────────
extern "C" cublasStatus_t cublasDgemv_v2(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double *alpha,
    const double *A, int lda,
    const double *x, int incx,
    const double *beta,
    double *y, int incy
) {
    ensure_init();
    process_pending();

    if (!real_cublasDgemv_v2) {
        real_cublasDgemv_v2 = (cublasDgemv_v2_t)get_cublas_symbol("cublasDgemv_v2");
        if (!real_cublasDgemv_v2) real_cublasDgemv_v2 = (cublasDgemv_v2_t)get_cublas_symbol("cublasDgemv");
    }
    if (!real_cublasGetStream_v2) {
        real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream_v2");
        if (!real_cublasGetStream_v2) real_cublasGetStream_v2 = (cublasGetStream_v2_t)get_cublas_symbol("cublasGetStream");
    }

    if (is_tracking_enabled_host()) {
        cudaStream_t stream = nullptr;
        if (real_cublasGetStream_v2 && handle) {
            real_cublasGetStream_v2(handle, &stream);
        }
        size_t size_a = 0, size_x = 0, size_y = 0;
        get_gemv_dimensions(trans, m, n, lda, incx, incy, sizeof(double), &size_a, &size_x, &size_y);
        cublas_record_range("cublasDgemv_v2", "A", A, size_a, stream);
        cublas_record_range("cublasDgemv_v2", "x", x, size_x, stream);
        cublas_record_range("cublasDgemv_v2", "y", y, size_y, stream);
    }

    if (real_cublasDgemv_v2) {
        return real_cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t cublasDgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double *alpha,
    const double *A, int lda,
    const double *x, int incx,
    const double *beta,
    double *y, int incy
) {
    return cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// ── Host-Side Static Analysis + Tripwire Temporal Correlator ─────────────────
struct InterceptTemporalMilestoneEntry {
    uint32_t kernel_id;
    uint32_t milestone_id;
    uint64_t timestamp_ns;
    uint64_t base_addr;
    uint32_t size_bytes;
    uint16_t block_id;
    uint16_t flags;
};

extern "C" void uvm_export_static_temporal_trace(const char* filename) {
    std::vector<StaticTemporalLaunchRecord> launches;
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        launches = get_static_temporal_launches();
    }
    if (launches.empty()) return;

    auto copy_fn = (uint32_t (*)(InterceptTemporalMilestoneEntry*, uint32_t))dlsym(RTLD_DEFAULT, "uvm_temporal_copy_entries");
    std::vector<InterceptTemporalMilestoneEntry> entries(65536);
    uint32_t count = copy_fn ? copy_fn(entries.data(), 65536) : 0;

    if (!copy_fn || count == 0) {
        // Standalone Host-Side Tripwire Trace Export (for general compute engines: llama.cpp, PyTorch, CuPy)
        FILE* f = fopen(filename, "w");
        if (!f) return;
        uint64_t t0 = launches[0].timestamp_ns;
        fprintf(f, "{\n  \"mode\": \"static_analysis_plus_host_tripwire\",\n");
        fprintf(f, "  \"total_milestones\": %zu,\n  \"total_launches\": %zu,\n  \"events\": [\n",
                launches.size(), launches.size());
        for (size_t i = 0; i < launches.size(); ++i) {
            const auto& rec = launches[i];
            double rel_us = (double)(rec.timestamp_ns - t0) / 1000.0;
            size_t total_bytes = 0;
            for (const auto& r : rec.ranges) total_bytes += r.second;
            size_t total_pages = (total_bytes + 4095) / 4096;
            uintptr_t base = rec.ranges.empty() ? 0 : rec.ranges[0].first;
            fprintf(f, "%s    {\"kernel\": \"%s\", \"kernel_id\": %u, \"phase\": \"KERNEL_LAUNCH\", "
                       "\"milestone_id\": %zu, \"rel_time_us\": %.3f, "
                       "\"static_slice_base\": \"0x%llx\", \"static_slice_bytes\": %zu, "
                       "\"static_slice_pages\": %zu, \"total_kernel_static_bytes\": %zu}",
                    (i == 0) ? "" : ",\n",
                    rec.kernel_name.c_str(), rec.kernel_id, i, rel_us,
                    (unsigned long long)base, total_bytes, total_pages, total_bytes);
        }
        fprintf(f, "\n  ]\n}\n");
        fclose(f);
        fprintf(stderr, "[UVM_STATIC_TRIPWIRE] Exported standalone host tripwire trace (%zu launches) to %s\n",
                launches.size(), filename);
        return;
    }

    FILE* f = fopen(filename, "w");
    if (!f) return;

    uint64_t t0 = entries[0].timestamp_ns;
    fprintf(f, "{\n  \"mode\": \"static_analysis_plus_host_tripwire\",\n");
    fprintf(f, "  \"total_milestones\": %u,\n  \"total_launches\": %zu,\n  \"events\": [\n",
            count, launches.size());

    size_t launch_idx = 0;
    bool first_ev = true;

    uint32_t i = 0;
    while (i < count) {
        if (entries[i].flags == 1 /* KERNEL_ENTRY */) {
            uint32_t kid = entries[i].kernel_id;
            // Find matching launch record
            const StaticTemporalLaunchRecord* rec = nullptr;
            for (size_t k = launch_idx; k < launches.size(); ++k) {
                if (launches[k].kernel_id == kid) {
                    rec = &launches[k];
                    launch_idx = k + 1;
                    break;
                }
            }
            if (!rec && !launches.empty()) {
                for (size_t k = 0; k < launches.size(); ++k) {
                    if (launches[k].kernel_id == kid) {
                        rec = &launches[k];
                        break;
                    }
                }
            }

            // Count outer loop iterations until KERNEL_EXIT (flags == 2)
            uint32_t end_i = i + 1;
            uint32_t num_iters = 0;
            while (end_i < count && entries[end_i].flags != 2) {
                if (entries[end_i].flags == 0) num_iters++;
                end_i++;
            }
            if (num_iters == 0) num_iters = 1;

            std::string kname = rec ? rec->kernel_name : "unknown_kernel";
            size_t total_static_bytes = 0;
            if (rec) {
                for (const auto& r : rec->ranges) total_static_bytes += r.second;
            }

            uint32_t iter_ord = 0;
            for (uint32_t j = i; j <= end_i && j < count; ++j) {
                const auto& e = entries[j];
                double rel_us = (double)(e.timestamp_ns - t0) / 1000.0;
                const char* phase = (e.flags == 1) ? "KERNEL_ENTRY" : ((e.flags == 2) ? "KERNEL_EXIT" : "OUTER_LOOP_STEP");

                uintptr_t slice_base = (uintptr_t)e.base_addr;
                size_t slice_bytes = e.size_bytes;
                if (e.flags == 0 && rec && !rec->ranges.empty()) {
                    // Partition primary static range across outer-loop steps
                    uintptr_t r_base = rec->ranges[0].first;
                    size_t r_size = rec->ranges[0].second;
                    size_t step_sz = r_size / num_iters;
                    slice_base = r_base + (iter_ord % num_iters) * step_sz;
                    slice_bytes = step_sz;
                    iter_ord++;
                } else if (e.flags != 0 && rec && !rec->ranges.empty()) {
                    slice_base = rec->ranges[0].first;
                    slice_bytes = total_static_bytes;
                }
                size_t slice_pages = (slice_bytes + 4095) / 4096;

                fprintf(f, "%s    {\"kernel\": \"%s\", \"kernel_id\": %u, \"phase\": \"%s\", "
                           "\"milestone_id\": %u, \"rel_time_us\": %.3f, "
                           "\"static_slice_base\": \"0x%llx\", \"static_slice_bytes\": %zu, "
                           "\"static_slice_pages\": %zu, \"total_kernel_static_bytes\": %zu}",
                        first_ev ? "" : ",\n",
                        kname.c_str(), e.kernel_id, phase, e.milestone_id, rel_us,
                        (unsigned long long)slice_base, slice_bytes, slice_pages, total_static_bytes);
                first_ev = false;
            }
            i = (end_i < count) ? (end_i + 1) : count;
        } else {
            i++;
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[UVM_STATIC_TRIPWIRE] Exported correlated host-static + temporal trace (%u milestones) to %s\n",
            count, filename);
}

// ── Cleanup Destructor: print proactive prefetcher stats ─────────────────────
static void uvm_sigterm_handler(int sig) {
    g_prefetcher.print_stats();
    _exit(0);
}

__attribute__((destructor)) static void uvm_prefetcher_cleanup() {
    static const char* st_env = getenv("UVM_STATIC_TRIPWIRE");
    if (st_env && st_env[0] == '1') {
        const char* out = getenv("UVM_STATIC_TEMPORAL_OUTPUT");
        if (!out) out = "temporal_host_trace.json";
        uvm_export_static_temporal_trace(out);
    }
    g_prefetcher.print_stats();
}

__attribute__((constructor)) static void uvm_prefetcher_init_signals() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = uvm_sigterm_handler;
    sigaction(SIGTERM, &sa, nullptr);
}


