
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

#pragma once

// ── Common definitions and utilities ───────────────────────────────────────────
#define DBG printf("At line: %d\n", __LINE__);
#define DBG_RET(ret) printf("At line: %d, returning %d\n", __LINE__, ret); return ret;
#define LOG printf


// ── Address decomposition (48-bit VA, 2-level + bitmap) ──────────────────────
//
//  [ 47 ......... 39 | 38 ......... 30 | 29 ......... 12 | 11 ......... 0 ]
//      L1 index (9b)     L2 index (9b)    L3 offset (18b)   page offset (12b)
//
//  Wait: 9 + 9 + 18 = 36 bits of VPN ✓  (2^36 pages in 48-bit space)
//
//  L1 table : 512  void**  pointers  (4KB, one page, static)
//  L2 table : 512  void*   pointers  (4KB, one page, on demand)
//  L3 leaf  : 2^18 bits = 32768 bytes = 32KB bitmap (on demand)
//             covers 2^18 pages = 2^18 * 2^12 = 1GB of VA per leaf

#define L1_ENTRIES   512          // 2^9  — one page
#define L2_ENTRIES   4096         // 2^12 — eight pages
#define L3_BYTES     4096         // 2^15 bits = 4096 bytes — one page

#define L3_SHIFT     12           // bits 12–26  (15 bits)
#define L2_SHIFT     27           // bits 27–38  (12 bits)
#define L1_SHIFT     39           // bits 39–47  (9 bits)
#define L1_MASK      0x1FFULL     // 9-bit mask
#define L2_MASK      0xFFFULL     // 12-bit mask
#define L3_MASK      0x7FFFULL    // 15-bit mask

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


// ── Binary page log format ────────────────────────────────────────────────────
//
//  HEADER (32 bytes):
//    [0..3]   magic      : 0x50474C47  ("PGLG")
//    [4..5]   version    : uint16  = 1
//    [6..7]   l1_entries : uint16
//    [8..9]   l2_entries : uint16
//    [10..11] l3_bytes   : uint16  (bytes per leaf)
//    [12..15] l1_shift   : uint32
//    [16..19] l2_shift   : uint32
//    [20..23] l3_shift   : uint32
//    [24..31] num_leaves : uint64  (number of populated L3 leaves)
//
//  INDEX (num_leaves * 12 bytes):
//    per entry:
//      [0..1]  l1_idx    : uint16
//      [2..3]  l2_idx    : uint16
//      [4..11] offset    : uint64  (byte offset from start of file to bitmap data)
//
//  BITMAP DATA:
//    num_leaves * L3_BYTES of raw bitmap bytes, in index order
//
// ─────────────────────────────────────────────────────────────────────────────

#define PAGELOG_MAGIC   0x50474C47u
#define PAGELOG_VERSION 1

#pragma pack(push, 1)
struct PageLogHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t l1_entries;
    uint16_t l2_entries;
    uint16_t l3_bytes;
    uint32_t l1_shift;
    uint32_t l2_shift;
    uint32_t l3_shift;
    uint64_t num_leaves;
};

struct PageLogIndexEntry {
    uint16_t l1_idx;
    uint16_t l2_idx;
    uint64_t offset;      // byte offset to bitmap data from start of file
};
#pragma pack(pop)


