#!/usr/bin/env python3
import argparse
import struct
import os

MAGIC       = 0x50474C47
HEADER_FMT  = "<IHHHHIIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
INDEX_FMT   = "<HHQ"
INDEX_SIZE  = struct.calcsize(INDEX_FMT)


def parse_pagelog(path):
    """Return a set of touched page base addresses (byte addresses, 4 KB aligned)."""
    with open(path, "rb") as f:
        raw = f.read()
    hdr = struct.unpack_from(HEADER_FMT, raw)
    magic, l3_bytes, l1_shift, l2_shift, l3_shift, num_leaves = (
        hdr[0], hdr[4], hdr[5], hdr[6], hdr[7], hdr[8])
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X} in {path}")
    pages = set()
    for k in range(num_leaves):
        l1_idx, l2_idx, data_off = struct.unpack_from(INDEX_FMT, raw, HEADER_SIZE + k * INDEX_SIZE)
        for w in range(l3_bytes // 8):
            word = struct.unpack_from("<Q", raw, data_off + w * 8)[0]
            if not word:
                continue
            for b in range(64):
                if word & (1 << b):
                    l3_off = w * 64 + b
                    pages.add((l1_idx << l1_shift) | (l2_idx << l2_shift) | (l3_off << l3_shift))
    return pages


def cluster_sizes(pages):
    """Return sorted list of contiguous-run lengths (in pages)."""
    if not pages:
        return []
    PAGE = 4096
    sp = sorted(pages)
    sizes, sz = [], 1
    for prev, cur in zip(sp, sp[1:]):
        if cur == prev + PAGE:
            sz += 1
        else:
            sizes.append(sz)
            sz = 1
    sizes.append(sz)
    return sorted(sizes)


def analyze_pages(filepath, print_pages=False):
    if not os.path.exists(filepath):
        print(f"Error: file '{filepath}' not found.")
        return

    pages = parse_pagelog(filepath)
    sizes = cluster_sizes(pages)

    print("=" * 40)
    print(f"Pagelog Analysis: {filepath}")
    print("=" * 40)
    print(f"Total unique pages touched: {len(pages)}")
    print(f"Contiguous clusters:        {len(sizes)}")
    if sizes:
        print(f"Cluster sizes (pages):      {sizes}")
        print(f"  min={min(sizes)}  max={max(sizes)}  total={sum(sizes)}")

    if print_pages and pages:
        print("\nTouched page addresses:")
        print("-" * 40)
        for addr in sorted(pages):
            print(f"  0x{addr:016x}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze UVM access tracking pagelogs.")
    parser.add_argument("file", nargs="?", default="access_log.bin",
                        help="Path to pagelog file (default: access_log.bin)")
    parser.add_argument("-p", "--print-pages", action="store_true",
                        help="Print the hex address of every touched page")
    args = parser.parse_args()
    analyze_pages(args.file, args.print_pages)
