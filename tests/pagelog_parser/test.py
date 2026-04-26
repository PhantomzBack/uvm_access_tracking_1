#!/usr/bin/env python3
"""
Validates parse_pagelog and cluster_sizes against coalesced.pagelog.
Expected: 1024 contiguous pages in a single cluster.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from page_analysis import parse_pagelog, cluster_sizes

PAGELOG = "coalesced.pagelog"

failures = []

def check(desc, actual, expected):
    if actual == expected:
        print(f"  ok   {desc}")
    else:
        print(f"  FAIL {desc}: expected {expected!r}, got {actual!r}")
        failures.append(desc)

if not os.path.exists(PAGELOG):
    print(f"SKIP: {PAGELOG} not found")
    sys.exit(0)

pages = parse_pagelog(PAGELOG)
sizes = cluster_sizes(pages)

check("total pages",    len(pages), 1024)
check("cluster count",  len(sizes), 1)
check("cluster size",   sizes[0] if sizes else None, 1024)
check("pages are 4 KB aligned", all(p % 4096 == 0 for p in pages), True)
check("pages contiguous", sum(sizes), len(pages))

if failures:
    print("FAIL")
    sys.exit(1)
print("PASS")
