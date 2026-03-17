#!/usr/bin/env python3
"""
pagelog_drill.py — hierarchical drill-down heatmap for shadow page table logs

Level 0 : 16×32 grid, each cell = one L1 region  (covers 2^39 bytes each)
Level 1 : 64×64 grid, each cell = one L2 region  (covers 2^27 bytes each)
Level 2 : 128×256 grid, each cell = one page      (the raw L3 bitmap)

Usage:
    pip install dash plotly numpy
    python pagelog_drill.py run.pagelog
    open http://127.0.0.1:8050
"""

import argparse
import struct
import sys
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ── binary format ─────────────────────────────────────────────────────────────
MAGIC            = 0x50474C47
HEADER_FMT       = "<IHHHHIIIq"
HEADER_SIZE      = struct.calcsize(HEADER_FMT)
INDEX_ENTRY_FMT  = "<HHQ"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FMT)

GH_DARK  = "#0d1117"
GH_CELL  = "#161b22"
GH_GREEN = [
    [0.00, "#161b22"],
    [0.01, "#0e4429"],
    [0.25, "#006d32"],
    [0.60, "#26a641"],
    [1.00, "#39d353"],
]

# ── load binary log ───────────────────────────────────────────────────────────
def load_log(path):
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        magic, version, l1_entries, l2_entries, l3_bytes, \
            l1_shift, l2_shift, l3_shift, num_leaves = struct.unpack(HEADER_FMT, raw)
        assert magic == MAGIC, f"Bad magic {magic:#010x}"

        hdr = dict(l1_entries=l1_entries, l2_entries=l2_entries,
                   l3_bytes=l3_bytes, l1_shift=l1_shift,
                   l2_shift=l2_shift, l3_shift=l3_shift,
                   num_leaves=num_leaves)

        index = []
        for _ in range(num_leaves):
            l1, l2, offset = struct.unpack(INDEX_ENTRY_FMT, f.read(INDEX_ENTRY_SIZE))
            index.append((l1, l2, offset))

        # load all bitmaps into a dict keyed by (l1, l2)
        leaves = {}
        for l1, l2, offset in index:
            f.seek(offset)
            raw_bitmap = f.read(l3_bytes)
            leaves[(l1, l2)] = np.frombuffer(raw_bitmap, dtype=np.uint8).copy()

    return hdr, leaves


# ── count helpers ─────────────────────────────────────────────────────────────
def count_bits(bitmap_bytes: np.ndarray) -> int:
    """Count set bits in a byte array."""
    return int(np.unpackbits(bitmap_bytes).sum())


def l1_counts(hdr, leaves):
    """Returns array[L1_ENTRIES] of touched-page counts."""
    counts = np.zeros(hdr["l1_entries"], dtype=np.int64)
    for (l1, l2), bitmap in leaves.items():
        counts[l1] += count_bits(bitmap)
    return counts


def l2_counts(hdr, leaves, l1_idx):
    """Returns array[L2_ENTRIES] of touched-page counts for a given L1."""
    counts = np.zeros(hdr["l2_entries"], dtype=np.int64)
    for (l1, l2), bitmap in leaves.items():
        if l1 == l1_idx:
            counts[l2] += count_bits(bitmap)
    return counts


def l3_bitmap(hdr, leaves, l1_idx, l2_idx):
    """Returns bit array[L3_BITS] for a given (L1, L2) leaf, or zeros."""
    bitmap = leaves.get((l1_idx, l2_idx),
                        np.zeros(hdr["l3_bytes"], dtype=np.uint8))
    return np.unpackbits(bitmap).astype(np.float32)


# ── VA label helpers ──────────────────────────────────────────────────────────
def l1_va(hdr, l1):
    return l1 << hdr["l1_shift"]

def l2_va(hdr, l1, l2):
    return (l1 << hdr["l1_shift"]) | (l2 << hdr["l2_shift"])

def l3_va(hdr, l1, l2, bit):
    return (l1 << hdr["l1_shift"]) | (l2 << hdr["l2_shift"]) | (bit << hdr["l3_shift"])


# ── figure builders ───────────────────────────────────────────────────────────
def make_l1_fig(hdr, leaves):
    counts = l1_counts(hdr, leaves)
    # 16 rows × 32 cols = 512
    rows, cols = 16, 32
    grid = counts.reshape(rows, cols).astype(np.float64)

    hover = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            va  = l1_va(hdr, idx)
            va_end = l1_va(hdr, idx + 1) - 1
            row.append(f"L1[{idx}]<br>"
                       f"{va:#018x} – {va_end:#018x}<br>"
                       f"{int(grid[r,c]):,} pages touched")
        hover.append(row)

    fig = go.Figure(go.Heatmap(
        z=grid, text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=GH_GREEN, showscale=False,
        xgap=2, ygap=2,
    ))
    _style(fig,
           title="L1 — full 48-bit address space  (click a cell to drill in)",
           subtitle=f"{int(counts.sum()):,} total pages touched across {hdr['l1_entries']} L1 regions",
           xlabel="L1 column", ylabel="L1 row")
    return fig


def make_l2_fig(hdr, leaves, l1_idx):
    counts = l2_counts(hdr, leaves, l1_idx)
    # 64 rows × 64 cols = 4096
    rows, cols = 64, 64
    grid = counts.reshape(rows, cols).astype(np.float64)

    hover = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            va     = l2_va(hdr, l1_idx, idx)
            va_end = l2_va(hdr, l1_idx, idx + 1) - 1
            row.append(f"L1[{l1_idx}] → L2[{idx}]<br>"
                       f"{va:#018x} – {va_end:#018x}<br>"
                       f"{int(grid[r,c]):,} pages touched")
        hover.append(row)

    va_base = l1_va(hdr, l1_idx)
    fig = go.Figure(go.Heatmap(
        z=grid, text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=GH_GREEN, showscale=False,
        xgap=1, ygap=1,
    ))
    _style(fig,
           title=f"L2 — L1[{l1_idx}]  base {va_base:#018x}  (click to drill into L3)",
           subtitle=f"{int(counts.sum()):,} pages touched across {hdr['l2_entries']} L2 regions",
           xlabel="L2 column", ylabel="L2 row")
    return fig


def make_l3_fig(hdr, leaves, l1_idx, l2_idx):
    bits = l3_bitmap(hdr, leaves, l1_idx, l2_idx)
    l3_bits = hdr["l3_bytes"] * 8
    # 128 rows × 256 cols = 32768 bits
    rows, cols = 128, 256
    grid = bits[:rows * cols].reshape(rows, cols)

    hover = []
    for r in range(rows):
        row = []
        for c in range(cols):
            bit = r * cols + c
            va  = l3_va(hdr, l1_idx, l2_idx, bit)
            state = "touched" if grid[r, c] else "not touched"
            row.append(f"L1[{l1_idx}] L2[{l2_idx}] page[{bit}]<br>"
                       f"{va:#018x}<br>{state}")
        hover.append(row)

    va_base = l2_va(hdr, l1_idx, l2_idx)
    touched = int(bits.sum())
    fig = go.Figure(go.Heatmap(
        z=grid, text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=GH_GREEN, showscale=False,
        xgap=1, ygap=1,
    ))
    _style(fig,
           title=f"L3 — L1[{l1_idx}] L2[{l2_idx}]  base {va_base:#018x}",
           subtitle=f"{touched:,} / {l3_bits:,} pages touched",
           xlabel="page column", ylabel="page row")
    return fig


def _style(fig, title, subtitle, xlabel, ylabel):
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            font=dict(color="#e6edf3", size=14),
        ),
        paper_bgcolor=GH_DARK,
        plot_bgcolor=GH_DARK,
        font=dict(color="#e6edf3", family="monospace", size=10),
        xaxis=dict(title=xlabel, gridcolor="#21262d", tickfont=dict(size=8)),
        yaxis=dict(title=ylabel, gridcolor="#21262d", tickfont=dict(size=8),
                   autorange="reversed"),
        margin=dict(l=60, r=20, t=80, b=50),
        dragmode="zoom",
        height=600,
    )


# ── Dash app ──────────────────────────────────────────────────────────────────
def run_app(path):
    hdr, leaves = load_log(path)
    print(f"[app] loaded {len(leaves)} leaves from {path}")

    app = dash.Dash(__name__)

    app.layout = html.Div(style={"backgroundColor": GH_DARK,
                                  "minHeight": "100vh",
                                  "fontFamily": "monospace",
                                  "color": "#e6edf3",
                                  "padding": "20px"}, children=[

        html.H2("Shadow Page Table — Access Heatmap",
                style={"color": "#e6edf3", "marginBottom": "4px"}),

        # breadcrumb
        html.Div(id="breadcrumb",
                 style={"color": "#8b949e", "marginBottom": "16px",
                        "fontSize": "13px"}),

        # back button
        html.Button("← Back",
                    id="back-btn",
                    n_clicks=0,
                    style={"display": "none",
                           "backgroundColor": "#21262d",
                           "color": "#e6edf3",
                           "border": "1px solid #30363d",
                           "padding": "6px 16px",
                           "borderRadius": "6px",
                           "cursor": "pointer",
                           "marginBottom": "12px"}),

        dcc.Graph(id="heatmap", config={"scrollZoom": True}),

        # state store: {level: 0/1/2, l1: int|None, l2: int|None}
        dcc.Store(id="nav-state", data={"level": 0, "l1": None, "l2": None}),
    ])

    @app.callback(
        Output("heatmap",    "figure"),
        Output("nav-state",  "data"),
        Output("breadcrumb", "children"),
        Output("back-btn",   "style"),
        Input("heatmap",   "clickData"),
        Input("back-btn",  "n_clicks"),
        State("nav-state", "data"),
    )
    def navigate(click_data, back_clicks, state):
        ctx   = dash.callback_context
        level = state["level"]
        l1    = state["l1"]
        l2    = state["l2"]

        btn_show  = {"display": "inline-block",
                     "backgroundColor": "#21262d", "color": "#e6edf3",
                     "border": "1px solid #30363d", "padding": "6px 16px",
                     "borderRadius": "6px", "cursor": "pointer",
                     "marginBottom": "12px"}
        btn_hide  = {"display": "none"}

        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        # ── back button ───────────────────────────────────────────────────────
        if "back-btn" in triggered:
            if level == 2:
                level, l2 = 1, None
            elif level == 1:
                level, l1, l2 = 0, None, None

        # ── cell click ────────────────────────────────────────────────────────
        elif "heatmap" in triggered and click_data:
            pt    = click_data["points"][0]
            r, c  = pt["y"], pt["x"]

            if level == 0:
                cols  = 32
                l1    = int(r) * cols + int(c)
                level = 1
            elif level == 1:
                cols  = 64
                l2    = int(r) * cols + int(c)
                level = 2

        # ── build figure ──────────────────────────────────────────────────────
        if level == 0:
            fig   = make_l1_fig(hdr, leaves)
            crumb = "full address space"
            btn   = btn_hide
        elif level == 1:
            fig   = make_l2_fig(hdr, leaves, l1)
            va    = l1_va(hdr, l1)
            crumb = f"full address space  ›  L1[{l1}] ({va:#018x})"
            btn   = btn_show
        else:
            fig   = make_l3_fig(hdr, leaves, l1, l2)
            va    = l2_va(hdr, l1, l2)
            crumb = (f"full address space  ›  L1[{l1}] ({l1_va(hdr,l1):#018x})"
                     f"  ›  L2[{l2}] ({va:#018x})")
            btn   = btn_show

        new_state = {"level": level, "l1": l1, "l2": l2}
        return fig, new_state, crumb, btn

    print("[app] starting at http://127.0.0.1:8050")
    app.run(debug=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    args = ap.parse_args()
    run_app(args.logfile)
