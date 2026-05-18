#!/usr/bin/env python3
"""
benchs/plot_bigann_scaling.py

Scaling plots for the BIGANN family (100k -> 100m), reading the per-size JSONs
written by bench_router_paper.py:

    benchs/results_router/results_bigann<size>.json

Sizes are auto-detected from filenames; supported tokens: 100k, 200k, 500k,
1m, 2m, 5m, 10m, 20m, 50m, 100m. Any subset present is plotted; missing sizes
are skipped silently.

Produces, under --out-dir (default benchs/figures_router/bigann_scaling/):
  - build_time_vs_n.{png,pdf}        single panel
  - size_vs_n.{png,pdf}              single panel: serialised index size
  - qps_vs_n_k10_r95.{png,pdf}       single panel: QPS at Recall@10>=0.95
  - crossover_vs_n_k10.{png,pdf}     two panels: N* vs n at r95 and r99

  Tables under tables/:
  - construction.csv
  - qps_k10_r95.csv
  - crossover_k10.csv

Usage:
  python benchs/plot_bigann_scaling.py
  python benchs/plot_bigann_scaling.py --results-dir benchs/bigann_results \\
                                       --out-dir benchs/figures_router/bigann_scaling \\
                                       --formats png pdf
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style (kept consistent with plot_router_paper.py)
# ---------------------------------------------------------------------------

INDICES = ["SuCo", "SHG", "CSPG", "HNSW32", "HNSW48"]

INDEX_COLOR = {
    "SuCo":   "#1f77b4",
    "SHG":    "#f4a261",
    "CSPG":   "#2ca02c",
    "HNSW32": "#d62728",
    "HNSW48": "#9467bd",
}

INDEX_MARKER = {
    "SuCo":   "o",
    "SHG":    "s",
    "CSPG":   "D",
    "HNSW32": "^",
    "HNSW48": "v",
}

RECALL_TARGET_LABEL = {"r80": "0.80", "r90": "0.90", "r95": "0.95", "r99": "0.99"}

# Canonical token -> numeric N. Anything not in this map is parsed from filename.
SIZE_ORDER = ["100k", "200k", "500k",
              "1m", "2m", "5m", "10m", "20m", "50m", "100m"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^results_bigann([0-9]+(?:k|m))\.json$", re.IGNORECASE)


def parse_size_token(tok):
    """'500k' -> 500_000 ; '20m' -> 20_000_000."""
    tok = tok.lower()
    m = re.fullmatch(r"([0-9]+)(k|m)", tok)
    if not m:
        raise ValueError(f"bad size token {tok!r}")
    n = int(m.group(1))
    return n * (1_000 if m.group(2) == "k" else 1_000_000)


def load_bigann(results_dir):
    """Return list of (size_token, n_int, results_dict) sorted by N."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "results_bigann*.json"))):
        m = _SIZE_RE.match(os.path.basename(path))
        if not m:
            continue
        tok = m.group(1).lower()
        try:
            n_int = parse_size_token(tok)
        except ValueError:
            continue
        with open(path) as f:
            res = json.load(f)
        n_int = res.get("n", n_int) or n_int
        out.append((tok, int(n_int), res))
    if not out:
        sys.exit(f"No results_bigann*.json found in {results_dir}")

    def sort_key(item):
        tok = item[0]
        if tok in SIZE_ORDER:
            return (0, SIZE_ORDER.index(tok))
        return (1, item[1])

    out.sort(key=sort_key)
    return out


def save_fig(fig, out_dir, name, formats):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt_n(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ---------------------------------------------------------------------------
# Shared panel drawer + legend
# ---------------------------------------------------------------------------

def _draw_vs_n_panel(ax, series, ylabel, title, log_y=True, xticks=None,
                     index_order=None):
    plotted = False
    order = index_order if index_order is not None else INDICES
    for idx in order:
        pts = series.get(idx, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=INDEX_MARKER[idx], color=INDEX_COLOR[idx],
                label=idx, linewidth=1.6, markersize=5)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("N (vectors)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([fmt_n(x) for x in xticks], fontsize=8)
        ax.minorticks_off()
    return plotted


def _shared_legend(fig, order=None):
    order = order if order is not None else INDICES
    handles = [
        plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                   marker=INDEX_MARKER[idx], label=idx, linewidth=1.6)
        for idx in order
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(order), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))


# ---------------------------------------------------------------------------
# (1) Build time vs N — headline
# ---------------------------------------------------------------------------

def plot_build_time_vs_n(loaded, out_dir, formats):
    xticks = [n for _, n, _ in loaded]
    series = {idx: [] for idx in INDICES}
    for _, n, res in loaded:
        cons = res.get("construction", {})
        for idx in INDICES:
            v = cons.get(idx, {}).get("build_time_s")
            if v is None or v == -1:
                continue
            series[idx].append((n, v))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _draw_vs_n_panel(ax, series, ylabel="Build time (s)",
                     title="Build time vs N (log-log)",
                     log_y=True, xticks=xticks)
    _shared_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir, "build_time_vs_n", formats)


# ---------------------------------------------------------------------------
# (2) Serialised index size vs N — companion to build time
# ---------------------------------------------------------------------------
# Peak build memory is not reported here: ru_maxrss is a process-lifetime
# high-water mark and the per-index delta zeroes out for every index built
# after the first. Serialised size is a faithful proxy at these scales
# because the raw vector matrix dominates and the build-time scratch is
# O(1) in n.

def plot_size_vs_n(loaded, out_dir, formats):
    xticks = [n for _, n, _ in loaded]
    series = {idx: [] for idx in INDICES}
    for _, n, res in loaded:
        cons = res.get("construction", {})
        for idx in INDICES:
            v = cons.get(idx, {}).get("size_mb")
            if v is None or v == -1:
                continue
            series[idx].append((n, v))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _draw_vs_n_panel(ax, series, ylabel="Serialised index size (MB)",
                     title="Serialised index size vs N (log-log)",
                     log_y=True, xticks=xticks)
    _shared_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir, "size_vs_n", formats)


# ---------------------------------------------------------------------------
# (3) QPS at matched Recall@10 >= 0.95 vs N
# ---------------------------------------------------------------------------

def plot_qps_at_recall_vs_n(loaded, out_dir, formats, k=10, target="r95"):
    xticks = [n for _, n, _ in loaded]
    series = {idx: [] for idx in INDICES}
    for _, n, res in loaded:
        sec = (res.get("time_at_recall", {})
                  .get(f"recall_k{k}", {}).get(target, {}))
        for idx in INDICES:
            v = sec.get(idx, {}).get("qps")
            if v is None:
                continue
            series[idx].append((n, v))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _draw_vs_n_panel(
        ax, series,
        ylabel=f"QPS at Recall@{k} >= {RECALL_TARGET_LABEL[target]}",
        title=f"QPS at matched recall vs N (k={k}, "
              f"recall>={RECALL_TARGET_LABEL[target]})",
        log_y=True, xticks=xticks,
    )
    _shared_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir, f"qps_vs_n_k{k}_{target}", formats)


# ---------------------------------------------------------------------------
# (4) Cumulative-cost crossover N* vs database size n
# ---------------------------------------------------------------------------
# For each database size n and each non-SuCo index, the crossover query volume
# is the smallest number of queries N* at which
#       build_time[SuCo] + N* . ms[SuCo]/1000
#     = build_time[idx]  + N* . ms[idx]/1000.
# Solving: N* = 1000 * (build[idx] - build[SuCo]) / (ms[SuCo] - ms[idx]).
# N* > 0 only when the graph index is both more expensive to build AND faster
# per query than SuCo; otherwise no crossover exists.

CROSSOVER_BASELINE = "SuCo"


def _crossover_query_count(build_a, ms_a, build_b, ms_b):
    if None in (build_a, ms_a, build_b, ms_b):
        return None
    if build_a < 0 or build_b < 0 or ms_a <= 0 or ms_b <= 0:
        return None
    db, dms = build_b - build_a, ms_a - ms_b
    if db <= 0 or dms <= 0:
        return None
    return 1000.0 * db / dms


def plot_crossover_vs_n(loaded, out_dir, formats, k=10,
                        recall_targets=("r95", "r99")):
    others = [idx for idx in INDICES if idx != CROSSOVER_BASELINE]
    xticks = [n for _, n, _ in loaded]
    fig, axes = plt.subplots(1, len(recall_targets),
                             figsize=(5.4 * len(recall_targets), 3.8),
                             squeeze=False)
    for j, tgt in enumerate(recall_targets):
        ax = axes[0][j]
        series = {idx: [] for idx in others}
        for _, n, res in loaded:
            cons = res.get("construction", {})
            tar = (res.get("time_at_recall", {})
                      .get(f"recall_k{k}", {}).get(tgt, {}))
            base_build = cons.get(CROSSOVER_BASELINE, {}).get("build_time_s")
            base_ms    = tar.get(CROSSOVER_BASELINE, {}).get("ms_per_query")
            for idx in others:
                build = cons.get(idx, {}).get("build_time_s")
                ms    = tar.get(idx, {}).get("ms_per_query")
                nstar = _crossover_query_count(base_build, base_ms, build, ms)
                if nstar is None or nstar <= 0:
                    continue
                series[idx].append((n, nstar))
        _draw_vs_n_panel(
            ax, series,
            ylabel=(f"crossover query volume N* vs {CROSSOVER_BASELINE}"
                    if j == 0 else ""),
            title=f"Recall@{k} >= {RECALL_TARGET_LABEL[tgt]}",
            log_y=True, xticks=xticks, index_order=others,
        )
    _shared_legend(fig, order=others)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir, f"crossover_vs_n_k{k}", formats)


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------

def write_construction_table(loaded, out_dir):
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    fields = ("build_time_s", "memory_mb", "size_mb")
    header = ["N", "size"] + [f"{idx}_{f}" for idx in INDICES for f in fields]
    rows = [header]
    for tok, n, res in loaded:
        row = [n, tok]
        cons = res.get("construction", {})
        for idx in INDICES:
            for f in fields:
                v = cons.get(idx, {}).get(f)
                row.append("" if v is None else v)
        rows.append(row)
    with open(os.path.join(tables_dir, "construction.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_qps_table(loaded, out_dir, k=10, target="r95"):
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    header = ["N", "size"] + [f"{idx}_qps" for idx in INDICES]
    rows = [header]
    for tok, n, res in loaded:
        row = [n, tok]
        sec = (res.get("time_at_recall", {})
                  .get(f"recall_k{k}", {}).get(target, {}))
        for idx in INDICES:
            v = sec.get(idx, {}).get("qps")
            row.append("" if v is None else f"{v:.2f}")
        rows.append(row)
    with open(os.path.join(tables_dir, f"qps_k{k}_{target}.csv"),
              "w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_crossover_table(loaded, out_dir, k=10,
                          recall_targets=("r95", "r99")):
    others = [idx for idx in INDICES if idx != CROSSOVER_BASELINE]
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    header = ["N", "size"] + [f"{idx}_{tgt}_Nstar"
                              for idx in others for tgt in recall_targets]
    rows = [header]
    for tok, n, res in loaded:
        row = [n, tok]
        cons = res.get("construction", {})
        for idx in others:
            for tgt in recall_targets:
                tar = (res.get("time_at_recall", {})
                          .get(f"recall_k{k}", {}).get(tgt, {}))
                nstar = _crossover_query_count(
                    cons.get(CROSSOVER_BASELINE, {}).get("build_time_s"),
                    tar.get(CROSSOVER_BASELINE, {}).get("ms_per_query"),
                    cons.get(idx, {}).get("build_time_s"),
                    tar.get(idx, {}).get("ms_per_query"),
                )
                row.append("" if nstar is None else f"{nstar:.0f}")
        rows.append(row)
    with open(os.path.join(tables_dir, f"crossover_k{k}.csv"),
              "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="benchs/results_router")
    p.add_argument("--out-dir",     default="benchs/figures_router/bigann_scaling")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = p.parse_args()

    loaded = load_bigann(args.results_dir)
    print(f"[plot_bigann_scaling] loaded sizes: "
          f"{', '.join(tok for tok, _, _ in loaded)}")

    plot_build_time_vs_n(loaded,     args.out_dir, args.formats)
    plot_size_vs_n(loaded,           args.out_dir, args.formats)
    plot_qps_at_recall_vs_n(loaded,  args.out_dir, args.formats,
                            k=10, target="r95")
    plot_crossover_vs_n(loaded,      args.out_dir, args.formats, k=10)

    write_construction_table(loaded, args.out_dir)
    write_qps_table(loaded,          args.out_dir, k=10, target="r95")
    write_crossover_table(loaded,    args.out_dir, k=10)

    print(f"[plot_bigann_scaling] wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
