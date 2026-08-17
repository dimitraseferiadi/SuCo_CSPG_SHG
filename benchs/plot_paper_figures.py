
#!/usr/bin/env python3
"""
benchs/plot_paper_figures.py

The two figures the paper carries beyond the Pareto grid, at IEEE column width.

The exploratory figures under figures_router/ are drawn through paper_style's
"big" sizing, which is calibrated for full-width slides.  At a 3.4in column the
same figures either overflow or collapse, and two of them plot a quantity other
than the one the text argues:

  latency_tail_r95        plots absolute percentiles.  SuCo's median is two
                          decades above the graph family, so its tail *ratio* --
                          the tightest in the study, and the entire point of the
                          paragraph -- is not visible anywhere on the figure.
  cumulative_cost_k10_r95 spends eleven panels (one of them empty) on a single
                          scalar per dataset, the crossover N*.

Both are redrawn here around the quantity the text claims, using the shared
palette and axis conventions of paper_style so they sit beside the thesis and
presentation figures without a visible style break -- but on PAPER_RCPARAMS
alone, without PAPER_BIG, which is what makes them survive the column.

Usage
  python benchs/plot_paper_figures.py --results-dir benchs/results_router \
                                      --out-dir benchs/figures_paper
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_style import (  # noqa: E402
    INDEX_COLOR, INDEX_MARKER, PAPER_RCPARAMS, clean_ax,
)
from plot_router_paper import (  # noqa: E402
    DATASETS, DATASET_LABEL, load_all,
)

# The shared look, at the base sizing rather than the slide sizing.
plt.rcParams.update(PAPER_RCPARAMS)
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.6,
})

COL_W = 3.45          # IEEE single column, inches
INDICES = ["SuCo", "SHG", "CSPG", "HNSW32", "HNSW48"]


def _dss(all_results):
    return [d for d in DATASETS if d in all_results]


def _xaxis(ax, dss):
    ax.set_xticks(np.arange(len(dss)))
    ax.set_xticklabels([DATASET_LABEL[d] for d in dss], rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(dss) - 0.5)


def _legend_below(ax, ncol=5, drop=0.46):
    """Legend under the panel, clear of the rotated dataset labels.

    Anchored to the axes rather than the figure: with bbox_inches="tight" a
    figure-anchored legend is placed before the rotated tick labels are measured
    and lands on top of them.
    """
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -drop), ncol=ncol,
              frameon=False, handlelength=1.4, columnspacing=1.1,
              handletextpad=0.4, borderaxespad=0.0)


# ---------------------------------------------------------------------------

def fig_tails(all_results, out_dir, target="r95", formats=("pdf", "png")):
    """Tail-to-median latency ratio p99.9/p50 at a fixed recall floor.

    Plotting the ratio rather than the percentiles is what makes the claim
    visible: on absolute axes SuCo sits two decades above the graph family and
    its flat tail cannot be read off at all.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 1.72))
    dss = _dss(all_results)

    for idx in INDICES:
        xs, ys = [], []
        for i, ds in enumerate(dss):
            node = ((all_results[ds].get("latency_tail") or {}).get(idx) or {}).get(target) or {}
            p50, p999 = node.get("p50"), node.get("p999")
            if p50 and p999:
                xs.append(i)
                ys.append(p999 / p50)
        if xs:
            ax.plot(xs, ys, marker=INDEX_MARKER.get(idx), label=idx,
                    color=INDEX_COLOR.get(idx), markeredgewidth=0)

    ax.axhline(1.0, color="0.35", lw=0.7, ls="--", zorder=0)
    # Log scale: SHG on Enron (17.4) is an order of magnitude above every other
    # point, and on a linear axis it flattens the 1.0-3.0 band that carries the
    # comparison between the remaining twelve.
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel(r"$p_{99.9}\,/\,p_{50}$")
    # latency_tail is measured at k=20 (bench_router_paper.run_latency_tail),
    # not at the k=10 used for the Pareto and crossover figures.
    ax.set_title(f"Tail-to-median latency at Recall@20 $\\geq$ "
                 f"0.{target[1:]}", pad=6)
    ax.grid(axis="y", alpha=0.45, which="both")
    clean_ax(ax)
    _xaxis(ax, dss)
    _legend_below(ax, ncol=5)
    _save(fig, out_dir, f"fig_tails_{target}", formats)


def fig_crossover(all_results, out_dir, k=10, target="r95", formats=("pdf", "png")):
    """Build-plus-query crossover N* of each graph index against SuCo.

    N* solves T_build(g) + N t_g = T_build(SuCo) + N t_SuCo: below it SuCo's
    near-zero build dominates, above it the graph's faster queries amortise its
    build.  One scalar per (dataset, index), so the eleven cumulative-cost panels
    collapse to a single axis without losing anything the text asserts.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 1.72))
    dss = _dss(all_results)
    graphs = ["SHG", "CSPG", "HNSW32", "HNSW48"]

    for idx in graphs:
        xs, ys = [], []
        for i, ds in enumerate(dss):
            res = all_results[ds]
            node = ((res.get("time_at_recall") or {}).get(f"recall_k{k}") or {}).get(target) or {}
            cons = res.get("construction") or {}
            ts = (node.get("SuCo") or {}).get("ms_per_query")
            tg = (node.get(idx) or {}).get("ms_per_query")
            bs = (cons.get("SuCo") or {}).get("build_time_s")
            bg = (cons.get(idx) or {}).get("build_time_s")
            if not (ts and tg and bs is not None and bg is not None):
                continue
            dt = (ts - tg) / 1000.0       # seconds the graph saves per query
            if dt <= 0:
                continue                   # no crossover where SuCo is faster
            xs.append(i)
            ys.append((bg - bs) / dt)
        if xs:
            ax.plot(xs, ys, marker=INDEX_MARKER.get(idx), label=idx,
                    color=INDEX_COLOR.get(idx), markeredgewidth=0)

    ax.set_yscale("log")
    ax.set_ylabel(r"$N^{*}$ (queries)")
    ax.set_title(r"Build-plus-query crossover against SuCo", pad=6)
    ax.grid(axis="y", alpha=0.45, which="both")
    clean_ax(ax)
    _xaxis(ax, dss)
    _legend_below(ax, ncol=4)
    _save(fig, out_dir, f"fig_crossover_k{k}_{target}", formats)


def fig_combined(all_results, out_dir, k=10, target="r95", formats=("pdf", "png")):
    """Both panels side by side across the full text width, one shared legend.

    The two share an x axis (the eleven datasets) and a palette, so drawing them
    as one figure removes a duplicated legend and a duplicated set of dataset
    labels, and costs roughly half the column space of two separate figures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.25))
    dss = _dss(all_results)

    # --- left: tail-to-median latency -------------------------------------
    for idx in INDICES:
        xs, ys = [], []
        for i, ds in enumerate(dss):
            node = ((all_results[ds].get("latency_tail") or {}).get(idx) or {}).get(target) or {}
            p50, p999 = node.get("p50"), node.get("p999")
            if p50 and p999:
                xs.append(i)
                ys.append(p999 / p50)
        if xs:
            ax1.plot(xs, ys, marker=INDEX_MARKER.get(idx), label=idx,
                     color=INDEX_COLOR.get(idx), markeredgewidth=0)
    ax1.axhline(1.0, color="0.35", lw=0.7, ls="--", zorder=0)
    ax1.set_yscale("log")
    ax1.set_yticks([1, 2, 5, 10, 20])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylabel(r"$p_{99.9}\,/\,p_{50}$")
    ax1.set_title("(a) Tail-to-median latency", pad=5)

    # --- right: build-plus-query crossover --------------------------------
    for idx in ["SHG", "CSPG", "HNSW32", "HNSW48"]:
        xs, ys = [], []
        for i, ds in enumerate(dss):
            res = all_results[ds]
            node = ((res.get("time_at_recall") or {}).get(f"recall_k{k}") or {}).get(target) or {}
            cons = res.get("construction") or {}
            ts = (node.get("SuCo") or {}).get("ms_per_query")
            tg = (node.get(idx) or {}).get("ms_per_query")
            bs = (cons.get("SuCo") or {}).get("build_time_s")
            bg = (cons.get(idx) or {}).get("build_time_s")
            if not (ts and tg and bs is not None and bg is not None):
                continue
            dt = (ts - tg) / 1000.0
            if dt <= 0:
                continue
            xs.append(i)
            ys.append((bg - bs) / dt)
        if xs:
            ax2.plot(xs, ys, marker=INDEX_MARKER.get(idx), label=idx,
                     color=INDEX_COLOR.get(idx), markeredgewidth=0)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$N^{*}$ (queries)")
    ax2.set_title(r"(b) Build-plus-query crossover vs SuCo", pad=5)

    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.45, which="both")
        clean_ax(ax)
        _xaxis(ax, dss)

    # One legend for both panels: ax1 carries all five series, ax2 a subset.
    _legend_below(ax1, ncol=5, drop=0.50)
    ax1.get_legend().set_bbox_to_anchor((1.16, -0.50), transform=ax1.transAxes)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, out_dir, f"fig_tails_crossover_k{k}_{target}", formats)


def _save(fig, out_dir, name, formats):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchs/results_router")
    ap.add_argument("--out-dir", default="benchs/figures_paper")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", default="r95")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()

    all_results = load_all(args.results_dir)
    fig_combined(all_results, args.out_dir, args.k, args.target, args.formats)
    fig_tails(all_results, args.out_dir, args.target, args.formats)
    fig_crossover(all_results, args.out_dir, args.k, args.target, args.formats)


if __name__ == "__main__":
    main()
