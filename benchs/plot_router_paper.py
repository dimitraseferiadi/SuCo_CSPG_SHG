#!/usr/bin/env python3
"""
benchs/plot_router_paper.py

Generate all figures from the router-paper benchmark results
(benchs/results_router/results_<dataset>.json, produced by bench_router_paper.py).

Indices: SuCo, SHG, CSPG, HNSW32, HNSW48
Datasets: sift1m, sift10m, gist1m, deep1m, deep10m, spacev10m,
          msong, enron, openai1m, msturing10m, uqv

Produces, under --out-dir (default benchs/figures_router/):
  Figures
  - pareto/<dataset>_k{1,10,20,50,100}.{png,pdf}    QPS vs Recall, log-y
  - pareto_grid_k{1,10,20,50,100}.{png,pdf}         3x4 grid, all datasets
  - construction_build_time.{png,pdf}               grouped bars, all datasets
  - construction_memory.{png,pdf}                   grouped bars (size_mb)
  - speedup_vs_hnsw32_k{1,10,20}.{png,pdf}          heatmap @ r80/r90/r95/r99
  - latency_tail_r{90,95,99}.{png,pdf}              p50/p95/p99/p999 panels
  - mre_at_recall.{png,pdf}                         MRE bars @ r90/r95/r99
  - robustness_box_k20.{png,pdf}                    per-query recall whiskers
  - hard_easy_recall.{png,pdf}                      hard vs easy mean recall
  - cold_warm_ratio.{png,pdf}                       cold/warm latency ratio
  - unseen_robustness.{png,pdf}                     mean recall on held-out

  Tables (under tables/, both .csv and .tex)
  - construction.{csv,tex}                          build time / size / memory
  - dataset_features.{csv,tex}                      n, d, nq, LID, pdist, kmeans
  - speedup_vs_hnsw32_k{1,10,20}.{csv,tex}          speedup at r80/r90/r95/r99
  - latency_tail_r{90,95,99}.{csv,tex}              p50/p95/p99/p999
  - mre_at_recall.{csv,tex}                         MRE at r90/r95/r99
  - robustness_k20.{csv,tex}                        min/q25/med/q75/max recall
  - hard_easy_recall.{csv,tex}                      easy/hard mean recall
  - cold_warm.{csv,tex}                             cold/warm latency + ratio
  - unseen_robustness.{csv,tex}                     held-out recall stats
  - recall_at_qps_targets_k{1,10,20}.{csv,tex}      best recall @ {1k,10k} qps

Datasets / metrics that are missing from a given file are skipped silently
(SuCo for example is not built on enron because d=1369 has no valid Ns).

Usage:
  python benchs/plot_router_paper.py
  python benchs/plot_router_paper.py --results-dir benchs/results_router \\
                                     --out-dir benchs/figures_router \\
                                     --formats png pdf
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DATASETS = [
    "sift1m", "sift10m", "gist1m", "deep1m", "deep10m", "spacev10m",
    "msong", "enron", "openai1m", "msturing10m", "uqv",
]

DATASET_LABEL = {
    "sift1m":     "SIFT1M",
    "sift10m":    "SIFT10M",
    "gist1m":     "GIST1M",
    "deep1m":     "DEEP1M",
    "deep10m":    "DEEP10M",
    "spacev10m":  "SPACEV10M",
    "msong":      "MSong",
    "enron":      "Enron",
    "openai1m":   "OpenAI1M",
    "msturing10m":"MSTuring10M",
    "uqv":        "UQ-V",
}

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

K_LIST = [1, 10, 20, 50, 100]
RECALL_TARGETS = ["r80", "r90", "r95", "r99"]
RECALL_TARGET_LABEL = {"r80": "0.80", "r90": "0.90", "r95": "0.95", "r99": "0.99"}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_all(results_dir):
    """Return {dataset: dict} for every results_<dataset>.json that exists."""
    out = {}
    for ds in DATASETS:
        path = os.path.join(results_dir, f"results_{ds}.json")
        if not os.path.isfile(path):
            print(f"[skip] missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            out[ds] = json.load(f)
    if not out:
        sys.exit(f"No results_*.json found in {results_dir}")
    return out


def save_fig(fig, out_dir, name, formats):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pareto_upper(points):
    """Given list of (recall, qps), keep only the Pareto-best by recall."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])
    out = []
    best_qps = -1.0
    for r, q in reversed(pts):
        if q > best_qps:
            out.append((r, q))
            best_qps = q
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Pareto QPS vs Recall (per dataset, per k)
# ---------------------------------------------------------------------------

def plot_pareto_single(ax, results, k, title=None, legend=True):
    key = f"recall_k{k}"
    section = results.get(key, {})
    plotted = False
    for idx in INDICES:
        rows = section.get(idx)
        if not rows:
            continue
        pts = [(r["recall"], r["qps"]) for r in rows
               if r.get("recall") is not None and r.get("qps")]
        if not pts:
            continue
        pareto = pareto_upper(pts)
        if not pareto:
            continue
        xs = [p[0] for p in pareto]
        ys = [p[1] for p in pareto]
        ax.plot(xs, ys,
                marker=INDEX_MARKER[idx], color=INDEX_COLOR[idx],
                label=idx, linewidth=1.6, markersize=5)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_yscale("log")
    ax.set_xlabel(f"Recall@{k}")
    ax.set_ylabel("QPS")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if title:
        ax.set_title(title, fontsize=10)
    if legend and plotted:
        ax.legend(fontsize=8, loc="lower left")


def plot_pareto_per_dataset(all_results, out_dir, formats):
    out = os.path.join(out_dir, "pareto")
    for ds, res in all_results.items():
        for k in K_LIST:
            if f"recall_k{k}" not in res:
                continue
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_pareto_single(
                ax, res, k,
                title=f"{DATASET_LABEL[ds]} — Recall@{k} vs QPS",
            )
            save_fig(fig, out, f"{ds}_k{k}", formats)


def plot_pareto_grid(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    n = len(datasets)
    if n == 0:
        return
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    for k in K_LIST:
        # only emit grid if at least one dataset has this k
        if not any(f"recall_k{k}" in all_results[ds] for ds in datasets):
            continue
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False
        )
        for i, ds in enumerate(datasets):
            ax = axes[i // ncols][i % ncols]
            plot_pareto_single(
                ax, all_results[ds], k,
                title=DATASET_LABEL[ds], legend=False,
            )
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        # one shared legend
        handles = [
            plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                       marker=INDEX_MARKER[idx], label=idx, linewidth=1.6)
            for idx in INDICES
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(INDICES), frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"QPS vs Recall@{k} — all datasets", fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        save_fig(fig, out_dir, f"pareto_grid_k{k}", formats)


# ---------------------------------------------------------------------------
# Construction cost
# ---------------------------------------------------------------------------

def _construction_table(all_results, field):
    """Returns dict[index] -> list of values aligned with DATASETS."""
    tab = {idx: [] for idx in INDICES}
    for ds in DATASETS:
        res = all_results.get(ds, {})
        cons = res.get("construction", {})
        for idx in INDICES:
            v = cons.get(idx, {}).get(field, np.nan)
            if v is None or v == -1 or (isinstance(v, str)):
                v = np.nan
            tab[idx].append(v)
    return tab


def plot_construction_bars(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    if not datasets:
        return

    for field, ylabel, name, log in [
        ("build_time_s", "Build time (s)",  "construction_build_time", True),
        ("size_mb",      "Index size (MB)", "construction_size",       True),
    ]:
        tab = _construction_table(all_results, field)
        # restrict to datasets we actually have
        x = np.arange(len(datasets))
        width = 0.16
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(datasets)), 4.2))
        for i, idx in enumerate(INDICES):
            ys = [tab[idx][DATASETS.index(ds)] for ds in datasets]
            offset = (i - (len(INDICES) - 1) / 2.0) * width
            ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
                   label=idx, color=INDEX_COLOR[idx])
            # annotate missing
            for xi, yi in zip(x + offset, ys):
                if np.isnan(yi):
                    ax.text(xi, 0, "×", ha="center", va="bottom",
                            color="gray", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.4)
        ax.legend(ncol=len(INDICES), fontsize=9, loc="upper left")
        save_fig(fig, out_dir, name, formats)


# ---------------------------------------------------------------------------
# Speedup vs HNSW32 at recall target — heatmap (k x dataset) per recall target
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    for k in (1, 10, 20):
        section_key = f"recall_k{k}"
        # rows = indices, cols = (dataset, recall_target)
        col_labels = []
        cells = []
        for ds in datasets:
            tar = all_results[ds].get("time_at_recall", {}).get(section_key)
            if not tar:
                continue
            for rt in RECALL_TARGETS:
                if rt not in tar:
                    continue
                col_labels.append(f"{DATASET_LABEL[ds]}\n@{RECALL_TARGET_LABEL[rt]}")
                col = []
                for idx in INDICES:
                    v = tar[rt].get(idx, {}).get("speedup_vs_HNSW32", np.nan)
                    if v is None:
                        v = np.nan
                    col.append(v)
                cells.append(col)
        if not cells:
            continue
        mat = np.array(cells).T  # shape: (n_indices, n_cols)

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(col_labels)),
                                        0.55 * len(INDICES) + 1.6))
        # log color scale, clipped
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mat = np.log10(np.where(mat > 0, mat, np.nan))
        vmax = np.nanmax(np.abs(log_mat)) if np.isfinite(log_mat).any() else 1.0
        im = ax.imshow(log_mat, aspect="auto", cmap="RdYlGn",
                       vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(INDICES)))
        ax.set_yticklabels(INDICES)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if abs(log_mat[i, j]) < 0.6 else "white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("log10(speedup vs HNSW32)")
        ax.set_title(f"Speedup vs HNSW32 at recall target — k={k}")
        save_fig(fig, out_dir, f"speedup_vs_hnsw32_k{k}", formats)


# ---------------------------------------------------------------------------
# Latency tail at recall target (p50/p95/p99/p999)
# ---------------------------------------------------------------------------

def plot_latency_tail(all_results, out_dir, formats, target="r95"):
    datasets = [ds for ds in DATASETS if ds in all_results]
    quantiles = ["p50", "p95", "p99", "p999"]
    fig, axes = plt.subplots(1, len(quantiles),
                             figsize=(4.0 * len(quantiles), 4.2),
                             sharey=False)
    for qi, q in enumerate(quantiles):
        ax = axes[qi]
        x = np.arange(len(datasets))
        width = 0.16
        for i, idx in enumerate(INDICES):
            ys = []
            for ds in datasets:
                lt = all_results[ds].get("latency_tail", {})
                v = lt.get(idx, {}).get(target, {}).get(q, np.nan)
                if v is None:
                    v = np.nan
                ys.append(v)
            offset = (i - (len(INDICES) - 1) / 2.0) * width
            ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
                   label=idx, color=INDEX_COLOR[idx])
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(f"{q} latency (ms)")
        ax.set_title(f"{q.upper()} @ recall {RECALL_TARGET_LABEL[target]}")
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
        if qi == 0:
            ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, out_dir, f"latency_tail_{target}", formats)


# ---------------------------------------------------------------------------
# MRE at recall targets
# ---------------------------------------------------------------------------

MRE_OUTLIER_THRESHOLD = 2.0   # MRE values above this are "broken" (zero-distance GT etc.)


def plot_mre(all_results, out_dir, formats, field="mre", name="mre_at_recall",
             stat_label="MRE"):
    """MRE bars with outlier-aware clipping: values > MRE_OUTLIER_THRESHOLD are
    plotted at the threshold and tagged with a red asterisk so the figure stays
    readable even on datasets with duplicate / zero-distance ground-truth pairs
    (e.g. spacev10m where MRE blows up to 1e10)."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    targets = ["r90", "r95", "r99"]
    fig, axes = plt.subplots(1, len(targets),
                             figsize=(4.5 * len(targets), 4.2),
                             sharey=True)
    for ti, t in enumerate(targets):
        ax = axes[ti]
        x = np.arange(len(datasets))
        width = 0.16
        for i, idx in enumerate(INDICES):
            ys, broken = [], []
            for ds in datasets:
                v = all_results[ds].get("mre", {}).get(idx, {}).get(t, {}).get(field, np.nan)
                if v is None or not np.isfinite(v):
                    ys.append(np.nan); broken.append(False); continue
                if v > MRE_OUTLIER_THRESHOLD:
                    ys.append(MRE_OUTLIER_THRESHOLD - 1.0); broken.append(True)
                else:
                    ys.append(max(v - 1.0, 0)); broken.append(False)
            offset = (i - (len(INDICES) - 1) / 2.0) * width
            ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
                   label=idx, color=INDEX_COLOR[idx])
            for xi, bad in zip(x + offset, broken):
                if bad:
                    ax.text(xi, MRE_OUTLIER_THRESHOLD - 1.0, "*",
                            ha="center", va="bottom", color="red", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{stat_label} − 1")
        ax.set_title(f"{stat_label} at recall {RECALL_TARGET_LABEL[t]}")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
        if ti == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, name, formats)


# ---------------------------------------------------------------------------
# Cumulative cost (SuCo Fig 13): build_time + n_queries · ms_per_query at fixed recall
# ---------------------------------------------------------------------------

def plot_cumulative_cost(all_results, out_dir, formats, k=10, recall_target="r95"):
    """Total wall-clock cost (s) = build_time + nq · ms_per_query / 1000 at fixed recall,
    plotted as a function of nq. Crossover points show when each index amortizes its build."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    section_key = f"recall_k{k}"
    nq_grid = np.logspace(2, 7, 40)  # 100 .. 10M queries
    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[di // ncols][di % ncols]
        cons = all_results[ds].get("construction", {})
        tar = all_results[ds].get("time_at_recall", {}).get(section_key, {}).get(recall_target, {})
        any_data = False
        for idx in INDICES:
            bt = cons.get(idx, {}).get("build_time_s")
            ms = tar.get(idx, {}).get("ms_per_query")
            if bt is None or bt < 0 or ms is None:
                continue
            cost = bt + nq_grid * ms / 1000.0
            ax.plot(nq_grid, cost, color=INDEX_COLOR[idx],
                    marker=INDEX_MARKER[idx], markevery=8, markersize=4,
                    label=idx, linewidth=1.4)
            any_data = True
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(DATASET_LABEL[ds], fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        if di // ncols == nrows - 1:
            ax.set_xlabel("number of queries")
        if di % ncols == 0:
            ax.set_ylabel("cumulative cost (s)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                          marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
               for idx in INDICES]
    fig.legend(handles=handles, loc="lower center", ncol=len(INDICES),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Cumulative cost = build + nq·ms/q  at recall@{k} ≥ "
                 f"{RECALL_TARGET_LABEL[recall_target]}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, out_dir, f"cumulative_cost_k{k}_{recall_target}", formats)


# ---------------------------------------------------------------------------
# k sensitivity: ms/query at recall>=target across k ∈ {1,10,20,50,100}
# ---------------------------------------------------------------------------

def plot_k_sensitivity(all_results, out_dir, formats, recall_target="r95"):
    """For each index and dataset, ms/query at fixed recall as a function of k."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    n = len(datasets); ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[di // ncols][di % ncols]
        any_data = False
        for idx in INDICES:
            xs, ys = [], []
            for k in K_LIST:
                tar = all_results[ds].get("time_at_recall", {}).get(
                    f"recall_k{k}", {}).get(recall_target, {})
                ms = tar.get(idx, {}).get("ms_per_query")
                if ms is not None:
                    xs.append(k); ys.append(ms)
            if xs:
                ax.plot(xs, ys, color=INDEX_COLOR[idx],
                        marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
                any_data = True
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(K_LIST); ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_title(DATASET_LABEL[ds], fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        if di // ncols == nrows - 1:
            ax.set_xlabel("k")
        if di % ncols == 0:
            ax.set_ylabel("ms / query")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                          marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
               for idx in INDICES]
    fig.legend(handles=handles, loc="lower center", ncol=len(INDICES),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"k sensitivity: ms/query at recall ≥ "
                 f"{RECALL_TARGET_LABEL[recall_target]}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, out_dir, f"k_sensitivity_{recall_target}", formats)


# ---------------------------------------------------------------------------
# Cold latency absolute values (not just ratio)
# ---------------------------------------------------------------------------

def plot_cold_latency_abs(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.0 * len(datasets) + 4), 4.4),
                             sharey=False)
    for ax, field, title in [
        (axes[0], "cold_mean_ms", "Cold-cache mean latency (ms)"),
        (axes[1], "warm_mean_ms", "Warm-cache mean latency (ms)"),
    ]:
        x = np.arange(len(datasets))
        width = 0.16
        for i, idx in enumerate(INDICES):
            ys = [all_results[d].get("cold_warm", {}).get(idx, {}).get(field, np.nan)
                  for d in datasets]
            ys = [np.nan if v is None else v for v in ys]
            offset = (i - (len(INDICES) - 1) / 2.0) * width
            ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
                   label=idx, color=INDEX_COLOR[idx])
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title); ax.set_ylabel("ms")
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    axes[0].legend(ncol=len(INDICES), fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_dir, "cold_warm_absolute", formats)


# ---------------------------------------------------------------------------
# Construction tradeoff scatter: build_time vs index size, bubble = mean_recall@10@r95
# ---------------------------------------------------------------------------

def plot_build_vs_size(all_results, out_dir, formats):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    plotted = False
    for idx in INDICES:
        xs, ys, ann = [], [], []
        for ds in DATASETS:
            if ds not in all_results:
                continue
            cons = all_results[ds].get("construction", {}).get(idx, {})
            bt = cons.get("build_time_s"); sz = cons.get("size_mb")
            if bt is None or sz is None or bt < 0 or sz < 0:
                continue
            xs.append(sz); ys.append(bt); ann.append(ds)
        if not xs:
            continue
        ax.scatter(xs, ys, s=70, color=INDEX_COLOR[idx],
                   marker=INDEX_MARKER[idx], label=idx,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        plotted = True
        for x, y, a in zip(xs, ys, ann):
            ax.annotate(DATASET_LABEL[a], (x, y), fontsize=6,
                        color="dimgray", alpha=0.7,
                        xytext=(4, 2), textcoords="offset points")
    if not plotted:
        return
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Index size (MB, log)")
    ax.set_ylabel("Build time (s, log)")
    ax.set_title("Construction tradeoff: build time vs serialized size")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_dir, "build_vs_size", formats)


# ---------------------------------------------------------------------------
# Dataset-difficulty scatter: LID vs speedup-vs-HNSW32 (recall@10 r95)
# ---------------------------------------------------------------------------

def plot_lid_vs_speedup(all_results, out_dir, formats, k=10, recall_target="r95"):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    section_key = f"recall_k{k}"
    plotted = False
    for idx in INDICES:
        xs, ys, ann = [], [], []
        for ds in DATASETS:
            if ds not in all_results:
                continue
            lid = all_results[ds].get("features", {}).get("lid_mle")
            sp = (all_results[ds].get("time_at_recall", {})
                  .get(section_key, {}).get(recall_target, {})
                  .get(idx, {}).get("speedup_vs_HNSW32"))
            if lid is None or sp is None:
                continue
            xs.append(lid); ys.append(sp); ann.append(ds)
        if not xs:
            continue
        ax.scatter(xs, ys, s=70, color=INDEX_COLOR[idx],
                   marker=INDEX_MARKER[idx], label=idx,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        plotted = True
        for x, y, a in zip(xs, ys, ann):
            ax.annotate(DATASET_LABEL[a], (x, y), fontsize=6,
                        color="dimgray", alpha=0.7,
                        xytext=(4, 2), textcoords="offset points")
    if not plotted:
        return
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("LID (MLE)")
    ax.set_ylabel(f"speedup vs HNSW32 @ recall@{k}≥"
                  f"{RECALL_TARGET_LABEL[recall_target]}")
    ax.set_title("Dataset difficulty (LID) vs achieved speedup")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_dir, f"lid_vs_speedup_k{k}_{recall_target}", formats)


# ---------------------------------------------------------------------------
# Recall-vs-time grid: ms/query (log-y) vs recall — same data as Pareto, easier read
# ---------------------------------------------------------------------------

def plot_recall_vs_time_grid(all_results, out_dir, formats, k=10):
    section_key = f"recall_k{k}"
    datasets = [ds for ds in DATASETS if ds in all_results]
    n = len(datasets); ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[di // ncols][di % ncols]
        any_data = False
        for idx in INDICES:
            rows = all_results[ds].get(section_key, {}).get(idx)
            if not rows:
                continue
            rs = sorted([(r["recall"], r["ms_per_query"]) for r in rows
                        if r.get("recall") is not None and r.get("ms_per_query")],
                       key=lambda p: p[0])
            if not rs:
                continue
            xs, ys = zip(*rs)
            ax.plot(xs, ys, marker=INDEX_MARKER[idx], color=INDEX_COLOR[idx],
                    label=idx, linewidth=1.4, markersize=4)
            any_data = True
        ax.set_yscale("log")
        ax.set_title(DATASET_LABEL[ds], fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        if di // ncols == nrows - 1:
            ax.set_xlabel(f"recall@{k}")
        if di % ncols == 0:
            ax.set_ylabel("ms / query")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                          marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
               for idx in INDICES]
    fig.legend(handles=handles, loc="lower center", ncol=len(INDICES),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"ms/query vs recall@{k} (log-y)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, out_dir, f"recall_vs_time_grid_k{k}", formats)


# ---------------------------------------------------------------------------
# Thread-scaling: QPS, speedup-vs-1-thread, parallel efficiency
# ---------------------------------------------------------------------------

def _thread_data(node):
    """Pull (sorted-threads list, qps list, speedup list, eff list) from a
    by_threads dict, dropping entries that errored out."""
    bt = (node or {}).get("by_threads", {}) or {}
    items = []
    for k, v in bt.items():
        try:
            t = int(k)
        except Exception:
            continue
        if not isinstance(v, dict) or v.get("qps") is None:
            continue
        items.append((t, v))
    items.sort(key=lambda x: x[0])
    if not items:
        return [], [], [], []
    ts   = [t for t, _ in items]
    qps  = [v.get("qps")                  for _, v in items]
    spd  = [v.get("speedup_vs_t1")        for _, v in items]
    eff  = [v.get("parallel_efficiency")  for _, v in items]
    return ts, qps, spd, eff


def plot_thread_scaling_grid(all_results, out_dir, formats, recall_target="r95"):
    """Grid of datasets: QPS vs threads (log-log) per index, with ideal-linear ref."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    n = len(datasets); ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    any_global = False
    for di, ds in enumerate(datasets):
        ax = axes[di // ncols][di % ncols]
        ts_node = all_results[ds].get("thread_scaling", {})
        any_data = False
        max_t = 1
        for idx in INDICES:
            ts, qps, _, _ = _thread_data(ts_node.get(idx, {}).get(recall_target))
            if not ts:
                continue
            ax.plot(ts, qps, marker=INDEX_MARKER[idx], color=INDEX_COLOR[idx],
                    label=idx, linewidth=1.4, markersize=5)
            any_data = True
            max_t = max(max_t, ts[-1])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(DATASET_LABEL[ds], fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        else:
            any_global = True
            xt = [1, 2, 4, 8, 16, 32, 64]
            xt = [t for t in xt if t <= max_t]
            ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt])
        if di // ncols == nrows - 1:
            ax.set_xlabel("threads")
        if di % ncols == 0:
            ax.set_ylabel("QPS")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if not any_global:
        plt.close(fig); return
    handles = [plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                          marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
               for idx in INDICES]
    fig.legend(handles=handles, loc="lower center", ncol=len(INDICES),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Thread-scaling: QPS vs threads "
                 f"@ recall ≥ {RECALL_TARGET_LABEL[recall_target]} (k=10)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, out_dir, f"thread_scaling_qps_{recall_target}", formats)


def plot_thread_scaling_speedup(all_results, out_dir, formats, recall_target="r95"):
    """Grid of datasets: speedup_vs_t1 vs threads, with y=x ideal-linear reference."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    n = len(datasets); ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    any_global = False
    for di, ds in enumerate(datasets):
        ax = axes[di // ncols][di % ncols]
        ts_node = all_results[ds].get("thread_scaling", {})
        any_data = False
        max_t = 1
        for idx in INDICES:
            ts, _, spd, _ = _thread_data(ts_node.get(idx, {}).get(recall_target))
            spd_clean = [s for s in spd if s is not None]
            if not ts or not spd_clean:
                continue
            ax.plot(ts, spd, marker=INDEX_MARKER[idx], color=INDEX_COLOR[idx],
                    label=idx, linewidth=1.4, markersize=5)
            any_data = True
            max_t = max(max_t, ts[-1])
        if any_data:
            any_global = True
            ideal = [1, 2, 4, 8, 16, 32, 64]
            ideal = [t for t in ideal if t <= max_t]
            ax.plot(ideal, ideal, color="black", linewidth=0.8,
                    linestyle="--", alpha=0.6, label="ideal (linear)")
            ax.set_xticks(ideal); ax.set_xticklabels([str(t) for t in ideal])
            ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_title(DATASET_LABEL[ds], fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        if di // ncols == nrows - 1:
            ax.set_xlabel("threads")
        if di % ncols == 0:
            ax.set_ylabel("speedup vs 1 thread")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if not any_global:
        plt.close(fig); return
    handles = [plt.Line2D([0], [0], color=INDEX_COLOR[idx],
                          marker=INDEX_MARKER[idx], label=idx, linewidth=1.4)
               for idx in INDICES]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              linewidth=0.8, label="ideal"))
    fig.legend(handles=handles, loc="lower center", ncol=len(INDICES) + 1,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Thread-scaling: speedup vs 1 thread "
                 f"@ recall ≥ {RECALL_TARGET_LABEL[recall_target]} (k=10)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, out_dir, f"thread_scaling_speedup_{recall_target}", formats)


def plot_thread_scaling_efficiency_summary(all_results, out_dir, formats,
                                           recall_target="r95", at_threads=None):
    """Bar chart: parallel efficiency at the highest available thread count
    (default = max across data) per dataset and index. Efficiency = speedup / threads."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    # Determine target thread count automatically if not supplied.
    if at_threads is None:
        seen = set()
        for ds in datasets:
            ts_node = all_results[ds].get("thread_scaling", {})
            for idx in INDICES:
                ts, _, _, _ = _thread_data(ts_node.get(idx, {}).get(recall_target))
                seen.update(ts)
        if not seen:
            return
        at_threads = max(seen)

    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(datasets) + 1), 4.4))
    x = np.arange(len(datasets))
    width = 0.16
    any_data = False
    for i, idx in enumerate(INDICES):
        ys = []
        for ds in datasets:
            node = (all_results[ds].get("thread_scaling", {})
                    .get(idx, {}).get(recall_target))
            ts, _, _, eff = _thread_data(node)
            ev = np.nan
            if ts:
                # find efficiency at exactly at_threads, else closest available
                if at_threads in ts:
                    ev = eff[ts.index(at_threads)]
                else:
                    j = min(range(len(ts)), key=lambda jj: abs(ts[jj] - at_threads))
                    ev = eff[j]
            if ev is None:
                ev = np.nan
            ys.append(ev)
            if not np.isnan(ev):
                any_data = True
        offset = (i - (len(INDICES) - 1) / 2.0) * width
        ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
               label=idx, color=INDEX_COLOR[idx])
    if not any_data:
        plt.close(fig); return
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("parallel efficiency  (speedup / threads)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Parallel efficiency at {at_threads} threads "
                 f"@ recall ≥ {RECALL_TARGET_LABEL[recall_target]} (k=10)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=len(INDICES), fontsize=9, loc="upper left")
    save_fig(fig, out_dir, f"thread_scaling_efficiency_t{at_threads}_{recall_target}",
             formats)


def table_thread_scaling(all_results, out_dir):
    """One table per recall target: dataset × index × thread count → QPS, eff."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    # Discover the union of thread counts seen across datasets.
    for tgt in ("r90", "r95"):
        thread_set = set()
        for ds in datasets:
            ts_node = all_results[ds].get("thread_scaling", {})
            for idx in INDICES:
                ts, _, _, _ = _thread_data(ts_node.get(idx, {}).get(tgt))
                thread_set.update(ts)
        if not thread_set:
            continue
        thread_list = sorted(thread_set)
        header = ["Dataset", "Index"]
        for t in thread_list:
            header += [f"qps@t{t}", f"eff@t{t}"]
        rows = []
        for ds in datasets:
            ts_node = all_results[ds].get("thread_scaling", {})
            for idx in INDICES:
                ts, qps, _, eff = _thread_data(ts_node.get(idx, {}).get(tgt))
                if not ts:
                    continue
                lookup = {t: (q, e) for t, q, e in zip(ts, qps, eff)}
                row = [DATASET_LABEL[ds], idx]
                for t in thread_list:
                    q, e = lookup.get(t, (None, None))
                    row += [_fmt(q, 1), _fmt(e, 3)]
                rows.append(row)
        if rows:
            write_table(out_dir, f"thread_scaling_{tgt}", header, rows,
                        caption=f"QPS and parallel efficiency at recall "
                                f"$\\geq$ {RECALL_TARGET_LABEL[tgt]} (k=10) "
                                f"as faiss.omp\\_set\\_num\\_threads is swept.",
                        label=f"thread-scaling-{tgt}")


# ---------------------------------------------------------------------------
# Robustness boxplot (per-query recall@20 distribution)
# ---------------------------------------------------------------------------

def plot_robustness_box(all_results, out_dir, formats):
    """We only have summary stats (min/q25/median/q75/max) — synthesize whisker
    plot from those quantiles per dataset/index."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(datasets) + 2), 5.0))
    n_idx = len(INDICES)
    width = 0.8 / n_idx
    positions = np.arange(len(datasets))
    for i, idx in enumerate(INDICES):
        for j, ds in enumerate(datasets):
            row = all_results[ds].get("robustness", {}).get(idx)
            if not row:
                continue
            x_center = positions[j] + (i - (n_idx - 1) / 2.0) * width
            box_lo, box_hi = row["q25_recall"], row["q75_recall"]
            wh_lo, wh_hi = row["min_recall"], row["max_recall"]
            med = row["median_recall"]
            color = INDEX_COLOR[idx]
            # whiskers
            ax.plot([x_center, x_center], [wh_lo, wh_hi],
                    color=color, linewidth=1.0, alpha=0.7)
            # box
            ax.add_patch(plt.Rectangle(
                (x_center - width / 2.5, box_lo), width / 1.25, box_hi - box_lo,
                facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.7))
            # median line
            ax.plot([x_center - width / 2.5, x_center + width / 2.5],
                    [med, med], color="black", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("per-query recall@20")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=INDEX_COLOR[idx], label=idx, alpha=0.7)
        for idx in INDICES
    ]
    ax.legend(handles=handles, ncol=len(INDICES), fontsize=9,
              loc="lower right")
    ax.set_title("Robustness — per-query recall@20 distribution "
                 "(min · 25–75% · median · max)")
    save_fig(fig, out_dir, "robustness_box_k20", formats)


# ---------------------------------------------------------------------------
# Hard vs Easy mean recall
# ---------------------------------------------------------------------------

def plot_hard_easy(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.0 * len(datasets) + 4), 4.2),
                             sharey=True)
    for split, ax in zip(("easy", "hard"), axes):
        x = np.arange(len(datasets))
        width = 0.16
        for i, idx in enumerate(INDICES):
            ys = []
            for ds in datasets:
                row = all_results[ds].get("hard_robustness", {}).get(idx, {})
                ys.append(row.get(split, {}).get("mean_recall", np.nan))
            offset = (i - (len(INDICES) - 1) / 2.0) * width
            ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
                   label=idx, color=INDEX_COLOR[idx])
        ax.set_title(f"{split.capitalize()} 10% — mean recall@20")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("mean recall@20")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, out_dir, "hard_easy_recall", formats)


# ---------------------------------------------------------------------------
# Cold/warm latency ratio
# ---------------------------------------------------------------------------

def plot_cold_warm(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(datasets) + 1), 4.2))
    x = np.arange(len(datasets))
    width = 0.16
    for i, idx in enumerate(INDICES):
        ys = []
        for ds in datasets:
            v = all_results[ds].get("cold_warm", {}).get(idx, {}).get(
                "cold_warm_ratio", np.nan)
            if v is None:
                v = np.nan
            ys.append(v)
        offset = (i - (len(INDICES) - 1) / 2.0) * width
        ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
               label=idx, color=INDEX_COLOR[idx])
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("cold / warm latency ratio")
    ax.set_title("Cold-cache vs warm-cache mean latency (≥1 = cold is slower)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=len(INDICES), fontsize=9, loc="upper left")
    save_fig(fig, out_dir, "cold_warm_ratio", formats)


# ---------------------------------------------------------------------------
# Unseen-base robustness
# ---------------------------------------------------------------------------

def plot_unseen_robustness(all_results, out_dir, formats):
    datasets = [ds for ds in DATASETS if ds in all_results]
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(datasets) + 1), 4.2))
    x = np.arange(len(datasets))
    width = 0.16
    for i, idx in enumerate(INDICES):
        ys = []
        for ds in datasets:
            ur = all_results[ds].get("unseen_robustness", {}).get("per_index", {})
            ys.append(ur.get(idx, {}).get("mean_recall", np.nan))
        offset = (i - (len(INDICES) - 1) / 2.0) * width
        ax.bar(x + offset, np.where(np.isnan(ys), 0, ys), width,
               label=idx, color=INDEX_COLOR[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean recall@20 on held-out queries")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_title("Unseen-base robustness (rebuild without queries)")
    ax.legend(ncol=len(INDICES), fontsize=9, loc="lower right")
    save_fig(fig, out_dir, "unseen_robustness", formats)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(v, digits=2):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        if not math.isfinite(v):
            return "—"
    except TypeError:
        return str(v)
    if v == 0:
        return "0"
    av = abs(v)
    # very small: scientific
    if av < 10 ** (-digits):
        return f"{v:.{digits}g}"
    # very large: thousands separators, no decimals
    if av >= 10000:
        return f"{v:,.0f}"
    # mid-range: fixed decimals, drop trailing zeros
    s = f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def write_table(out_dir, name, header, rows, caption=None, label=None,
                col_align=None):
    """Write rows to both <out_dir>/tables/<name>.csv and .tex."""
    tdir = os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)

    # CSV
    csv_path = os.path.join(tdir, f"{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    # LaTeX (booktabs)
    if col_align is None:
        col_align = "l" + "r" * (len(header) - 1)
    tex_path = os.path.join(tdir, f"{name}.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{tab:{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_align}}}\n\\toprule\n")
        def _tex(c):
            s = str(c)
            # in LaTeX, prefer thin-space thousand separators over commas
            return s.replace(",", "\\,")
        f.write(" & ".join(_tex(h) for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(_tex(c) for c in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_construction(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset"]
    for idx in INDICES:
        header += [f"{idx} build (s)", f"{idx} size (MB)", f"{idx} mem (MB)"]
    rows = []
    for ds in datasets:
        cons = all_results[ds].get("construction", {})
        row = [DATASET_LABEL[ds]]
        for idx in INDICES:
            c = cons.get(idx, {})
            failed = c.get("build_failed")
            if failed:
                row += ["FAIL", "FAIL", "FAIL"]
                continue
            bt = c.get("build_time_s")
            sz = c.get("size_mb")
            mm = c.get("memory_mb")
            row += [_fmt(bt, 2), _fmt(sz, 1),
                    _fmt(mm, 1) if (mm is not None and mm > 0) else "—"]
        rows.append(row)
    write_table(out_dir, "construction", header, rows,
                caption="Construction cost: build time, on-disk size, "
                        "and peak resident-memory delta during build "
                        "(— = not reported, FAIL = build skipped).",
                label="construction")


def table_dataset_features(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset", "n", "d", "nq", "LID (MLE)",
              "pdist mean", "pdist std", "k-means inertia ratio (k=16)"]
    rows = []
    for ds in datasets:
        f = all_results[ds].get("features", {})
        rows.append([
            DATASET_LABEL[ds],
            f.get("n", "—"),
            f.get("d", "—"),
            f.get("nq", "—"),
            _fmt(f.get("lid_mle"), 2),
            _fmt(f.get("pdist_mean"), 3),
            _fmt(f.get("pdist_std"), 3),
            _fmt(f.get("kmeans_inertia_ratio_16"), 3),
        ])
    write_table(out_dir, "dataset_features", header, rows,
                caption="Dataset statistics used by the router as features.",
                label="dataset-features")


def table_speedup(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    for k in (1, 10, 20):
        section_key = f"recall_k{k}"
        header = ["Dataset", "Index"] + [
            f"speedup@{RECALL_TARGET_LABEL[t]}" for t in RECALL_TARGETS]
        rows = []
        any_data = False
        for ds in datasets:
            tar = all_results[ds].get("time_at_recall", {}).get(section_key)
            if not tar:
                continue
            for idx in INDICES:
                row = [DATASET_LABEL[ds], idx]
                for rt in RECALL_TARGETS:
                    v = tar.get(rt, {}).get(idx, {}).get(
                        "speedup_vs_HNSW32")
                    row.append(_fmt(v, 3))
                    if v is not None:
                        any_data = True
                rows.append(row)
        if not any_data:
            continue
        write_table(out_dir, f"speedup_vs_hnsw32_k{k}", header, rows,
                    caption=f"Speedup vs HNSW32 at recall@{k} target "
                            "(>1 means faster than HNSW32 at the same recall).",
                    label=f"speedup-k{k}")


def table_latency_tail(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    quantiles = ["p50", "p95", "p99", "p999"]
    for tgt in ("r90", "r95", "r99"):
        header = ["Dataset", "Index"] + [f"{q} (ms)" for q in quantiles] + [
            "achieved recall"]
        rows = []
        for ds in datasets:
            lt = all_results[ds].get("latency_tail", {})
            for idx in INDICES:
                cell = lt.get(idx, {}).get(tgt)
                if not cell:
                    continue
                row = [DATASET_LABEL[ds], idx]
                for q in quantiles:
                    row.append(_fmt(cell.get(q), 3))
                row.append(_fmt(cell.get("achieved_recall"), 3))
                rows.append(row)
        if not rows:
            continue
        write_table(out_dir, f"latency_tail_{tgt}", header, rows,
                    caption=f"Per-query latency tail (ms) at recall target "
                            f"{RECALL_TARGET_LABEL[tgt]}.",
                    label=f"latency-tail-{tgt}")


def table_mre(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    targets = ["r90", "r95", "r99"]
    header = ["Dataset", "Index"]
    for t in targets:
        header += [f"mean@{RECALL_TARGET_LABEL[t]}",
                   f"med@{RECALL_TARGET_LABEL[t]}"]
    rows = []
    for ds in datasets:
        mre = all_results[ds].get("mre", {})
        for idx in INDICES:
            row = [DATASET_LABEL[ds], idx]
            has = False
            for t in targets:
                cell = mre.get(idx, {}).get(t, {})
                vm = cell.get("mre")
                vmd = cell.get("mre_median")
                row.append(_fmt(vm, 4))
                row.append(_fmt(vmd, 4))
                if vm is not None or vmd is not None:
                    has = True
            if has:
                rows.append(row)
    if rows:
        write_table(out_dir, "mre_at_recall", header, rows,
                    caption="Mean and median relative error (MRE) at recall "
                            "targets (1.0 = exact distances; lower is better). "
                            "Median is robust to duplicate / zero-distance "
                            "ground-truth pairs that inflate the mean.",
                    label="mre")


def table_robustness(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset", "Index", "mean", "min", "q25",
              "median", "q75", "max", "ms/query"]
    rows = []
    for ds in datasets:
        rob = all_results[ds].get("robustness", {})
        for idx in INDICES:
            r = rob.get(idx)
            if not r:
                continue
            rows.append([
                DATASET_LABEL[ds], idx,
                _fmt(r.get("mean_recall"), 3),
                _fmt(r.get("min_recall"), 3),
                _fmt(r.get("q25_recall"), 3),
                _fmt(r.get("median_recall"), 3),
                _fmt(r.get("q75_recall"), 3),
                _fmt(r.get("max_recall"), 3),
                _fmt(r.get("ms_per_query"), 3),
            ])
    if rows:
        write_table(out_dir, "robustness_k20", header, rows,
                    caption="Per-query recall@20 distribution at the "
                            "best operating point (≈recall 0.95).",
                    label="robustness")


def table_hard_easy(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset", "Index", "easy mean", "easy median",
              "hard mean", "hard median", "hard p10"]
    rows = []
    for ds in datasets:
        hr = all_results[ds].get("hard_robustness", {})
        for idx in INDICES:
            r = hr.get(idx)
            if not r:
                continue
            e = r.get("easy", {})
            h = r.get("hard", {})
            rows.append([
                DATASET_LABEL[ds], idx,
                _fmt(e.get("mean_recall"), 3),
                _fmt(e.get("median_recall"), 3),
                _fmt(h.get("mean_recall"), 3),
                _fmt(h.get("median_recall"), 3),
                _fmt(h.get("p10_recall"), 3),
            ])
    if rows:
        write_table(out_dir, "hard_easy_recall", header, rows,
                    caption="Recall on the 10\\% easiest vs 10\\% hardest "
                            "queries (stratified by GT $k$-th distance).",
                    label="hard-easy")


def table_cold_warm(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset", "Index", "cold mean (ms)", "cold p95 (ms)",
              "warm mean (ms)", "warm p95 (ms)", "cold/warm"]
    rows = []
    for ds in datasets:
        cw = all_results[ds].get("cold_warm", {})
        for idx in INDICES:
            r = cw.get(idx)
            if not r:
                continue
            rows.append([
                DATASET_LABEL[ds], idx,
                _fmt(r.get("cold_mean_ms"), 3),
                _fmt(r.get("cold_p95_ms"), 3),
                _fmt(r.get("warm_mean_ms"), 3),
                _fmt(r.get("warm_p95_ms"), 3),
                _fmt(r.get("cold_warm_ratio"), 2),
            ])
    if rows:
        write_table(out_dir, "cold_warm", header, rows,
                    caption="Cold-cache vs warm-cache mean and p95 latency.",
                    label="cold-warm")


def table_unseen_robustness(all_results, out_dir):
    datasets = [ds for ds in DATASETS if ds in all_results]
    header = ["Dataset", "Index", "mean", "median", "p10", "p25", "min"]
    rows = []
    for ds in datasets:
        ur = all_results[ds].get("unseen_robustness", {}).get("per_index", {})
        for idx in INDICES:
            r = ur.get(idx)
            if not r:
                continue
            rows.append([
                DATASET_LABEL[ds], idx,
                _fmt(r.get("mean_recall"), 3),
                _fmt(r.get("median_recall"), 3),
                _fmt(r.get("p10_recall"), 3),
                _fmt(r.get("p25_recall"), 3),
                _fmt(r.get("min_recall"), 3),
            ])
    if rows:
        write_table(out_dir, "unseen_robustness", header, rows,
                    caption="Recall@20 on held-out base vectors after rebuilding "
                            "the index without them.",
                    label="unseen")


def table_recall_at_qps(all_results, out_dir, qps_targets=(1000, 10000)):
    """For each (dataset, index, k), the highest recall achievable at >=qps."""
    datasets = [ds for ds in DATASETS if ds in all_results]
    for k in (1, 10, 20):
        section = f"recall_k{k}"
        header = ["Dataset", "Index"] + [
            f"recall@{int(q)} qps" for q in qps_targets]
        rows = []
        for ds in datasets:
            sec = all_results[ds].get(section, {})
            for idx in INDICES:
                pts = sec.get(idx)
                if not pts:
                    continue
                row = [DATASET_LABEL[ds], idx]
                has = False
                for q_thr in qps_targets:
                    feasible = [p for p in pts
                                if (p.get("qps") or 0) >= q_thr
                                and p.get("recall") is not None]
                    best = max((p["recall"] for p in feasible), default=None)
                    if best is not None:
                        has = True
                    row.append(_fmt(best, 3))
                if has:
                    rows.append(row)
        if rows:
            write_table(out_dir, f"recall_at_qps_targets_k{k}", header, rows,
                        caption=f"Highest achieved recall@{k} at "
                                f"throughput \\(\\geq Q\\) qps.",
                        label=f"recall-at-qps-k{k}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot all router-paper benchmark figures.")
    ap.add_argument("--results-dir", default="benchs/results_router",
                    help="directory with results_<dataset>.json files")
    ap.add_argument("--out-dir", default="benchs/figures_router",
                    help="output directory for figures")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"],
                    help="output formats")
    args = ap.parse_args()

    all_results = load_all(args.results_dir)
    print(f"Loaded {len(all_results)} datasets: "
          f"{', '.join(all_results.keys())}")

    # Pareto curves
    plot_pareto_per_dataset(all_results, args.out_dir, args.formats)
    plot_pareto_grid(all_results, args.out_dir, args.formats)

    # Construction
    plot_construction_bars(all_results, args.out_dir, args.formats)

    # Speedup heatmaps
    plot_speedup_heatmap(all_results, args.out_dir, args.formats)

    # Latency tail (default at r95)
    for tgt in ("r90", "r95", "r99"):
        plot_latency_tail(all_results, args.out_dir, args.formats, target=tgt)

    # MRE — both mean (legacy) and median (outlier-robust)
    plot_mre(all_results, args.out_dir, args.formats,
             field="mre", name="mre_at_recall", stat_label="MRE")
    #plot_mre(all_results, args.out_dir, args.formats,
    #         field="mre_median", name="mre_median_at_recall",
    #         stat_label="MRE (median)")

    # Robustness
    plot_robustness_box(all_results, args.out_dir, args.formats)
    plot_hard_easy(all_results, args.out_dir, args.formats)

    # Cold/warm + unseen
    plot_cold_warm(all_results, args.out_dir, args.formats)
    plot_cold_latency_abs(all_results, args.out_dir, args.formats)
    plot_unseen_robustness(all_results, args.out_dir, args.formats)

    # New analytical figures
    plot_cumulative_cost(all_results, args.out_dir, args.formats, k=10, recall_target="r95")
    plot_cumulative_cost(all_results, args.out_dir, args.formats, k=10, recall_target="r99")
    plot_k_sensitivity(all_results, args.out_dir, args.formats, recall_target="r95")
    plot_k_sensitivity(all_results, args.out_dir, args.formats, recall_target="r99")
    plot_build_vs_size(all_results, args.out_dir, args.formats)
    plot_lid_vs_speedup(all_results, args.out_dir, args.formats, k=10, recall_target="r95")
    plot_recall_vs_time_grid(all_results, args.out_dir, args.formats, k=1)
    plot_recall_vs_time_grid(all_results, args.out_dir, args.formats, k=10)
    plot_recall_vs_time_grid(all_results, args.out_dir, args.formats, k=20)

    # Thread-scaling figures (one per recall target)
    for tgt in ("r90", "r95"):
        plot_thread_scaling_grid(all_results, args.out_dir, args.formats, recall_target=tgt)
        plot_thread_scaling_speedup(all_results, args.out_dir, args.formats, recall_target=tgt)
        plot_thread_scaling_efficiency_summary(
            all_results, args.out_dir, args.formats, recall_target=tgt)

    # Tables (csv + tex)
    table_dataset_features(all_results, args.out_dir)
    table_construction(all_results, args.out_dir)
    table_speedup(all_results, args.out_dir)
    table_latency_tail(all_results, args.out_dir)
    table_mre(all_results, args.out_dir)
    table_robustness(all_results, args.out_dir)
    table_hard_easy(all_results, args.out_dir)
    table_cold_warm(all_results, args.out_dir)
    table_unseen_robustness(all_results, args.out_dir)
    table_recall_at_qps(all_results, args.out_dir)
    table_thread_scaling(all_results, args.out_dir)

    print(f"figures written to {args.out_dir}/")
    print(f"tables  written to {os.path.join(args.out_dir, 'tables')}/")


if __name__ == "__main__":
    main()
