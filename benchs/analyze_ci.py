#!/usr/bin/env python3
"""
benchs/analyze_ci.py

Confidence intervals for matched-recall throughput ratios.

Motivation.  The cross-algorithm comparison reports ratios of two independently
measured quantities, many of which lie within a few percent of unity ("0.60-1.10x
HNSW32", "within 3% of each other").  A point estimate is not interpretable at
that scale: a ratio of two measurements each carrying a few percent of run-to-run
variance carries the sum of those variances, which can be the same size as the
effect.  This script re-derives every such ratio from the raw per-run timings
already stored in results_<dataset>.json and attaches an interval, so that
claims of near-parity can be separated from claims of measured difference.

Method.  For each (dataset, index, k) the sweep stores, at every parameter point,
the recall and the wall-clock of each of N_RUNS repetitions (`run_times_s`).  We

  1. form one recall-time curve per repetition r, so that the r-th curve uses
     only the r-th timing at every parameter point;
  2. interpolate each curve independently at the recall target, giving N_RUNS
     independent estimates t^(r) of the matched-recall query time;
  3. take the ratio of means R = mean(t_base) / mean(t_idx), and build the
     interval on log R by the delta method,
         SE(log R)^2 = (s_base / (sqrt(N) t_base))^2 + (s_idx / (sqrt(N) t_idx))^2,
     with Welch-Satterthwaite degrees of freedom and a Student-t quantile.  The
     log scale is used because a throughput ratio is multiplicative: it keeps the
     interval positive and symmetric in the quantity actually being claimed.

A ratio is reported as indistinguishable from parity when its interval contains
1.  This is the statement the paper needs: not that CSPG is 0.87x HNSW32, but
that it is 0.87x with an interval excluding 1, hence a measured slowdown rather
than measurement noise.

Outputs
  <out_dir>/ci_ratios_k<k>.csv    one row per (dataset, index, target)
  <out_dir>/ci_ratios_k<k>.tex    LaTeX table, ratios with intervals
  <out_dir>/ci_summary_k<k>.txt   per-index range over datasets, and the
                                  parity-indistinguishable count

Usage
  python benchs/analyze_ci.py --results-dir benchs/results_router \
                              --out-dir     benchs/figures_router/tables
  python benchs/analyze_ci.py --k 10 --targets 0.95 0.99 --baseline HNSW32
  python benchs/analyze_ci.py --paired-baseline     # SHG vs HNSW48, CSPG vs HNSW32
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

# Baseline each method is compared against in its own source paper.  Using the
# matched baseline is what makes the ratio a statement about the method rather
# than about the graph budget: SHG and HNSW48 share (M=48, efC=80), CSPG and
# HNSW32 share (M=32, efC=128).
PAIRED_BASELINE = {
    "SuCo":          "HNSW32",
    "SHG":           "HNSW48",
    "CSPG":          "HNSW32",
    "HNSW48":        "HNSW32",
    "HNSW32":        "HNSW32",
    "IVFFlat":       "HNSW32",
    "OPQ-IVFPQ":     "HNSW32",
    "OPQ-IVFPQ+SQ8": "HNSW32",
}

INDEX_ORDER = ["SuCo", "SHG", "CSPG", "HNSW32", "HNSW48",
               "IVFFlat", "OPQ-IVFPQ", "OPQ-IVFPQ+SQ8"]


# ---------------------------------------------------------------------------
# Per-repetition curves
# ---------------------------------------------------------------------------

def per_run_curves(rows, nq):
    """Return (recalls, T) where T has shape (n_runs, n_points), in ms/query.

    Points whose run_times_s is missing or ragged are dropped, so a partially
    re-run sweep degrades to the points it does have rather than failing.
    """
    recalls, cols = [], []
    n_runs = None
    for r in rows:
        rt = r.get("run_times_s")
        if not rt:
            continue
        if n_runs is None:
            n_runs = len(rt)
        if len(rt) != n_runs:
            continue
        recalls.append(float(r["recall"]))
        cols.append([t / nq * 1000.0 for t in rt])
    if not recalls or n_runs is None or n_runs < 2:
        return None, None
    order = np.argsort(recalls)
    return np.asarray(recalls)[order], np.asarray(cols)[order].T


def interp_at_recall(recalls, times, target):
    """Query time at `target` recall on one repetition's curve.

    The curve is made monotone from the high-recall end (a cumulative minimum
    scanning down from the highest recall), which is the upper-envelope
    convention of the sweep expressed in the time domain: at any recall the cost
    is the cheapest setting that attains at least that recall.  Returns None if
    the sweep never reaches the target.
    """
    if recalls.max() < target:
        return None
    best = np.minimum.accumulate(times[::-1])[::-1]
    j = int(np.searchsorted(recalls, target, side="left"))
    if j == 0:
        return float(best[0])
    r_lo, r_hi = recalls[j - 1], recalls[j]
    if r_hi <= r_lo:
        return float(best[j])
    w = (target - r_lo) / (r_hi - r_lo)
    return float((1 - w) * best[j - 1] + w * best[j])


def matched_recall_times(rows, nq, target):
    """N_RUNS independent matched-recall time estimates, or None."""
    recalls, T = per_run_curves(rows, nq)
    if recalls is None:
        return None
    vals = [interp_at_recall(recalls, T[r], target) for r in range(T.shape[0])]
    if any(v is None for v in vals):
        return None
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# Ratio interval
# ---------------------------------------------------------------------------

def _t_quantile(dof, p=0.975):
    """Student-t two-sided quantile without a SciPy dependency.

    Table for the small dof this design produces (N_RUNS=3 gives dof near 2-4);
    falls back to a Cornish-Fisher expansion around the normal quantile above
    the tabulated range, where the correction is already below 1%.
    """
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
             20: 2.086, 30: 2.042, 60: 2.000}
    dof_r = max(1, int(round(dof)))
    if dof_r in table:
        return table[dof_r]
    keys = sorted(table)
    if dof_r > keys[-1]:
        z = 1.959964
        return z + (z ** 3 + z) / (4.0 * dof_r)
    lo = max(k for k in keys if k < dof_r)
    hi = min(k for k in keys if k > dof_r)
    w = (dof_r - lo) / (hi - lo)
    return (1 - w) * table[lo] + w * table[hi]


def ratio_ci(t_base, t_idx, conf=0.95):
    """Ratio base/idx with a delta-method interval on the log scale.

    Returns dict with the point estimate, interval, relative half-width and
    whether the interval excludes parity.  t_base and t_idx are arrays of
    per-repetition matched-recall times.
    """
    nb, ni = len(t_base), len(t_idx)
    mb, mi = float(np.mean(t_base)), float(np.mean(t_idx))
    sb, si = float(np.std(t_base, ddof=1)), float(np.std(t_idx, ddof=1))
    if mb <= 0 or mi <= 0:
        return None

    # Squared relative standard errors of the two means.
    vb = (sb / (math.sqrt(nb) * mb)) ** 2
    vi = (si / (math.sqrt(ni) * mi)) ** 2
    se_log = math.sqrt(vb + vi)

    # Welch-Satterthwaite on the two relative-variance terms.
    if se_log == 0:
        dof = float(nb + ni - 2)
    else:
        num = (vb + vi) ** 2
        den = (vb ** 2 / max(nb - 1, 1)) + (vi ** 2 / max(ni - 1, 1))
        dof = num / den if den > 0 else float(nb + ni - 2)

    tq = _t_quantile(dof, 0.5 + conf / 2.0)
    ratio = mb / mi
    lo = ratio * math.exp(-tq * se_log)
    hi = ratio * math.exp(+tq * se_log)
    return {
        "ratio": ratio, "lo": lo, "hi": hi,
        "half_width_pct": 100.0 * (hi - lo) / (2.0 * ratio),
        "cv_base_pct": 100.0 * sb / mb,
        "cv_idx_pct": 100.0 * si / mi,
        "dof": dof,
        "excludes_parity": (lo > 1.0) or (hi < 1.0),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_all(results_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        out[d.get("dataset", os.path.basename(path))] = d
    return out


def collect(all_results, k, targets, baseline, paired, conf):
    rows = []
    key = f"recall_k{k}"
    for ds, res in sorted(all_results.items()):
        curves = res.get(key) or {}
        nq = int(res.get("nq") or 0)
        if not curves or nq <= 0:
            continue
        cache = {}
        for target in targets:
            def times_for(name):
                if (name, target) not in cache:
                    r = curves.get(name)
                    cache[(name, target)] = (
                        matched_recall_times(r, nq, target) if r else None
                    )
                return cache[(name, target)]

            for idx_name in INDEX_ORDER:
                if idx_name not in curves:
                    continue
                base_name = PAIRED_BASELINE.get(idx_name, baseline) if paired else baseline
                if idx_name == base_name:
                    continue
                tb, ti = times_for(base_name), times_for(idx_name)
                if tb is None or ti is None:
                    rows.append({"dataset": ds, "index": idx_name,
                                 "baseline": base_name, "target": target,
                                 "ratio": None})
                    continue
                ci = ratio_ci(tb, ti, conf=conf)
                if ci is None:
                    continue
                rows.append({"dataset": ds, "index": idx_name,
                             "baseline": base_name, "target": target,
                             "ms_baseline": float(np.mean(tb)),
                             "ms_index": float(np.mean(ti)), **ci})
    return rows


def write_csv(rows, path):
    cols = ["dataset", "index", "baseline", "target", "ratio", "lo", "hi",
            "half_width_pct", "cv_base_pct", "cv_idx_pct", "dof",
            "excludes_parity", "ms_baseline", "ms_index"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def write_tex(rows, path, targets, conf):
    """Ratio +- interval per (dataset, index), one column block per target."""
    by = defaultdict(dict)
    idxs, dss = [], []
    for r in rows:
        if r.get("ratio") is None:
            continue
        by[(r["dataset"], r["index"])][r["target"]] = r
        if r["index"] not in idxs:
            idxs.append(r["index"])
        if r["dataset"] not in dss:
            dss.append(r["dataset"])
    idxs = [i for i in INDEX_ORDER if i in idxs]

    ncol = len(idxs) * len(targets)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Matched-recall throughput ratio against each method's own "
        rf"HNSW reference, with {int(conf*100)}\% confidence intervals from "
        r"$N{=}3$ repetitions. Intervals containing $1$ (\dag) denote parity "
        r"that the measurement cannot distinguish from a difference.}",
        r"\label{tab:ci-ratios}", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l" + "c" * ncol + r"@{}}", r"\toprule",
    ]
    hdr = " & ".join(rf"\multicolumn{{{len(targets)}}}{{c}}{{\textbf{{{i}}}}}" for i in idxs)
    lines.append(" & " + hdr + r" \\")
    cm = "".join(rf"\cmidrule(lr){{{2+j*len(targets)}-{1+(j+1)*len(targets)}}}"
                 for j in range(len(idxs)))
    lines.append(cm)
    lines.append(r"\textbf{Dataset} & " +
                 " & ".join(f"${t}$" for _ in idxs for t in targets) + r" \\")
    lines.append(r"\midrule")
    for ds in dss:
        cells = []
        for i in idxs:
            for t in targets:
                r = by.get((ds, i), {}).get(t)
                if r is None:
                    cells.append("---")
                else:
                    mark = "" if r["excludes_parity"] else r"$^\dag$"
                    cells.append(rf"{r['ratio']:.2f}\,\tiny[{r['lo']:.2f},{r['hi']:.2f}]{mark}")
        lines.append(f"{ds} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_summary(rows, path, conf):
    ok = [r for r in rows if r.get("ratio") is not None]
    lines = [f"Matched-recall ratio intervals ({int(conf*100)}% CI, N=3 repetitions)", ""]
    if ok:
        hw = np.array([r["half_width_pct"] for r in ok])
        cv = np.array([r["cv_idx_pct"] for r in ok])
        lines += [
            f"measurements                : {len(ok)}",
            f"per-point CV                : median {np.median(cv):.2f}%, "
            f"p95 {np.percentile(cv, 95):.2f}%, max {cv.max():.2f}%",
            f"ratio interval half-width   : median {np.median(hw):.2f}%, "
            f"p95 {np.percentile(hw, 95):.2f}%, max {hw.max():.2f}%",
            f"indistinguishable from 1.0  : "
            f"{sum(1 for r in ok if not r['excludes_parity'])} / {len(ok)}",
            "",
        ]
    for idx in INDEX_ORDER:
        sub = [r for r in ok if r["index"] == idx]
        if not sub:
            continue
        base = sorted({r["baseline"] for r in sub})
        rr = np.array([r["ratio"] for r in sub])
        los = np.array([r["lo"] for r in sub])
        his = np.array([r["hi"] for r in sub])
        par = [r for r in sub if not r["excludes_parity"]]
        lines.append(
            f"{idx:14s} vs {'/'.join(base):8s}  "
            f"point {rr.min():.2f}-{rr.max():.2f}   "
            f"interval-union {los.min():.2f}-{his.max():.2f}   "
            f"parity-indistinguishable {len(par)}/{len(sub)}"
        )
        for r in sorted(par, key=lambda r: r["dataset"]):
            lines.append(f"    dag {r['dataset']:12s} @{r['target']}  "
                         f"{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="benchs/results_router")
    ap.add_argument("--out-dir", default="benchs/figures_router/tables")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--baseline", default="HNSW32",
                    help="baseline when --paired-baseline is not set")
    ap.add_argument("--paired-baseline", action="store_true",
                    help="compare each method against the HNSW configuration "
                         "its own paper adopts (SHG vs HNSW48, CSPG vs HNSW32)")
    ap.add_argument("--conf", type=float, default=0.95)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = load_all(args.results_dir)
    if not all_results:
        raise SystemExit(f"no results_*.json under {args.results_dir}")

    rows = collect(all_results, args.k, args.targets,
                   args.baseline, args.paired_baseline, args.conf)
    tag = f"k{args.k}" + ("_paired" if args.paired_baseline else "")
    write_csv(rows, os.path.join(args.out_dir, f"ci_ratios_{tag}.csv"))
    write_tex(rows, os.path.join(args.out_dir, f"ci_ratios_{tag}.tex"),
              args.targets, args.conf)
    print(write_summary(rows, os.path.join(args.out_dir, f"ci_summary_{tag}.txt"),
                        args.conf))
    print(f"\nwrote ci_ratios_{tag}.{{csv,tex}} and ci_summary_{tag}.txt "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
