#!/usr/bin/env python3
"""
benchs/bench_m2_distcount.py

Distance evaluations per query at matched recall, for every index.

This is the quantity CSPG and SHG optimise and the quantity their papers report
gains in.  The cross-algorithm benchmark measures wall-clock instead, so a flat
wall-clock result is open to two readings: the distance-count reduction did not
reproduce (an implementation fault), or it reproduced but did not convert into
time (the paper's thesis).  Counting distances separates them, on an axis that
does not depend on the machine the benchmark ran on.

Protocol
--------
Timing and counting are never mixed.  This script runs an *untimed* counting
pass over the parameter sweep and takes matched-recall wall-clock from
results_<dataset>.json, produced by the ordinary uninstrumented benchmark.  The
counting pass may use every core: each query is searched end-to-end by one
thread, so per-query counts are independent of the thread count, and the
counters are per-thread accumulators flushed once per thread (SHG) or once per
query (CSPG, SuCo) — never from an inner loop.

Matched recall is reached by the same linear interpolation in recall that
produced Table II's throughput ratios (bench_router_paper.time_and_std_at_recall),
applied to the count curve.  Count ratio and throughput ratio are therefore
quoted at the same recall, which is what makes the pair meaningful.

Each method is paired with the HNSW baseline its own paper adopts:
CSPG with HNSW32 (M=32, efC=128), SHG with HNSW48 (M=48, efC=80).

What is counted
---------------
  full     evaluations against full-width base vectors (d floats).  Directly
           comparable across indexes: same kernel, same width, same layout.
  narrow   evaluations at reduced width — SHG's compressed levels, SuCo's
           centroids.  Counted apart because a single scalar would hide
           exactly the effect under study.
  floats   full * d + narrow_floats, the total float traffic of the distance
           path.  This is the honest comparison for SHG, whose entire claim is
           that its evaluations are narrower rather than fewer.

Counters for HNSW32/HNSW48 are built into FAISS.  SuCo, CSPG and SHG are
instrumented in this fork (see benchs/M2_DISTANCE_COUNTERS.md); an index whose
counters are missing from the build is reported as uninstrumented rather than
silently wrong.

Outputs
  <out>/m2_distcount_<dataset>.json   per index: counts, bytes, ms at each target
  <out>/m2_distcount_table.tex        the paper table, if --latex is given
  stdout                              the table, and the paired ratio summary

Usage
  python benchs/bench_m2_distcount.py --dataset sift1m
  python benchs/bench_m2_distcount.py --dataset all --targets 0.95 0.99 --latex
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with custom index support first.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_router_paper import (  # noqa: E402
    ALL_DATASETS, BUILDERS, SEARCH_FACTORY, DEFAULT_INDEX_TYPES,
    compute_recall_at_k, load_dataset, resolve_suco_nsubspaces,
)
# Reuse the router's on-disk naming rather than reconstructing it: SuCo's file
# carries its subspace count (<ds>_suco_ns<N>.idx), so a naive <ds>_<kind>.idx
# misses the prebuilt index and rebuilds one -- with the auto-rule Ns, not the
# per-dataset override, which on GIST1M and SpaceV10M is a different index.
from bench_thread_scaling import index_path_for  # noqa: E402

BYTES_PER_FLOAT = 4

# Each method against the HNSW configuration its own paper adopts.  Comparing
# CSPG with HNSW48 or SHG with HNSW32 would confound the method with the
# out-degree, which is the confound the whole reproduction exists to remove.
BASELINE_OF = {
    "cspg": "hnsw32",
    "shg":  "hnsw48",
    "suco": "hnsw32",
}

# No index reaches Recall@10=0.95 on UQ-V, so its row is quoted at 0.80 — the
# same substitution Table II makes in its footnote (a).  Anything not listed
# here uses --latex-target.
LATEX_TARGET_OVERRIDE = {"uqv": 0.80}


# ---------------------------------------------------------------------------
# Counter access
# ---------------------------------------------------------------------------

def _get_stats(name):
    """Resolve a FAISS stats global under any of the three binding styles.

    SWIG wraps `FAISS_API extern` globals into `faiss.cvar` for some headers and
    not others, so this fork also exports explicit `<name>_get()` / `<name>_reset()`
    helpers.  Returns (snapshot_fn, reset_fn) or None.
    """
    # faiss.cvar is absent entirely from some builds, so it cannot be reached
    # through a bare attribute access.
    obj = getattr(getattr(faiss, "cvar", None), name, None)
    if obj is not None:
        return (lambda: obj), obj.reset

    getter = getattr(faiss, f"{name}_get", None)
    resetter = getattr(faiss, f"{name}_reset", None)
    if getter is not None and resetter is not None:
        return getter, resetter

    getter = getattr(faiss, f"get_{name}", None)
    if getter is not None:
        return getter, (lambda: getter().reset())
    return None


# Which counter object and fields describe each index kind.  `full` counts
# distance evaluations over full-width base vectors, the only figure directly
# comparable across indexes; `narrow` counts evaluations at reduced width
# (SHG's compressed levels, SuCo's centroids), which cost proportionally less.
COUNTER_SPEC = {
    "hnsw32": ("hnsw_stats", {"full": "ndis"}),
    "hnsw48": ("hnsw_stats", {"full": "ndis"}),
    "cspg":   ("cspg_stats", {"full": "n_dis",
                              "stage1": "n_dis_stage1"}),
    "shg":    ("shg_stats",  {"full": "n_dis_full",
                              "narrow": "n_dis_compressed",
                              "narrow_floats": "compressed_floats",
                              "lb_pruned": "n_lb_pruned"}),
    "suco":   ("suco_stats", {"full": "n_dis_rerank",
                              "narrow": "n_dis_centroid",
                              "narrow_floats": "centroid_floats"}),
}


def read_counters(kind, d, nq):
    """Per-query counts and bytes for `kind`, or None if uninstrumented."""
    spec = COUNTER_SPEC.get(kind)
    if spec is None:
        return None
    name, fields = spec
    handles = _get_stats(name)
    if handles is None:
        return None
    st = handles[0]()
    out = {}
    for label, attr in fields.items():
        val = getattr(st, attr, None)
        if val is None:
            return None
        out[label] = float(val) / nq
    full = out.get("full", 0.0)
    narrow_floats = out.get("narrow_floats", 0.0)
    out["floats_per_query"] = full * d + narrow_floats
    out["bytes_per_query"] = out["floats_per_query"] * BYTES_PER_FLOAT
    return out


def reset_counters(kind):
    spec = COUNTER_SPEC.get(kind)
    if spec is None:
        return
    handles = _get_stats(spec[0])
    if handles is not None:
        handles[1]()


# ---------------------------------------------------------------------------
# Matched-recall selection
# ---------------------------------------------------------------------------

def sweep_counts(idx, kind, xq, gt, k, threads=0):
    """Untimed pass over the parameter grid, recording recall and counts.

    Runs multi-threaded by default.  Each query is searched end-to-end by a
    single thread, so per-query counts do not depend on how many threads are
    running; and because nothing here is timed, the counter flush costs
    nothing that matters.  Pass threads=1 to force the serial path.
    """
    factory, params, pname = SEARCH_FACTORY[kind]
    saved = faiss.omp_get_max_threads()
    if threads and threads > 0:
        faiss.omp_set_num_threads(threads)
    rows = []
    try:
        for p in params:
            fn = factory(p)
            reset_counters(kind)
            _, I = fn(idx, xq, k)
            c = read_counters(kind, xq.shape[1], xq.shape[0])
            recall = compute_recall_at_k(I, gt, k)
            rows.append({"param": float(p) if isinstance(p, float) else int(p),
                         "param_name": pname,
                         "recall": round(recall, 6),
                         "counts": c})
            print(f"    {pname}={p}: recall@{k}={recall:.4f}, "
                  f"dis/q={c['full']:.1f}"
                  + (f", narrow/q={c['narrow']:.1f}" if c.get("narrow") else "")
                  + f", kB/q={c['bytes_per_query']/1024:.1f}")
    finally:
        faiss.omp_set_num_threads(saved)
    rows.sort(key=lambda r: r["recall"])
    return rows


def interp_at_recall(rows, target, getter):
    """Linear interpolation of `getter(row)` at a recall target, or None.

    Mirrors bench_router_paper.time_and_std_at_recall exactly, so that a count
    ratio and a throughput ratio quoted at the same target are quoted at the
    same point of the frontier.
    """
    rows = [r for r in rows if getter(r) is not None]
    rows = sorted(rows, key=lambda r: r["recall"])
    above = [r for r in rows if r["recall"] >= target]
    below = [r for r in rows if r["recall"] < target]
    if not above:
        return None
    if not below:
        return float(getter(above[0]))
    lo, hi = below[-1], above[0]
    if hi["recall"] == lo["recall"]:
        return float(getter(hi))
    t = (target - lo["recall"]) / (hi["recall"] - lo["recall"])
    return float(getter(lo) + t * (getter(hi) - getter(lo)))


def cheapest_reaching(rows, target):
    """Sweep point of smallest full-width count that reaches `target`."""
    ok = [r for r in rows if r["recall"] >= target and r["counts"]]
    if not ok:
        return None
    return min(ok, key=lambda r: r["counts"]["full"])


def timing_curve(prior, label, k):
    """The (param, recall, ms_per_query) curve the paper's ratios came from."""
    return (prior.get(f"recall_k{k}") or {}).get(label) or []


def ms_at_target(prior, label, k, target):
    """Matched-recall wall-clock from the ordinary (uninstrumented) benchmark.

    Prefers the derived time_at_recall block so the figure is byte-identical to
    Table II's; falls back to interpolating the raw curve.
    """
    tar = (prior.get("time_at_recall") or {}).get(f"recall_k{k}") or {}
    node = tar.get(f"r{int(round(target * 100))}") or {}
    entry = node.get(label) or {}
    ms = entry.get("ms_per_query")
    if ms is not None:
        return float(ms)
    return interp_at_recall(timing_curve(prior, label, k), target,
                            lambda r: r.get("ms_per_query"))


def recall_agreement(count_rows, time_rows):
    """Largest |recall_counting - recall_timing| over shared sweep points.

    The two passes are separate runs of the same sweep, so a large disagreement
    would mean the interpolations sit on different curves and no ratio drawn
    across them is trustworthy.  Reported rather than asserted.
    """
    by_param = {r["param"]: r["recall"] for r in time_rows if "recall" in r}
    diffs = [abs(r["recall"] - by_param[r["param"]])
             for r in count_rows if r["param"] in by_param]
    return max(diffs) if diffs else None


# ---------------------------------------------------------------------------

def run_dataset(ds_name, args):
    xb, xq, gt = load_dataset(ds_name, args.data_dir, args.index_dir)
    d = int(xb.shape[1])
    if args.max_queries and xq.shape[0] > args.max_queries:
        print(f"  subsampling {xq.shape[0]} -> {args.max_queries} queries")
        xq = np.ascontiguousarray(xq[:args.max_queries])
        gt = np.ascontiguousarray(gt[:args.max_queries])

    results_path = os.path.join(args.results_dir, f"results_{ds_name}.json")
    prior = {}
    if os.path.exists(results_path):
        with open(results_path) as fh:
            prior = json.load(fh)
    else:
        print(f"  note: {results_path} absent; ms/query will be blank")

    out = {"dataset": ds_name, "n": int(xb.shape[0]), "d": d,
           "k": args.k, "targets": args.targets,
           "nq": int(xq.shape[0]),
           "count_threads": args.count_threads or int(faiss.omp_get_max_threads()),
           "indexes": {}}

    for kind in args.index_types:
        label, builder = BUILDERS[kind]
        idx_path = index_path_for(ds_name, kind, args.index_dir, d)
        # Dropped at the end of every branch: five indexes at up to 13 GB each
        # do not fit alongside one another, so each must be released before the
        # next is read.
        idx = None
        # One index that fails to load or build must not cost the whole sweep:
        # the run is long, the indexes are independent, and a partial table is
        # more useful than a traceback.
        try:
            if os.path.exists(idx_path):
                print(f"  {label}: loading {idx_path}")
                idx = faiss.read_index(idx_path)
            else:
                print(f"  {label}: building (no prebuilt index at {idx_path})")
                if kind == "suco":
                    ns, _ = resolve_suco_nsubspaces(ds_name, d)
                    built = builder(xb, d, n_override=ns)
                else:
                    built = builder(xb, d)
                idx = built[0] if isinstance(built, tuple) else built
        except Exception as e:
            print(f"  {label}: UNAVAILABLE — {e}")
            out["indexes"][label] = {"instrumented": False, "error": str(e)}
            continue

        if read_counters(kind, d, 1) is None:
            print(f"  {label}: no distance counter compiled in "
                  f"(see benchs/M2_DISTANCE_COUNTERS.md) — skipped")
            out["indexes"][label] = {"instrumented": False}
            idx = None
            continue

        try:
            rows = sweep_counts(idx, kind, xq, gt, args.k, args.count_threads)
        except Exception as e:
            print(f"  {label}: sweep FAILED — {e}")
            out["indexes"][label] = {"instrumented": True, "error": str(e)}
            idx = None
            continue

        t_rows = timing_curve(prior, label, args.k)
        entry = {
            "instrumented": True,
            "kind": kind,
            "baseline": BUILDERS[BASELINE_OF[kind]][0] if kind in BASELINE_OF else None,
            "sweep": rows,
            "recall_agreement_vs_timing": recall_agreement(rows, t_rows),
            "at_target": {},
        }
        for t in args.targets:
            dis = interp_at_recall(rows, t, lambda r: r["counts"]["full"])
            if dis is None:
                entry["at_target"][str(t)] = None
                continue
            point = cheapest_reaching(rows, t)
            entry["at_target"][str(t)] = {
                # Interpolated at the target: the figure to pair with ms/query.
                "dis_per_query": dis,
                "narrow_dis_per_query": interp_at_recall(
                    rows, t, lambda r: r["counts"].get("narrow")),
                "floats_per_query": interp_at_recall(
                    rows, t, lambda r: r["counts"]["floats_per_query"]),
                "bytes_per_query": interp_at_recall(
                    rows, t, lambda r: r["counts"]["bytes_per_query"]),
                "lb_pruned_per_query": interp_at_recall(
                    rows, t, lambda r: r["counts"].get("lb_pruned")),
                "stage1_dis_per_query": interp_at_recall(
                    rows, t, lambda r: r["counts"].get("stage1")),
                # The cheapest sweep point actually reaching the target, for
                # readers who distrust interpolation on a non-monotone sweep.
                "point_param": point["param"] if point else None,
                "point_recall": point["recall"] if point else None,
                "point_dis_per_query": point["counts"]["full"] if point else None,
                "ms_per_query": ms_at_target(prior, label, args.k, t),
            }
        out["indexes"][label] = entry
        idx = None

    add_ratios(out, args)

    path = os.path.join(args.out_dir, f"m2_distcount_{ds_name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {path}")
    return out


def add_ratios(out, args):
    """Attach each method's count/traffic/throughput ratio against its baseline.

    A ratio below 1 on the count axis and at or below 1 on the throughput axis
    is the paper's claim: the distance reduction reproduces and does not
    convert into time.
    """
    by_label = out["indexes"]
    for kind, base_kind in BASELINE_OF.items():
        label = BUILDERS[kind][0]
        base_label = BUILDERS[base_kind][0]
        e = by_label.get(label)
        b = by_label.get(base_label)
        if not e or not b or not e.get("instrumented") or not b.get("instrumented"):
            continue
        e["ratios"] = {}
        for t in args.targets:
            a = e["at_target"].get(str(t))
            bb = b["at_target"].get(str(t))
            if not a or not bb:
                e["ratios"][str(t)] = None
                continue

            def _r(num, den):
                if num is None or den in (None, 0):
                    return None
                return round(num / den, 4)

            e["ratios"][str(t)] = {
                "baseline": base_label,
                # <1 means the method evaluates fewer full-width distances.
                "dis_ratio": _r(a["dis_per_query"], bb["dis_per_query"]),
                # <1 means it streams fewer floats overall, narrow ones included.
                "floats_ratio": _r(a["floats_per_query"], bb["floats_per_query"]),
                # >1 means it is faster.  This is Table II's quantity.
                "qps_ratio": _r(bb["ms_per_query"], a["ms_per_query"])
                             if (a["ms_per_query"] and bb["ms_per_query"]) else None,
            }


def print_table(out, targets):
    print(f"\n{out['dataset']}  (n={out['n']}, d={out['d']}, "
          f"k={out['k']}, nq={out['nq']})")
    hdr = (f"{'index':16s}{'target':>7s}{'dis/q':>13s}{'narrow/q':>13s}"
           f"{'kB/q':>10s}{'ms/q':>10s}{'ns/dis':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for label, e in out["indexes"].items():
        if not e.get("instrumented"):
            print(f"{label:16s}{'—':>7s}  not instrumented")
            continue
        for t in targets:
            a = e["at_target"].get(str(t))
            if a is None:
                print(f"{label:16s}{t:>7.2f}{'—':>13s}   (target not reached)")
                continue
            ms = a["ms_per_query"]
            nsd = (ms * 1e6 / a["dis_per_query"]) if (ms and a["dis_per_query"]) else None
            print(f"{label:16s}{t:>7.2f}{a['dis_per_query']:>13.1f}"
                  f"{(a['narrow_dis_per_query'] or 0):>13.1f}"
                  f"{a['bytes_per_query']/1024:>10.1f}"
                  f"{(ms if ms else float('nan')):>10.4f}"
                  f"{(nsd if nsd else float('nan')):>9.2f}")
        ra = e.get("recall_agreement_vs_timing")
        if ra is not None and ra > 0.005:
            print(f"{'':16s}  warning: counting and timing sweeps disagree on "
                  f"recall by up to {ra:.4f}")


def print_ratio_summary(outs, targets):
    """The claim, one line per (dataset, method, target)."""
    print("\n" + "=" * 78)
    print("MATCHED-RECALL RATIOS AGAINST EACH METHOD'S OWN HNSW BASELINE")
    print("  dis   = full-width distance evaluations per query, method / baseline")
    print("  float = total float traffic per query, method / baseline")
    print("  QPS   = throughput, method / baseline (Table II's quantity)")
    print("=" * 78)
    hdr = (f"{'dataset':12s}{'index':7s}{'base':8s}{'target':>7s}"
           f"{'dis':>8s}{'float':>8s}{'QPS':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for out in outs:
        for label, e in out["indexes"].items():
            for t in targets:
                r = (e.get("ratios") or {}).get(str(t))
                if not r:
                    continue
                def f(x):
                    return f"{x:>8.3f}" if x is not None else f"{'—':>8s}"
                print(f"{out['dataset']:12s}{label:7s}{r['baseline']:8s}{t:>7.2f}"
                      f"{f(r['dis_ratio'])}{f(r['floats_ratio'])}{f(r['qps_ratio'])}")


def write_latex(outs, targets, path, main_target=0.95):
    """The paper table: distance count and throughput at matched recall."""
    if main_target not in targets:
        main_target = targets[0]
    overridden = []
    lines = [
        r"% Generated by benchs/bench_m2_distcount.py -- do not edit by hand.",
        r"\begin{table}[!tb]",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Full-Width Distance Evaluations per Query and Throughput at "
        r"Matched $\Rk{10}$, Each Method Against the HNSW Configuration Its Own "
        r"Paper Adopts (CSPG\,/\,HNSW32, SHG\,/\,HNSW48). Ratios ${<}1$ on "
        r"$n_{\mathrm{dis}}$ Denote Fewer Evaluations; Ratios ${>}1$ on QPS "
        r"Denote Faster}",
        r"\label{tab:distcount}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l rr c rr c@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{CSPG vs.\ HNSW32}} & "
        r"\multicolumn{3}{c}{\textbf{SHG vs.\ HNSW48}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"\textbf{Dataset} & $n_{\mathrm{dis}}$ & ratio & QPS ratio "
        r"& $n_{\mathrm{dis}}$ & ratio & QPS ratio \\",
        r"\midrule",
    ]
    for out in outs:
        ds = out["dataset"]
        target = LATEX_TARGET_OVERRIDE.get(ds, main_target)
        if target != main_target:
            overridden.append((ds, target))
        cells = [ds]
        for kind in ("cspg", "shg"):
            label = BUILDERS[kind][0]
            e = out["indexes"].get(label) or {}
            a = (e.get("at_target") or {}).get(str(target))
            r = (e.get("ratios") or {}).get(str(target))
            if not a or not r:
                cells += ["---", "---", "---"]
                continue
            cells.append(f"{a['dis_per_query']:.0f}")
            cells.append(f"{r['dis_ratio']:.2f}" if r["dis_ratio"] else "---")
            cells.append(f"{r['qps_ratio']:.2f}" if r["qps_ratio"] else "---")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\raggedright\scriptsize",
        f"Counts are interpolated at $\\Rk{{10}}={main_target}$ on the same "
        r"recall axis as the throughput ratios, from an untimed instrumented "
        r"pass. SHG additionally evaluates compressed distances at the upper "
        r"levels and in its cross-level lower bound; those are excluded here "
        r"and reported as float traffic in the text."
        + ("".join(f" No index reaches $\\Rk{{10}}={main_target}$ on "
                   f"{ds.upper()}; that row is quoted at {t} instead."
                   for ds, t in overridden)),
        r"\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["sift1m"])
    ap.add_argument("--data-dir", default=os.environ.get("ANN_DATA_DIR", "data"))
    ap.add_argument("--index-dir", default="indexes_router")
    ap.add_argument("--results-dir", default="benchs/results_router")
    ap.add_argument("--out-dir", default="benchs/results_router")
    ap.add_argument("--index-types", nargs="+", default=DEFAULT_INDEX_TYPES,
                    choices=sorted(BUILDERS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--count-threads", type=int, default=0,
                    help="0 = use OMP_NUM_THREADS. Counts are thread-count "
                         "independent; nothing here is timed.")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = use the full query set. Subsampling shifts recall "
                         "and therefore the matched-recall point; use only to "
                         "smoke-test.")
    ap.add_argument("--latex", action="store_true",
                    help="also emit m2_distcount_table.tex")
    ap.add_argument("--latex-target", type=float, default=0.95,
                    help="recall target the LaTeX table quotes (default 0.95, "
                         "matching Table II's main column). Datasets in "
                         "LATEX_TARGET_OVERRIDE fall back to their own target.")
    ap.add_argument("--summarize", action="store_true",
                    help="re-read the per-dataset JSONs and print the summary "
                         "(and --latex table) without re-running anything. Lets "
                         "the cluster run one dataset per invocation and still "
                         "produce one table across all of them.")
    args = ap.parse_args()

    if args.dataset == ["all"]:
        args.dataset = list(ALL_DATASETS)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    if args.summarize:
        outs = []
        for ds in args.dataset:
            path = os.path.join(args.out_dir, f"m2_distcount_{ds}.json")
            if not os.path.exists(path):
                print(f"  {ds}: {path} absent — skipped")
                continue
            with open(path) as fh:
                outs.append(json.load(fh))
        if not outs:
            sys.exit("nothing to summarize")
        for out in outs:
            print_table(out, args.targets)
        print_ratio_summary(outs, args.targets)
        if args.latex:
            write_latex(outs, args.targets,
                        os.path.join(args.out_dir, "m2_distcount_table.tex"),
                        main_target=args.latex_target)
        return

    missing = [k for k in args.index_types if read_counters(k, 1, 1) is None]
    if missing:
        print(f"WARNING: no counters compiled in for {', '.join(missing)} — "
              f"those rows will be empty. See benchs/M2_DISTANCE_COUNTERS.md.")

    outs = []
    for ds in args.dataset:
        print(f"\n=== {ds} ===")
        try:
            out = run_dataset(ds, args)
            print_table(out, args.targets)
            outs.append(out)
        except Exception as e:
            print(f"  {ds} FAILED — {e}; continuing to the next dataset")

    if outs:
        print_ratio_summary(outs, args.targets)
        if args.latex:
            write_latex(outs, args.targets,
                        os.path.join(args.out_dir, "m2_distcount_table.tex"),
                        main_target=args.latex_target)


if __name__ == "__main__":
    main()
