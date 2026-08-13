#!/usr/bin/env python3
"""
benchs/bench_m2_distcount.py

Distance evaluations per query at matched recall, for every index.

This is the quantity CSPG and SHG optimise and the quantity their papers report
gains in.  The cross-algorithm benchmark measures wall-clock instead, so a flat
wall-clock result is currently open to two readings: the distance-count reduction
did not reproduce (an implementation fault), or it reproduced but did not convert
into time (the paper's thesis).  Counting distances separates them.

Timing and counting are never mixed.  The counters are global and updated under a
critical section, so a run that reads them is not a run whose wall-clock means
anything.  This script therefore runs an untimed counting pass over the parameter
sweep, and takes matched-recall wall-clock from results_<dataset>.json, produced
by the ordinary uninstrumented benchmark.

Counters for HNSW32/HNSW48 are built into FAISS.  SuCo, CSPG and SHG require the
patch in benchs/M2_DISTANCE_COUNTERS.md; without it those rows are reported as
unavailable rather than silently wrong.

Outputs
  <out>/m2_distcount_<dataset>.json   per index: counts, bytes, ms at each target
  stdout                              the table to put in the paper

Usage
  python benchs/bench_m2_distcount.py --dataset sift1m
  python benchs/bench_m2_distcount.py --dataset all --targets 0.95 0.99
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


# ---------------------------------------------------------------------------
# Counter access
# ---------------------------------------------------------------------------

def _get_stats(name):
    """Resolve a FAISS stats global under either binding style, else None."""
    obj = getattr(faiss.cvar, name, None)
    if obj is not None:
        return obj
    getter = getattr(faiss, f"get_{name}", None)
    return getter() if getter is not None else None


# Which counter object and fields describe each index kind.  `full` counts
# distance evaluations over full-width base vectors, the only figure directly
# comparable across indexes; `narrow` counts evaluations at reduced width
# (SHG's compressed levels, SuCo's centroids), which cost proportionally less.
COUNTER_SPEC = {
    "hnsw32": ("hnsw_stats", {"full": "ndis"}),
    "hnsw48": ("hnsw_stats", {"full": "ndis"}),
    "cspg":   ("cspg_stats", {"full": "n_dis"}),
    "shg":    ("shg_stats",  {"full": "n_dis_full",
                              "narrow": "n_dis_compressed",
                              "narrow_floats": "compressed_floats"}),
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
    st = _get_stats(name)
    if st is None:
        return None
    out = {}
    for label, attr in fields.items():
        val = getattr(st, attr, None)
        if val is None:
            return None
        out[label] = float(val) / nq
    full = out.get("full", 0.0)
    narrow_floats = out.get("narrow_floats", 0.0)
    out["bytes_per_query"] = (full * d + narrow_floats) * BYTES_PER_FLOAT
    return out


def reset_counters(kind):
    spec = COUNTER_SPEC.get(kind)
    if spec is None:
        return
    st = _get_stats(spec[0])
    if st is not None:
        st.reset()


# ---------------------------------------------------------------------------
# Matched-recall selection
# ---------------------------------------------------------------------------

def sweep_counts(idx, kind, xq, gt, k, threads_for_counting=1):
    """Untimed pass over the parameter grid, recording recall and counts.

    Counting runs single threaded by default: the counter flush is a critical
    section, so a multi-threaded counting pass would be both slower and, for any
    counter that is not strictly additive, less trustworthy.  Counts per query
    are thread-count independent, so this costs nothing in fidelity.
    """
    factory, params, pname = SEARCH_FACTORY[kind]
    saved = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads_for_counting)
    rows = []
    try:
        for p in params:
            fn = factory(p)
            reset_counters(kind)
            _, I = fn(idx, xq, k)
            c = read_counters(kind, xq.shape[1], xq.shape[0])
            rows.append({"param": p, "param_name": pname,
                         "recall": compute_recall_at_k(I, gt, k),
                         "counts": c})
    finally:
        faiss.omp_set_num_threads(saved)
    return rows


def at_target(rows, target):
    """Cheapest sweep point reaching `target`, by full-width distance count.

    Selecting by count rather than by parameter value keeps the comparison on the
    axis under study and is well defined even where the sweep is non-monotone.
    """
    ok = [r for r in rows if r["recall"] >= target and r["counts"]]
    if not ok:
        return None
    return min(ok, key=lambda r: r["counts"]["full"])


def ms_at_target(results_json, kind_label, k, target):
    """Matched-recall wall-clock from the ordinary (uninstrumented) benchmark."""
    tar = (results_json.get("time_at_recall") or {}).get(f"recall_k{k}") or {}
    node = tar.get(f"r{int(round(target*100))}") or {}
    entry = node.get(kind_label) or {}
    return entry.get("ms_per_query")


# ---------------------------------------------------------------------------

def run_dataset(ds_name, args):
    xb, xq, gt = load_dataset(ds_name, args.data_dir, args.index_dir)
    d = xb.shape[1]

    results_path = os.path.join(args.results_dir, f"results_{ds_name}.json")
    prior = {}
    if os.path.exists(results_path):
        with open(results_path) as fh:
            prior = json.load(fh)
    else:
        print(f"  note: {results_path} absent; ms/query will be blank")

    out = {"dataset": ds_name, "n": int(xb.shape[0]), "d": int(d),
           "k": args.k, "targets": args.targets, "indexes": {}}

    for kind in args.index_types:
        label, builder = BUILDERS[kind]
        idx_path = index_path_for(ds_name, kind, args.index_dir, d)
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

        if read_counters(kind, d, 1) is None:
            print(f"  {label}: no distance counter compiled in "
                  f"(see benchs/M2_DISTANCE_COUNTERS.md) — skipped")
            out["indexes"][label] = {"instrumented": False}
            del idx
            continue

        rows = sweep_counts(idx, kind, xq, gt, args.k, args.count_threads)
        entry = {"instrumented": True, "sweep": rows, "at_target": {}}
        for t in args.targets:
            sel = at_target(rows, t)
            if sel is None:
                entry["at_target"][str(t)] = None
                continue
            entry["at_target"][str(t)] = {
                "param": sel["param"],
                "recall": sel["recall"],
                "dis_per_query": sel["counts"]["full"],
                "narrow_dis_per_query": sel["counts"].get("narrow"),
                "bytes_per_query": sel["counts"]["bytes_per_query"],
                "ms_per_query": ms_at_target(prior, label, args.k, t),
            }
        out["indexes"][label] = entry
        del idx

    path = os.path.join(args.out_dir, f"m2_distcount_{ds_name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {path}")
    return out


def print_table(out, targets):
    print(f"\n{out['dataset']}  (n={out['n']}, d={out['d']}, k={out['k']})")
    hdr = f"{'index':16s}{'target':>8s}{'dis/q':>12s}{'narrow/q':>12s}" \
          f"{'kB/q':>10s}{'ms/q':>10s}{'ns/dis':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for label, e in out["indexes"].items():
        if not e.get("instrumented"):
            print(f"{label:16s}{'—':>8s}{'not instrumented':>46s}")
            continue
        for t in targets:
            a = e["at_target"].get(str(t))
            if a is None:
                print(f"{label:16s}{t:>8.2f}{'—':>12s}")
                continue
            ms = a["ms_per_query"]
            nsd = (ms * 1e6 / a["dis_per_query"]) if (ms and a["dis_per_query"]) else None
            print(f"{label:16s}{t:>8.2f}{a['dis_per_query']:>12.1f}"
                  f"{(a['narrow_dis_per_query'] or 0):>12.1f}"
                  f"{a['bytes_per_query']/1024:>10.1f}"
                  f"{(ms if ms else float('nan')):>10.4f}"
                  f"{(nsd if nsd else float('nan')):>10.2f}")


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
    ap.add_argument("--count-threads", type=int, default=1)
    args = ap.parse_args()

    if args.dataset == ["all"]:
        args.dataset = list(ALL_DATASETS)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    for ds in args.dataset:
        print(f"\n=== {ds} ===")
        print_table(run_dataset(ds, args), args.targets)


if __name__ == "__main__":
    main()
