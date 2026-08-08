#!/usr/bin/env python3
"""
Thread-scaling experiment: QPS / speedup / parallel efficiency as
faiss.omp_set_num_threads is swept, at fixed recall targets.

This is the producer for the "thread_scaling" node that plot_router_paper.py
consumes (plot_thread_scaling_grid / _speedup / _efficiency_summary and
table_thread_scaling). It used to live inside bench_router_paper.py as
--benchmark thread_scaling; it is split out here so that re-running it does
not touch the rest of the (expensive, already-validated) router pipeline and
does not overwrite results_router/log_router_<ds>.txt.

Each index is *loaded* from --index-dir; nothing is built. Run the router job
first (benchs/router_hpc.sbatch) so the indexes exist. Results are merged into
the existing results_router/results_<dataset>.json under the "thread_scaling"
key; every other key in that file is preserved.

Operating points come from the stored recall@10 curve when the JSON already has
one (the usual case after a router run), so the sweep costs only the timed runs.
Pass --refresh-curve to re-measure the curve instead.

Usage:
  python benchs/bench_thread_scaling.py --dataset sift1m
  python benchs/bench_thread_scaling.py --dataset all --threads 1 2 4 8 16 32
  python benchs/bench_thread_scaling.py --dataset gist1m --index-type suco hnsw32
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with custom index support first.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_router_paper import (  # noqa: E402
    ALL_DATASETS,
    ALL_INDEX_TYPES,
    BIGANN_SCALING_DATASETS,
    BUILDERS,
    DEFAULT_INDEX_TYPES,
    INDEX_FAMILIES,
    SEARCH_FACTORY,
    SUCO_NSUBSPACES_OVERRIDE,
    load_dataset,
    pick_param_for_recall,
    recall_time_curve,
    resolve_suco_nsubspaces,
)

# Sweep configuration. THREAD_COUNTS[0] is the speedup/efficiency baseline.
THREAD_COUNTS = (1, 2, 4, 8, 16, 32)
THREAD_RECALL_TARGETS = (0.90, 0.95)
THREAD_K = 10                       # measure recall@k=10 throughput
N_WARMUP = 2                        # warmup searches at each thread count


def index_path_for(dataset, kind, index_dir, d):
    """Mirror bench_router_paper's on-disk naming (SuCo carries its Ns tag)."""
    if kind == "suco":
        suco_n, _ = resolve_suco_nsubspaces(dataset, d)
        return os.path.join(index_dir, f"{dataset}_suco_ns{suco_n}.idx")
    return os.path.join(index_dir, f"{dataset}_{kind}.idx")


def run_thread_scaling(idx, kind, xq, gt, k=THREAD_K, recall_curve=None,
                       thread_counts=THREAD_COUNTS,
                       recall_targets=THREAD_RECALL_TARGETS):
    """Sweep faiss.omp_set_num_threads ∈ thread_counts at each recall target.
    Reports per-thread QPS, ms/query (batch-amortized), speedup vs the first
    thread count and parallel efficiency (speedup / threads). Restores the
    caller's thread count on exit."""
    factory, params, _ = SEARCH_FACTORY[kind]
    if recall_curve is None:
        recall_curve = recall_time_curve(idx, kind, xq, gt, k, factory, params,
                                         n_runs=1)

    try:
        prev_threads = int(faiss.omp_get_max_threads())
    except Exception:
        prev_threads = -1

    nq = int(xq.shape[0])
    out = {}
    try:
        for target in recall_targets:
            chosen = pick_param_for_recall(recall_curve, target)
            if chosen is None:
                print(f"  {kind}: recall≥{target} unreachable on the k={k} "
                      f"curve — skipping this target")
                continue
            search_fn = factory(chosen["param"])

            per_target = {
                "param": chosen["param"],
                "achieved_recall": chosen["recall"],
                "target_recall": float(target),
                "k": int(k),
                "nq": nq,
                "by_threads": {},
            }
            qps_base = None
            for t in thread_counts:
                try:
                    faiss.omp_set_num_threads(int(t))
                except Exception as e:
                    per_target["by_threads"][str(t)] = {"error": f"omp_set: {e}"}
                    continue
                # Warmup at this thread count (spins up the OMP team, warms cache).
                for _ in range(N_WARMUP):
                    search_fn(idx, xq[: min(50, nq)], k)

                t0 = time.perf_counter()
                _, _ = search_fn(idx, xq, k)
                elapsed = time.perf_counter() - t0
                qps = nq / elapsed if elapsed > 0 else 0.0
                ms_per_q = (elapsed / nq) * 1000.0
                if t == thread_counts[0]:
                    qps_base = qps
                speedup = (qps / qps_base) if (qps_base and qps_base > 0) else None
                eff = (speedup / t) if (speedup is not None and t > 0) else None
                per_target["by_threads"][str(t)] = {
                    "threads":      int(t),
                    "qps":          round(float(qps),      2),
                    "ms_per_query": round(float(ms_per_q), 6),
                    "speedup_vs_t1":       round(float(speedup), 3) if speedup is not None else None,
                    "parallel_efficiency": round(float(eff),     3) if eff is not None else None,
                }
                print(f"  {kind} threads={t} @recall≥{target}: "
                      f"qps={qps:.0f} ms/q={ms_per_q:.4f} "
                      f"speedup={speedup if speedup is None else round(speedup, 2)} "
                      f"eff={eff if eff is None else round(eff, 2)}",
                      flush=True)
            out[f"r{int(round(target * 100))}"] = per_target
    finally:
        if prev_threads > 0:
            try:
                faiss.omp_set_num_threads(prev_threads)
            except Exception:
                pass
    return out


def run_dataset(dataset, index_types, data_dir, index_dir, output_dir,
                thread_counts, recall_targets, refresh_curve, max_queries):
    print(f"\n{'#'*70}\n# Thread scaling: {dataset.upper()}\n{'#'*70}", flush=True)

    out_path = os.path.join(output_dir, f"results_{dataset}.json")
    all_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                all_results = json.load(f)
        except Exception as e:
            print(f"  Could not read {out_path} ({e}) — starting a fresh file")
            all_results = {}
    else:
        print(f"  Note: {out_path} does not exist yet; a thread-scaling-only "
              f"results file will be created.")

    print(f"\nLoading {dataset}...", flush=True)
    t0 = time.time()
    xb, xq, gt = load_dataset(dataset, data_dir, index_dir)
    d, n = int(xb.shape[1]), int(xb.shape[0])
    print(f"  Loaded in {time.time()-t0:.1f}s: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")

    if max_queries and xq.shape[0] > max_queries:
        print(f"  Capping queries {xq.shape[0]} -> {max_queries} (--max-queries)")
        xq, gt = xq[:max_queries], gt[:max_queries]

    # The base vectors are only needed to derive d and to satisfy loaders that
    # return them; the sweep itself searches an already-built index.
    del xb
    gc.collect()

    all_results.setdefault("dataset", dataset)
    all_results.setdefault("n", n)
    all_results.setdefault("d", d)
    thread_results = dict(all_results.get("thread_scaling", {}) or {})
    stored_curves = all_results.get(f"recall_k{THREAD_K}", {}) or {}

    for kind in index_types:
        if kind not in BUILDERS:
            print(f"  Unknown index type {kind!r}, skipping")
            continue
        label = BUILDERS[kind][0]
        idx_path = index_path_for(dataset, kind, index_dir, d)
        if not os.path.exists(idx_path):
            print(f"\n--- {label}: MISSING index {idx_path} — skipping "
                  f"(run benchs/router_hpc.sbatch first) ---")
            continue

        print(f"\n--- {label} thread scaling (threads={list(thread_counts)}) ---")
        print(f"    loading {idx_path}", flush=True)
        try:
            idx = faiss.read_index(idx_path)
        except Exception as e:
            print(f"  {label}: LOAD FAILED — {e}")
            continue

        curve = None if refresh_curve else stored_curves.get(label)
        if curve:
            print(f"    reusing stored recall_k{THREAD_K} curve "
                  f"({len(curve)} operating points)")
        else:
            print(f"    no stored recall_k{THREAD_K} curve — measuring one")

        try:
            thread_results[label] = run_thread_scaling(
                idx, kind, xq, gt, k=THREAD_K, recall_curve=curve,
                thread_counts=thread_counts, recall_targets=recall_targets,
            )
        except Exception as e:
            print(f"  {label} thread_scaling FAILED: {e}")
            traceback.print_exc()

        del idx
        gc.collect()

        # Checkpoint after every index so a walltime kill keeps what ran.
        all_results["thread_scaling"] = thread_results
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults merged into {out_path}")
    del xq, gt
    gc.collect()
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Thread-scaling sweep for the router paper")
    ap.add_argument("--data-dir",   default=os.environ.get("DATA_DIR", "data/"))
    ap.add_argument("--index-dir",  default=os.environ.get("INDEX_DIR", "indices/"))
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to benchs/results_router (merges into results_<ds>.json).")
    ap.add_argument("--dataset",    nargs="+", default=["all"],
                    choices=ALL_DATASETS + BIGANN_SCALING_DATASETS
                            + ["all", "bigann_scaling"])
    ap.add_argument("--index-type", nargs="+", default=DEFAULT_INDEX_TYPES,
                    choices=ALL_INDEX_TYPES + ["all"])
    ap.add_argument("--families", nargs="+", default=None,
                    choices=sorted(INDEX_FAMILIES),
                    help="Index-family groups (graph/collision/quant). Overrides --index-type.")
    ap.add_argument("--threads", nargs="+", type=int, default=list(THREAD_COUNTS),
                    help="Thread counts to sweep. The first is the speedup baseline. "
                         "Default 1 2 4 8 16 32 (the published tables used 1 2 4 8 16).")
    ap.add_argument("--recall-targets", nargs="+", type=float,
                    default=list(THREAD_RECALL_TARGETS),
                    help="Recall targets that fix the operating point. Default 0.90 0.95.")
    ap.add_argument("--refresh-curve", action="store_true",
                    help="Re-measure the recall@10 curve instead of reusing the stored one.")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="Cap the query set (0 = use all). For smoke tests only — "
                         "QPS is not comparable across different nq.")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_router")

    if args.families:
        chosen = {k for f in args.families for k in INDEX_FAMILIES[f]}
    else:
        chosen = set(ALL_INDEX_TYPES if "all" in args.index_type else args.index_type)
    index_types = [k for k in ALL_INDEX_TYPES if k in chosen]

    datasets = []
    for ds in args.dataset:
        if ds == "all":
            datasets.extend(ALL_DATASETS)
        elif ds == "bigann_scaling":
            datasets.extend(BIGANN_SCALING_DATASETS)
        else:
            datasets.append(ds)
    seen = set()
    datasets = [ds for ds in datasets if not (ds in seen or seen.add(ds))]

    # An oversubscribed measurement (threads > cores in the cgroup) is noise,
    # not scaling — drop those points rather than publishing them.
    try:
        avail = int(faiss.omp_get_max_threads())
    except Exception:
        avail = 0
    threads = sorted(set(int(t) for t in args.threads if t > 0))
    if avail:
        kept = [t for t in threads if t <= avail]
        if len(kept) != len(threads):
            print(f"WARNING: dropping thread counts > {avail} available cores: "
                  f"{[t for t in threads if t > avail]}")
        threads = kept or [1]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir:       {args.data_dir}")
    print(f"Index dir:      {args.index_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Datasets:       {datasets}")
    print(f"Indexes:        {index_types}")
    print(f"Thread counts:  {threads}   (cores available: {avail})")
    print(f"Recall targets: {args.recall_targets}")

    for ds in datasets:
        try:
            run_dataset(ds, index_types, args.data_dir, args.index_dir,
                        args.output_dir, tuple(threads),
                        tuple(args.recall_targets), args.refresh_curve,
                        int(args.max_queries))
        except Exception as e:
            print(f"\nERROR processing {ds}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
