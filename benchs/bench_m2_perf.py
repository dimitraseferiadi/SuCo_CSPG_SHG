#!/usr/bin/env python3
"""
benchs/bench_m2_perf.py

A search driver stripped down far enough that `perf stat` can wrap it, so DRAM
traffic is measured rather than inferred.

Why this exists.  `bench_m2_bandwidth.py` reports achieved bandwidth as
`ndis * nq * d * 4 / t` -- the bytes the *vectors* occupy.  That is not the
traffic.  The base-layer loop in faiss/impl/HNSW.cpp also reads a 64-entry
int32 neighbour list per popped node and issues a prefetch and a set into a
1 MB per-thread visited table for every neighbour in range, and none of that
scales with d.  Omitting it understates traffic at the widths where the
paper's claim is weakest and barely matters at the widest padded ones, which
is exactly the shape needed to manufacture a rising "achieved bandwidth".

Counting rather than inferring needs a hardware counter, and a counter needs a
process that does nothing else.  Hence this file.

Measurement protocol.  perf cannot be started partway through the process
without version-specific control plumbing, so the build and the load would land
in the counts.  Instead run the same command twice with different `--passes`
and difference the counts:

    perf stat -x, -e <events> -o a.csv -- python bench_m2_perf.py --passes 1
    perf stat -x, -e <events> -o b.csv -- python bench_m2_perf.py --passes 11
    traffic per pass = (b - a) / 10

Everything that is not a search pass -- interpreter start-up, dataset load,
graph build, transplant -- appears identically in both and cancels exactly.
`--graph-cache` keeps the two runs on the same topology and skips the rebuild.

Events.  Uncore IMC counters are the real answer and usually need
perf_event_paranoid <= 0 or CAP_PERFMON, which a shared cluster rarely grants:

    uncore_imc/cas_count_read/,uncore_imc/cas_count_write/     (x 64 B)

The core-PMU fallback works unprivileged at paranoid=2 and is close enough to
settle a 2x question:

    mem_load_retired.l3_miss,LLC-load-misses,LLC-store-misses  (x 64 B)

m2b_hpc.sbatch probes for both and uses whichever is available.

Usage
  python benchs/bench_m2_perf.py --dataset sift1m --pad-dim 128 --passes 11 \
      --graph-cache /tmp/g_sift1m.npz
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with custom index support first.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_datasets import get_dataset            # noqa: E402
from bench_m2_bandwidth import (                  # noqa: E402
    _HNSW_VECS, build_hnsw, build_hnsw_with_graph, extract_graph, pad_to,
)


def load_or_build_graph(xb0, M, efc, cache):
    """One topology, shared by both perf runs and by every width.

    Rebuilding would give the two runs different graphs -- level assignment is
    randomised -- and the difference of their counters would then include the
    difference between two graphs rather than the cost of ten search passes.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        g = {name: z[name] for name, _ in _HNSW_VECS}
        g["entry_point"] = int(z["entry_point"])
        g["max_level"] = int(z["max_level"])
        print(f"graph: loaded {cache}", flush=True)
        return g

    t0 = time.perf_counter()
    idx, _ = build_hnsw(xb0, xb0.shape[1], M, efc)
    g = extract_graph(idx)
    del idx
    print(f"graph: built in {time.perf_counter()-t0:.1f}s", flush=True)
    if cache:
        np.savez(cache, **g)
        print(f"graph: cached to {cache}", flush=True)
    return g


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--data-dir", default=os.environ.get("ANN_DATA_DIR", "data"))
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out-dir", default="benchs/results_router")
    ap.add_argument("--pad-dim", type=int, default=None,
                    help="width to pad to; default the dataset's own d")
    ap.add_argument("--passes", type=int, default=1,
                    help="search passes; run twice and difference the counts")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=128)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--graph-cache", default=None)
    args = ap.parse_args()

    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    ds = get_dataset(args.dataset, args.data_dir,
                     gt_cache_dir=args.index_dir or args.out_dir)
    xb0 = ds.get_database()
    xq0 = ds.get_queries()
    d_pad = args.pad_dim or xb0.shape[1]

    graph = load_or_build_graph(xb0, args.M, args.efc, args.graph_cache)
    xb = pad_to(xb0, d_pad)
    xq = pad_to(xq0, d_pad)
    del xb0
    idx = build_hnsw_with_graph(xb, d_pad, args.M, graph)
    idx.hnsw.efSearch = int(args.ef)

    nq = xq.shape[0]
    print(f"searching: d={d_pad} nq={nq} ef={args.ef} passes={args.passes} "
          f"threads={faiss.omp_get_max_threads()}", flush=True)

    t0 = time.perf_counter()
    for _ in range(args.passes):
        idx.search(xq, args.k)
    dt = time.perf_counter() - t0

    # Only the difference between two invocations is meaningful; this figure is
    # printed as a cross-check that the two runs did the same work per pass.
    print(f"total {dt:.4f}s over {args.passes} passes "
          f"({dt/args.passes/nq*1000:.6f} ms/query, pass 1 is cold)")


if __name__ == "__main__":
    main()
