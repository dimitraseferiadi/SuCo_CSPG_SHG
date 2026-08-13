#!/usr/bin/env python3
"""
benchs/bench_m2_bandwidth.py

Direct evidence for the claim that FAISS's distance path is memory-bandwidth
bound, and that this is what compresses the margin available to distance-count
reductions.

The paper currently asserts the mechanism (batch-4 L2 kernels over a flat
node-major array) and supports it indirectly (parallel efficiency falls with d).
Neither establishes that the path is bandwidth bound rather than, say, limited
by a fixed serial fraction or by cache capacity.  This script closes that gap
with three experiments, none of which requires modifying FAISS.

E1 -- zero-padding ablation (the decisive one).
    Appending zero columns to base and query vectors leaves every squared L2
    distance exactly unchanged.  The graph built on the padded data is therefore
    identical, the search visits the same nodes in the same order, and the number
    of distance evaluations per query is identical -- but each evaluation streams
    d/d0 times as many bytes.  Query time as a function of d at fixed algorithmic
    work is thus a pure measurement of how the distance path scales with bytes:

        d(log t)/d(log d) -> 1   the path is byte-bound (streaming/bandwidth)
        d(log t)/d(log d) -> 0   the path is bound by something other than bytes

    A slope near 1 is the claim the paper needs; anything well below it means the
    claim must be weakened.  The script verifies the invariance assumption rather
    than trusting it, by asserting that recall and the returned neighbour sets
    are identical across padding widths.

E2 -- achieved bandwidth against a measured ceiling.
    A STREAM-triad estimate of attainable DRAM bandwidth on the node, plus the
    bandwidth actually achieved by fvec_L2sqr_ny over an out-of-cache array.  E1's
    achieved GB/s is then reported as a fraction of this ceiling: a kernel running
    at a large fraction of peak is bandwidth bound by definition, and no reduction
    in distance count can help it further than the bytes it removes.

E3 -- roofline placement.
    Arithmetic intensity of the L2 kernel is 2 flops per 4 bytes streamed
    (one subtract, one fused multiply-add per float loaded), i.e. 0.5 flop/byte
    for a cold stream.  Reporting the measured operational point against the
    machine balance shows directly which side of the ridge the kernel sits on.

Outputs
  <out>/m2_padding_<dataset>.csv   E1: per-d recall, QPS, bytes/s, slope inputs
  <out>/m2_bandwidth.json          E2/E3: ceilings, achieved fractions, slope fit
  stdout                           a summary table and the fitted slope

Usage
  python benchs/bench_m2_bandwidth.py --dataset sift1m --data-dir <dir>
  python benchs/bench_m2_bandwidth.py --dataset sift1m --pad-dims 128 256 512 1024
  python benchs/bench_m2_bandwidth.py --dataset gist1m --index hnsw32 --threads 32
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with custom index support first.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_datasets import get_dataset  # noqa: E402

BYTES_PER_FLOAT = 4


# ---------------------------------------------------------------------------
# E2: attainable bandwidth
# ---------------------------------------------------------------------------

def stream_copy_gbs(size_mb=2048, repeats=5, threads=1):
    """STREAM Copy (a[:] = b[:]) over arrays far larger than last-level cache.

    Copy rather than Triad because numpy expresses a triad only via a temporary
    (`b + s*c` materialises `s*c`), which inflates the traffic actually moved and
    would understate the ceiling.  Copy moves exactly 2n floats in one pass and
    has no such ambiguity.

    `threads` slices the copy across a thread pool.  numpy releases the GIL
    inside copyto, so this measures genuinely concurrent DRAM traffic and gives a
    ceiling comparable with a multi-threaded FAISS search.  Best of `repeats` is
    reported, following the STREAM convention for a ceiling.
    """
    n = int(size_mb * 1024 * 1024 / BYTES_PER_FLOAT / 2)
    b = np.ones(n, dtype=np.float32)
    a = np.empty(n, dtype=np.float32)

    if threads <= 1:
        def do_copy():
            np.copyto(a, b)
    else:
        from concurrent.futures import ThreadPoolExecutor
        bounds = np.linspace(0, n, threads + 1).astype(int)
        pool = ThreadPoolExecutor(max_workers=threads)

        def do_copy():
            list(pool.map(
                lambda j: np.copyto(a[bounds[j]:bounds[j + 1]],
                                    b[bounds[j]:bounds[j + 1]]),
                range(threads)))

    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        do_copy()
        dt = time.perf_counter() - t0
        best = max(best, 2 * n * BYTES_PER_FLOAT / dt / 1e9)
    return best


def fvec_l2sqr_ny_gbs(d=128, size_mb=2048, repeats=5):
    """Bandwidth achieved by FAISS's own one-to-many L2 kernel, single threaded.

    This is the same kernel family the graph indexes call, run over a purely
    sequential layout, so it is an upper bound on what a random-access graph
    traversal can achieve with the same arithmetic.  It is single threaded, so it
    must be compared against the single-threaded copy ceiling, not the pooled one.
    """
    ny = int(size_mb * 1024 * 1024 / (BYTES_PER_FLOAT * d))
    xb = np.random.rand(ny, d).astype("float32")
    xq = np.random.rand(1, d).astype("float32")
    out = np.empty(ny, dtype="float32")
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        faiss.fvec_L2sqr_ny(faiss.swig_ptr(out), faiss.swig_ptr(xq),
                            faiss.swig_ptr(xb), d, ny)
        dt = time.perf_counter() - t0
        best = max(best, ny * d * BYTES_PER_FLOAT / dt / 1e9)
    return best, ny


# ---------------------------------------------------------------------------
# E1: zero-padding ablation
# ---------------------------------------------------------------------------

def pad_to(x, d_target):
    """Right-pad with zero columns. Squared L2 between any pair is unchanged."""
    n, d = x.shape
    if d_target == d:
        return np.ascontiguousarray(x)
    assert d_target > d, "padding can only increase d"
    out = np.zeros((n, d_target), dtype="float32")
    out[:, :d] = x
    return np.ascontiguousarray(out)


def build_hnsw(xb, d, M, efc):
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    idx.hnsw.efConstruction = efc
    t0 = time.perf_counter()
    idx.add(xb)
    return idx, time.perf_counter() - t0


# Graph fields that fully determine HNSW traversal, with their element types.
_HNSW_VECS = [
    ("assign_probas", "float64"),
    ("cum_nneighbor_per_level", "int32"),
    ("levels", "int32"),
    ("offsets", "uint64"),
    ("neighbors", "int32"),
]


def extract_graph(idx):
    """Snapshot the HNSW topology as plain numpy arrays."""
    g = {name: faiss.vector_to_array(getattr(idx.hnsw, name))
         for name, _ in _HNSW_VECS}
    g["entry_point"] = int(idx.hnsw.entry_point)
    g["max_level"] = int(idx.hnsw.max_level)
    return g


def build_hnsw_with_graph(xb, d, M, graph):
    """An HNSW over `xb` carrying a transplanted, pre-built topology.

    HNSW level assignment is randomised and insertion is concurrent, so two
    builds over the same distances do not produce the same graph.  Rebuilding at
    each padded width would therefore confound the byte-width effect with
    graph-to-graph variation.  Transplanting one topology onto storage of a
    different width removes that variation by construction: the traversal visits
    the same nodes in the same order at every width, and the only thing that
    changes is how many bytes each distance evaluation streams.

    The vectors are added to the flat storage directly, bypassing graph
    construction; the topology then comes wholesale from `graph`.
    """
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    idx.storage.add(xb)
    idx.ntotal = idx.storage.ntotal
    idx.is_trained = True
    for name, dtype in _HNSW_VECS:
        faiss.copy_array_to_vector(
            np.ascontiguousarray(graph[name], dtype=dtype),
            getattr(idx.hnsw, name))
    idx.hnsw.entry_point = graph["entry_point"]
    idx.hnsw.max_level = graph["max_level"]
    return idx


def timed_search(idx, xq, k, ef, n_warmup=2, n_runs=5):
    idx.hnsw.efSearch = int(ef)
    for _ in range(n_warmup):
        idx.search(xq, k)
    times, I_ref = [], None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        D, I = idx.search(xq, k)
        times.append(time.perf_counter() - t0)
        if I_ref is None:
            I_ref = I
    return np.asarray(times), I_ref


def hnsw_ndis(idx, xq, k, ef):
    """Distance evaluations per query, from FAISS's own HNSW counter.

    Run untimed and single-threaded-safe: the counter is a global updated under
    a critical section, so it must never share a run with a timing measurement.
    """
    idx.hnsw.efSearch = int(ef)
    faiss.cvar.hnsw_stats.reset()
    idx.search(xq, k)
    return faiss.cvar.hnsw_stats.ndis / xq.shape[0]


def recall_at_k(I, gt, k):
    kk = min(k, gt.shape[1], I.shape[1])
    return float(np.mean([
        len(set(I[i, :kk].tolist()) & set(gt[i, :kk].tolist())) / kk
        for i in range(I.shape[0])
    ]))


def run_padding_ablation(xb0, xq0, gt, pad_dims, M, efc, ef, k, n_runs):
    """Time one transplanted-graph HNSW per padded width; assert invariance."""
    # One build, at the native width; every padded index reuses its topology.
    base_idx, t_build = build_hnsw(xb0, xb0.shape[1], M, efc)
    graph = extract_graph(base_idx)
    del base_idx

    rows, ref = [], {}
    for d_pad in pad_dims:
        xb = pad_to(xb0, d_pad)
        xq = pad_to(xq0, d_pad)
        idx = build_hnsw_with_graph(xb, d_pad, M, graph)

        ndis = hnsw_ndis(idx, xq, k, ef)
        times, I = timed_search(idx, xq, k, ef, n_runs=n_runs)
        rec = recall_at_k(I, gt, k)

        nq = xq.shape[0]
        t_mean = float(times.mean())
        qps = nq / t_mean
        bytes_streamed = ndis * nq * d_pad * BYTES_PER_FLOAT
        gbs = bytes_streamed / t_mean / 1e9

        if not ref:
            ref = {"ndis": ndis, "recall": rec, "I": I}
            invariant = True
        else:
            # The padding argument is only valid if the search really is
            # unchanged.  Check it rather than assume it.
            invariant = (
                abs(ndis - ref["ndis"]) / max(ref["ndis"], 1) < 1e-3
                and abs(rec - ref["recall"]) < 1e-6
                and np.array_equal(I, ref["I"])
            )

        rows.append({
            "d": d_pad,
            "build_s": round(t_build, 3) if d_pad == pad_dims[0] else 0.0,
            "recall": round(rec, 6),
            "ndis_per_query": round(ndis, 1),
            "ms_per_query": round(t_mean / nq * 1000.0, 6),
            "ms_std": round(float(times.std(ddof=1)) / nq * 1000.0, 6),
            "qps": round(qps, 1),
            "achieved_gbs": round(gbs, 2),
            "invariant": bool(invariant),
        })
        print(f"  d={d_pad:5d}  recall={rec:.4f}  ndis/q={ndis:8.1f}  "
              f"ms/q={t_mean/nq*1000:8.4f}  {gbs:7.2f} GB/s  "
              f"{'ok' if invariant else 'INVARIANCE BROKEN'}")
        del idx, xb, xq
    return rows


def fit_slope(rows):
    """Least-squares slope of log(ms/query) against log(d).

    1.0 means time is proportional to bytes streamed; 0.0 means bytes do not
    govern the cost at all.
    """
    good = [r for r in rows if r["invariant"]]
    if len(good) < 2:
        return None
    x = np.log(np.array([r["d"] for r in good], dtype=float))
    y = np.log(np.array([r["ms_per_query"] for r in good], dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (float(res[0]) / ss_tot if len(res) and ss_tot > 0 else 0.0)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2,
            "n_points": len(good)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--data-dir", default=os.environ.get("ANN_DATA_DIR", "data"))
    ap.add_argument("--index-dir", default=None, help="ground-truth cache")
    ap.add_argument("--out-dir", default="benchs/results_router")
    ap.add_argument("--pad-dims", type=int, nargs="+", default=None,
                    help="widths to pad to; default d, 2d, 4d, 8d")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=128)
    ap.add_argument("--ef", type=int, default=100,
                    help="efSearch, held fixed across widths")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--stream-mb", type=int, default=2048)
    ap.add_argument("--skip-padding", action="store_true")
    args = ap.parse_args()

    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    nthreads = faiss.omp_get_max_threads()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"threads={nthreads}")
    print("\nE2: attainable bandwidth")
    copy_1t = stream_copy_gbs(args.stream_mb, threads=1)
    copy_nt = stream_copy_gbs(args.stream_mb, threads=nthreads)
    print(f"  STREAM copy, 1 thread       : {copy_1t:7.2f} GB/s")
    print(f"  STREAM copy, {nthreads:<2d} threads     : {copy_nt:7.2f} GB/s "
          f"({copy_nt/copy_1t:.2f}x the single-thread figure)")
    kern = {}
    for d in (32, 128, 512):
        gbs, ny = fvec_l2sqr_ny_gbs(d=d, size_mb=args.stream_mb)
        kern[d] = gbs
        print(f"  fvec_L2sqr_ny d={d:<4d} (1 thr) : {gbs:7.2f} GB/s "
              f"({100*gbs/copy_1t:5.1f}% of the 1-thread ceiling, ny={ny})")

    out = {
        "threads": nthreads,
        "stream_copy_gbs_1t": copy_1t,
        "stream_copy_gbs_nt": copy_nt,
        "fvec_L2sqr_ny_gbs_1t": kern,
        # 1 subtract + 1 FMA per float loaded = 2 flops per 4 bytes.
        "kernel_arithmetic_intensity_flop_per_byte": 0.5,
    }

    if not args.skip_padding:
        ds = get_dataset(args.dataset, args.data_dir,
                         gt_cache_dir=args.index_dir or args.out_dir)
        xb0 = ds.get_database()
        xq0 = ds.get_queries()
        gt = ds.get_groundtruth(k=100, xb=xb0, xq=xq0)[: xq0.shape[0]]
        d0 = xb0.shape[1]
        pads = args.pad_dims or [d0, 2 * d0, 4 * d0, 8 * d0]

        print(f"\nE1: zero-padding ablation on {args.dataset} "
              f"(n={xb0.shape[0]}, d={d0}, M={args.M}, efC={args.efc}, "
              f"efSearch={args.ef})")
        rows = run_padding_ablation(xb0, xq0, gt, pads, args.M, args.efc,
                                    args.ef, args.k, args.n_runs)
        fit = fit_slope(rows)
        out["padding"] = {"dataset": args.dataset, "d0": d0, "M": args.M,
                          "efc": args.efc, "ef": args.ef, "k": args.k,
                          "rows": rows, "fit": fit}
        if fit:
            print(f"\n  d(log t)/d(log d) = {fit['slope']:.3f} "
                  f"(R^2={fit['r2']:.4f}, {fit['n_points']} widths)")
            print("  1.0 => time is set by bytes streamed (bandwidth bound); "
                  "0.0 => bytes are not the constraint")
            peak = max(r["achieved_gbs"] for r in rows if r["invariant"])
            print(f"  peak achieved during graph search: {peak:.2f} GB/s "
                  f"({100*peak/copy_nt:.1f}% of the {nthreads}-thread ceiling)")
            out["padding"]["peak_achieved_gbs"] = peak
            out["padding"]["peak_fraction_of_ceiling_nt"] = peak / copy_nt

        csv_path = os.path.join(args.out_dir, f"m2_padding_{args.dataset}.csv")
        with open(csv_path, "w") as fh:
            cols = list(rows[0].keys())
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"\nwrote {csv_path}")

    json_path = os.path.join(args.out_dir, "m2_bandwidth.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
