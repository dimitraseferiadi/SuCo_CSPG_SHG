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


def fit_affine(rows):
    """Least-squares fit of t = a + b*d, in seconds and per dimension.

    The log-log slope alone cannot support the claim the paper makes from it.
    A slope below 1 is produced equally by a fixed per-query cost and by an
    achievable bandwidth that itself rises with the width of each read, and the
    two are the same function: t = c*d/BW(d) with BW(d) = BW_inf*d/(d+d0) is
    exactly t = a + b*d.  Padding cannot separate them.

    What padding does identify is `b`: the marginal cost of a byte.  Converted
    to GB/s at fixed `ndis` it gives the bandwidth the path achieves on the
    bytes it actually adds, which is the bandwidth-bound quantity -- unlike the
    average GB/s, which is dragged down at small d by everything in `a`.
    """
    good = [r for r in rows if r["invariant"]]
    if len(good) < 2:
        return None
    d = np.array([r["d"] for r in good], dtype=float)
    t = np.array([r["ms_per_query"] for r in good], dtype=float) * 1e-3
    ndis = float(good[0]["ndis_per_query"])
    A = np.vstack([d, np.ones_like(d)]).T
    (b, a), res, *_ = np.linalg.lstsq(A, t, rcond=None)
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - (float(res[0]) / ss_tot if len(res) and ss_tot > 0 else 0.0)

    # Marginal bandwidth per adjacent pair, which shows whether b is constant
    # (a single ceiling) or itself rising with the width of each read.
    seg = []
    for i in range(len(good) - 1):
        db = ndis * (d[i + 1] - d[i]) * BYTES_PER_FLOAT
        seg.append({"d_lo": int(d[i]), "d_hi": int(d[i + 1]),
                    "marginal_gbs": round(db / (t[i + 1] - t[i]) / 1e9, 3)})
    return {
        "fixed_s_per_query": float(a),
        "per_dim_s_per_query": float(b),
        "r2": r2,
        "marginal_gbs_global": float(ndis * BYTES_PER_FLOAT / b / 1e9),
        "fixed_fraction_at_d": {int(dd): float(a / (a + b * dd)) for dd in d},
        "segments": seg,
    }


# ---------------------------------------------------------------------------
# C: does achieved bandwidth saturate in threads?
# ---------------------------------------------------------------------------

def run_thread_sweep(xb0, xq0, gt, pad_dims, M, efc, ef, k, n_runs, threads,
                     nq):
    """Achieved bandwidth against thread count, at fixed width.

    This is the direct test of saturation that the padding ablation does not
    perform.  A path limited by a shared resource stops gaining bandwidth as
    threads are added; a path limited by per-core memory-level parallelism
    keeps gaining.  Both are memory-bound and neither is compute-bound, but
    only the first may be described as a bandwidth ceiling, and only the first
    licenses a "% of STREAM" statement.

    Queries are subsampled so that the single-thread points stay affordable at
    the widest padding; the subsample is identical across thread counts, so the
    speedup and the achieved bandwidth are comparable within a width.
    """
    # Take the count back from the array: a dataset with fewer queries than the
    # requested subsample (GIST1M and MSong ship 1000 and 200) otherwise leaves
    # `nq` above the number actually searched, and every per-query time and
    # bandwidth derived from it is wrong by that ratio.
    xq0 = xq0[:nq]
    nq = int(xq0.shape[0])
    gt = gt[:nq]
    # A dataset may hold fewer queries than the subsample asks for -- GIST1M
    # and OpenAI1M ship 1000, MSong and Enron 200 -- and normalising by the
    # request rather than by what the slice returned scales ms/query and
    # achieved GB/s by nq_requested/nq_actual. On GIST1M that is exactly 2x.
    nq = int(xq0.shape[0])
    # Ascending, so the first timing is the single-thread baseline that
    # speedup and efficiency are measured against.
    threads = sorted(int(t) for t in threads)
    base_idx, _ = build_hnsw(xb0, xb0.shape[1], M, efc)
    graph = extract_graph(base_idx)
    del base_idx

    rows = []
    for d_pad in pad_dims:
        xb = pad_to(xb0, d_pad)
        xq = pad_to(xq0, d_pad)
        idx = build_hnsw_with_graph(xb, d_pad, M, graph)
        ndis = hnsw_ndis(idx, xq, k, ef)
        t1 = None
        for t in threads:
            faiss.omp_set_num_threads(int(t))
            times, I = timed_search(idx, xq, k, ef, n_runs=n_runs)
            t_mean = float(times.mean())
            if t1 is None:
                t1 = t_mean
            gbs = ndis * nq * d_pad * BYTES_PER_FLOAT / t_mean / 1e9
            rows.append({
                "d": d_pad, "threads": int(t), "nq": int(nq),
                "recall": round(recall_at_k(I, gt, k), 6),
                "ndis_per_query": round(ndis, 1),
                "ms_per_query": round(t_mean / nq * 1000.0, 6),
                "speedup_vs_1t": round(t1 / t_mean, 3),
                "efficiency": round(t1 / t_mean / t, 4),
                "achieved_gbs": round(gbs, 2),
            })
            print(f"  d={d_pad:5d}  t={t:3d}  ms/q={t_mean/nq*1000:9.4f}  "
                  f"speedup={t1/t_mean:6.2f}x  eta={t1/t_mean/t:5.3f}  "
                  f"{gbs:7.2f} GB/s")
        del idx, xb, xq
    return rows


# ---------------------------------------------------------------------------
# D: the elasticity the conclusions actually rest on
# ---------------------------------------------------------------------------

def run_ef_sweep(xb0, xq0, gt, M, efc, ef_list, k, n_runs):
    """d(log t)/d(log ndis): what removing a distance evaluation actually buys.

    The padding ablation measures d(log t)/d(log d) at fixed distance count.
    Every conclusion drawn from it -- that CSPG's routing and SHG's shortcut
    cannot convert a distance-count reduction into wall-clock -- is about
    d(log t)/d(log ndis) at fixed d.  Those are different derivatives, and a
    bandwidth-bound path predicts the second is 1: fewer evaluations read
    proportionally fewer bytes.

    Counting and timing are run as separate passes, since the HNSW counter is
    global and updated under a critical section.
    """
    idx, _ = build_hnsw(xb0, xb0.shape[1], M, efc)
    nq = xq0.shape[0]
    rows = []
    for ef in ef_list:
        ndis = hnsw_ndis(idx, xq0, k, ef)
        times, I = timed_search(idx, xq0, k, ef, n_runs=n_runs)
        t_mean = float(times.mean())
        rows.append({
            "ef": int(ef),
            "ndis_per_query": round(ndis, 1),
            "recall": round(recall_at_k(I, gt, k), 6),
            "ms_per_query": round(t_mean / nq * 1000.0, 6),
            "qps": round(nq / t_mean, 1),
        })
        print(f"  ef={ef:5d}  ndis/q={ndis:9.1f}  recall={rows[-1]['recall']:.4f}"
              f"  ms/q={t_mean/nq*1000:9.5f}")
    del idx
    return rows


def fit_ndis_elasticity(rows, recall_lo=0.95, recall_hi=0.995):
    """Slope of log(ms/query) against log(ndis/query).

    Reported twice: over the whole sweep, and over the recall band in which the
    cross-algorithm comparisons are actually made, since a fixed per-query
    overhead flattens the slope at the cheap end where nothing is compared.
    """
    def slope(sel):
        if len(sel) < 3:
            return None
        x = np.log(np.array([r["ndis_per_query"] for r in sel], dtype=float))
        y = np.log(np.array([r["ms_per_query"] for r in sel], dtype=float))
        A = np.vstack([x, np.ones_like(x)]).T
        (s, _), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - (float(res[0]) / ss_tot if len(res) and ss_tot > 0 else 0.0)
        return {"slope": float(s), "r2": r2, "n_points": len(sel)}

    return {
        "all": slope(rows),
        "matched_recall_band": slope(
            [r for r in rows if recall_lo <= r["recall"] <= recall_hi]),
        "band": [recall_lo, recall_hi],
    }


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
    ap.add_argument("--skip-stream", action="store_true",
                    help="skip the numpy STREAM proxy; m2_ceilings.c measures "
                         "the real thing, and this script's copy-based "
                         "estimate should not be quoted once it has run")
    ap.add_argument("--thread-sweep", type=int, nargs="+", default=None,
                    help="thread counts at which to re-time the padded search "
                         "(e.g. 1 2 4 8 16 32); tests saturation directly")
    ap.add_argument("--thread-sweep-dims", type=int, nargs="+", default=None,
                    help="widths for the thread sweep; default: native and 4x")
    ap.add_argument("--sweep-nq", type=int, default=2000,
                    help="queries subsampled for the thread sweep")
    ap.add_argument("--ef-sweep", type=int, nargs="+", default=None,
                    help="efSearch values at which to record ndis and time, "
                         "for d(log t)/d(log ndis) at native width")
    args = ap.parse_args()

    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    nthreads = faiss.omp_get_max_threads()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"threads={nthreads}")
    out = {"threads": nthreads}
    copy_nt = None

    if not args.skip_stream:
        print("\nE2: attainable bandwidth")
        print("  NOTE: this is a numpy copy, not STREAM, and a copy ceiling is "
              "the wrong\n        reference for a read-only random-access path."
              "  Prefer m2_ceilings.")
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
        out.update({
            "stream_copy_gbs_1t": copy_1t,
            "stream_copy_gbs_nt": copy_nt,
            "fvec_L2sqr_ny_gbs_1t": kern,
            "stream_proxy_caveat":
                "numpy copy, not STREAM; counts 2n bytes while moving 3n "
                "unless libc elects non-temporal stores; a copy ceiling "
                "understates a read-only path and a sequential ceiling "
                "overstates a dependent random-read one. Use m2_ceilings.",
        })

    def write_csv(name, rows):
        path = os.path.join(args.out_dir, f"{name}_{args.dataset}.csv")
        with open(path, "w") as fh:
            cols = list(rows[0].keys())
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {path}")

    need_data = (not args.skip_padding) or args.thread_sweep or args.ef_sweep
    if need_data:
        ds = get_dataset(args.dataset, args.data_dir,
                         gt_cache_dir=args.index_dir or args.out_dir)
        xb0 = ds.get_database()
        xq0 = ds.get_queries()
        gt = ds.get_groundtruth(k=100, xb=xb0, xq=xq0)[: xq0.shape[0]]
        d0 = xb0.shape[1]
        pads = args.pad_dims or [d0, 2 * d0, 4 * d0, 8 * d0]

    if not args.skip_padding:
        print(f"\nE1: zero-padding ablation on {args.dataset} "
              f"(n={xb0.shape[0]}, d={d0}, M={args.M}, efC={args.efc}, "
              f"efSearch={args.ef})")
        rows = run_padding_ablation(xb0, xq0, gt, pads, args.M, args.efc,
                                    args.ef, args.k, args.n_runs)
        fit = fit_slope(rows)
        aff = fit_affine(rows)
        out["padding"] = {"dataset": args.dataset, "d0": d0, "M": args.M,
                          "efc": args.efc, "ef": args.ef, "k": args.k,
                          "rows": rows, "fit": fit, "affine": aff}
        if fit:
            print(f"\n  d(log t)/d(log d) = {fit['slope']:.3f} "
                  f"(R^2={fit['r2']:.4f}, {fit['n_points']} widths)")
            peak = max(r["achieved_gbs"] for r in rows if r["invariant"])
            out["padding"]["peak_achieved_gbs"] = peak
            print(f"  peak average achieved: {peak:.2f} GB/s at d="
                  f"{max(r['d'] for r in rows if r['invariant'])}")
            if copy_nt:
                out["padding"]["peak_fraction_of_copy_proxy"] = peak / copy_nt
                print(f"    = {100*peak/copy_nt:.1f}% of the copy proxy -- do "
                      f"not quote this; use BW_rand(4d) from m2_ceilings")
        if aff:
            print(f"\n  t = {aff['fixed_s_per_query']*1e6:.2f} us + "
                  f"{aff['per_dim_s_per_query']*1e9:.3f} ns/dim   "
                  f"(R^2={aff['r2']:.5f})")
            print(f"  d-independent share of query time: " + "  ".join(
                f"d={d}:{100*f:.0f}%"
                for d, f in aff["fixed_fraction_at_d"].items()))
            print(f"  MARGINAL bandwidth (the bandwidth-bound quantity), "
                  f"global fit: {aff['marginal_gbs_global']:.2f} GB/s")
            for s in aff["segments"]:
                print(f"    {s['d_lo']:5d} -> {s['d_hi']:5d}: "
                      f"{s['marginal_gbs']:7.2f} GB/s")
            print("  Compare each against BW_rand(4d) at the matching width, "
                  "not against STREAM.")
        write_csv("m2_padding", rows)

    if args.thread_sweep:
        # Native and 4x by default: the contrast between a short burst and a
        # long one on the same graph is the whole point, since saturation is
        # expected to arrive at fewer threads as the vector widens.
        dims = args.thread_sweep_dims or [d0, 4 * d0]
        print(f"\nC: thread sweep at fixed width on {args.dataset} "
              f"(widths={dims}, nq={args.sweep_nq})")
        print("  Rising GB/s at 32 threads => not saturated; a flat tail => a "
              "shared-resource ceiling.")
        trows = run_thread_sweep(xb0, xq0, gt, dims, args.M, args.efc, args.ef,
                                 args.k, max(3, args.n_runs // 2),
                                 args.thread_sweep, args.sweep_nq)
        out["thread_sweep"] = {"dataset": args.dataset, "rows": trows}
        write_csv("m2_threadsweep", trows)
        faiss.omp_set_num_threads(nthreads)

    if args.ef_sweep:
        print(f"\nD: efSearch sweep at native d={d0} on {args.dataset} "
              f"(ndis and time, separate passes)")
        erows = run_ef_sweep(xb0, xq0, gt, args.M, args.efc, args.ef_sweep,
                             args.k, args.n_runs)
        el = fit_ndis_elasticity(erows)
        out["ef_sweep"] = {"dataset": args.dataset, "d0": d0,
                           "rows": erows, "elasticity": el}
        for label in ("all", "matched_recall_band"):
            f = el[label]
            if f:
                print(f"  d(log t)/d(log ndis) [{label}] = {f['slope']:.3f} "
                      f"(R^2={f['r2']:.4f}, {f['n_points']} points)")
        print("  ~1.0 => removing a distance evaluation buys its full share of "
              "query time,\n         so a distance-count reduction that does "
              "not show up as speed did not\n         happen at the size "
              "claimed.")
        write_csv("m2_efsweep", erows)

    # Per dataset: the old single filename was overwritten by each dataset in
    # turn, so only the last one survived a multi-dataset run.
    json_path = os.path.join(args.out_dir, f"m2_bandwidth_{args.dataset}.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
