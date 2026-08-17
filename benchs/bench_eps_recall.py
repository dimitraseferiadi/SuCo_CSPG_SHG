#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""eps-recall: recall that credits a returned point by distance, not by id.

A returned point r is credited when

    d(q, r) <= (1 + eps) * d_k(q),

with d_k(q) the k-th ground-truth distance, so eps-recall@k is the fraction of
the k returned points that are at worst a (1+eps) factor from the k-th true
neighbour.  eps=0 is the strictest form and still differs from id recall
wherever distances tie: a point sitting at *exactly* d_k(q) is credited,
whichever copy of it the index happened to return.

This metric exists because id recall conflates two different outcomes on data
with duplicate or near-duplicate base vectors.  A search that returns the wrong
id at the right distance is indistinguishable, under id recall, from one that
returns a genuinely worse point.  On UQ-V the first case dominates -- 43% of
queries have a base vector at distance exactly 0, and a query's k-th neighbour
is typically one of many points at the same distance -- so id recall reports a
ceiling that is an artefact of tie-breaking rather than a search failure.

Index parameters and search knobs are imported from bench_router_paper, and
indexes are reloaded from --index-dir under that script's own naming
convention, so a run here reuses what the cross-algorithm benchmark already
built and cannot drift from its configuration.

Usage:
    python benchs/bench_eps_recall.py --data-dir DIR --index-dir DIR \\
        --dataset uqv
    python benchs/bench_eps_recall.py --data-dir DIR --index-dir DIR \\
        --dataset uqv --k 1 10 100 --plateau
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss  # noqa: E402

import bench_datasets as bd  # noqa: E402
from bench_router_paper import (  # noqa: E402
    BUILDERS,
    SEARCH_FACTORY,
    SUCO_NSUBSPACES_OVERRIDE,
    build_index_suco,
    resolve_suco_nsubspaces,
)

DEFAULT_EPS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
DEFAULT_INDEX_TYPES = ["suco", "shg", "cspg", "hnsw32", "hnsw48"]

# Search budgets.  A subset of the bench_router_paper sweeps: eps-recall is a
# property of the returned set alone, so the curve only needs enough points to
# show where it saturates, not the full timing grid.
SWEEPS = {
    "efSearch":        [10, 20, 40, 80, 150, 300, 600, 1000, 2000],
    "candidate_ratio": [0.0005, 0.002, 0.005, 0.02, 0.05, 0.1, 0.2],
    "nprobe":          [1, 4, 16, 64, 256, 1024],
}

# Per-query percentiles kept alongside the mean: the mean alone cannot say
# whether the residual loss is spread over all queries or concentrated on a few.
PCTILES = (1, 10, 50)

# float64 working-set target for the distance recomputation, in bytes.
CHUNK_BYTES = 1 << 27


# ---------------------------------------------------------------------------
# Exact distances
#
# Every distance this metric compares is recomputed in float64 from the raw
# vectors rather than taken from the index.  The quantisation-family indexes
# return approximate distances outright, and even the exact-distance ones
# accumulate float32 error of the same order as the gaps the eps rule has to
# resolve -- on a tie plateau that decides the credit.
# ---------------------------------------------------------------------------

def exact_d2(xb, xq, ids):
    """Exact squared L2 from each query to its returned ids; +inf where id < 0."""
    nq, k = ids.shape
    d = xb.shape[1]
    out = np.full((nq, k), np.inf, dtype=np.float64)
    rows = max(1, int(CHUNK_BYTES // (8 * d * max(k, 1))))
    for s in range(0, nq, rows):
        e = min(s + rows, nq)
        blk = ids[s:e]
        valid = blk >= 0
        flat = blk[valid]
        if flat.size == 0:
            continue
        qrep = np.repeat(xq[s:e], k, axis=0).reshape(e - s, k, d)[valid]
        diff = xb[flat].astype(np.float64) - qrep.astype(np.float64)
        out[s:e][valid] = np.einsum("ij,ij->i", diff, diff)
    return out


def gt_distances(xb, xq, gt):
    """Exact squared L2 to every ground-truth id, ascending per query."""
    out = exact_d2(xb, xq, gt)
    out.sort(axis=1)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def eps_recall(ids, xb, xq, gt, gt_d2, k, eps_list):
    """id recall@k plus eps-recall@k for every eps in eps_list."""
    nq = ids.shape[0]
    ids = ids[:, :k]

    hits = np.fromiter(
        (len((set(ids[i].tolist()) - {-1}) & set(gt[i, :k].tolist()))
         for i in range(nq)), dtype=np.float64, count=nq)
    res = {"recall": float(hits.mean() / k)}

    d2 = exact_d2(xb, xq, ids)
    dk = np.sqrt(gt_d2[:, k - 1])
    for e in eps_list:
        # (1+eps) scales a distance, so it squares for the squared metric. The
        # 1+1e-12 slack keeps an exact tie from being lost to rounding in the
        # threshold arithmetic itself.
        thr = ((1.0 + e) * dk) ** 2 * (1.0 + 1e-12)
        per_q = (d2 <= thr[:, None]).sum(axis=1) / k
        res[f"eps_recall_{e}"] = float(per_q.mean())
        res[f"eps_recall_{e}_perfect"] = float((per_q >= 1.0).mean())
        for p in PCTILES:
            res[f"eps_recall_{e}_p{p}"] = float(np.percentile(per_q, p))
    return res


def plateau_counts(xb, xq, gt_d2, k, eps_list, qb=500, bb=250_000):
    """|{x in base : d(q,x) <= (1+eps) d_k(q)}| per query, by brute force.

    The number of base points that are legitimate answers under the eps rule.
    Where this greatly exceeds k, id recall is scoring a choice among ties
    rather than the quality of the points found.  Costs one full pass over the
    base per query block, so it is off by default.
    """
    nb, nq = xb.shape[0], xq.shape[0]
    dk = np.sqrt(gt_d2[:, k - 1])
    thr = {e: ((1.0 + e) * dk) ** 2 * (1.0 + 1e-9) for e in eps_list}
    counts = {e: np.zeros(nq, dtype=np.int64) for e in eps_list}

    xb_sq = np.einsum("ij,ij->i", xb, xb).astype(np.float32)
    for qs in range(0, nq, qb):
        qe = min(qs + qb, nq)
        q = xq[qs:qe]
        q_sq = np.einsum("ij,ij->i", q, q).astype(np.float64)
        # ||q||^2 is folded out of the kernel and into the thresholds.
        thr_adj = {e: (thr[e][qs:qe] - q_sq).astype(np.float32)
                   for e in eps_list}
        thr_max = np.max(np.stack(list(thr_adj.values())), axis=0)
        for bs in range(0, nb, bb):
            be = min(bs + bb, nb)
            D = q @ xb[bs:be].T
            D *= -2.0
            D += xb_sq[bs:be][None, :]
            mask = D <= thr_max[:, None]
            rows, _ = np.nonzero(mask)
            if rows.size == 0:
                continue
            vals = D[mask]
            for e in eps_list:
                hit = vals <= thr_adj[e][rows]
                counts[e][qs:qe] += np.bincount(rows[hit], minlength=qe - qs)
    return counts


# ---------------------------------------------------------------------------
# Index acquisition
# ---------------------------------------------------------------------------

def load_or_build(kind, dataset, xb, d, index_dir):
    """Reload the index bench_router_paper built, or build and save it.

    Path convention is that script's: {index_dir}/{dataset}_{kind}.idx, with
    SuCo tagged by Ns so a run with a different subspace count is not silently
    reloaded.
    """
    label, builder = BUILDERS[kind]
    suco_n_override = None
    if kind == "suco":
        suco_n, _ = resolve_suco_nsubspaces(dataset, d)
        suco_n_override = SUCO_NSUBSPACES_OVERRIDE.get(dataset)
        idx_path = os.path.join(index_dir, f"{dataset}_suco_ns{suco_n}.idx")
    else:
        idx_path = os.path.join(index_dir, f"{dataset}_{kind}.idx")

    if os.path.exists(idx_path):
        print(f"    loading {label} from {idx_path}", flush=True)
        try:
            return label, faiss.read_index(idx_path), -1.0
        except Exception as e:
            print(f"    load failed, rebuilding: {e}", flush=True)

    print(f"    building {label}", flush=True)
    if kind == "suco":
        idx, build_s = build_index_suco(xb, d, n_override=suco_n_override)
    else:
        idx, build_s = builder(xb, d)
    try:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(idx, idx_path)
        print(f"    saved to {idx_path}", flush=True)
    except Exception as e:
        print(f"    could not save: {e}", flush=True)
    return label, idx, build_s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dataset(name, args):
    print(f"\n{'#'*70}\n# Dataset: {name.upper()}\n{'#'*70}", flush=True)
    ds = bd.get_dataset(name, args.data_dir, nb=args.nb,
                        gt_cache_dir=args.index_dir)
    xb = ds.get_database()
    xq = ds.get_queries()
    gt = ds.get_groundtruth(k=max(100, max(args.k)), xb=xb, xq=xq)
    if gt.max() >= xb.shape[0]:
        print("  groundtruth ids exceed the base size — recomputing")
        gt = bd.compute_ground_truth(xb, xq, k=gt.shape[1])
    d = int(xb.shape[1])
    print(f"  nb={xb.shape[0]} d={d} nq={xq.shape[0]} "
          f"threads={faiss.omp_get_max_threads()}", flush=True)

    t0 = time.time()
    gt_d2 = gt_distances(xb, xq, gt)
    print(f"  exact gt distances: {time.time()-t0:.1f}s", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"eps_recall_{name}.json")
    out = {}
    if os.path.exists(path) and not args.force:
        try:
            with open(path) as f:
                out = json.load(f)
        except Exception:
            out = {}
    out.update({"dataset": name, "nb": int(xb.shape[0]), "d": d,
                "nq": int(xq.shape[0]), "eps": args.eps, "ks": args.k})
    out.setdefault("indexes", {})

    def save():
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    if args.plateau:
        out.setdefault("plateau", {})
        for k in args.k:
            if f"k{k}" in out["plateau"] and not args.force:
                continue
            t0 = time.time()
            c = plateau_counts(xb, xq, gt_d2, k, args.eps)
            out["plateau"][f"k{k}"] = {
                str(e): {"mean": float(v.mean()),
                         "median": float(np.median(v)),
                         "p99": float(np.percentile(v, 99)),
                         "max": int(v.max()),
                         "frac_above_k": float((v > k).mean())}
                for e, v in c.items()
            }
            print(f"  plateau k={k} ({time.time()-t0:.1f}s): "
                  + "  ".join(f"eps={e}:{v.mean():.1f}" for e, v in c.items()),
                  flush=True)
            save()

    for kind in args.index_type:
        if kind in out["indexes"] and not args.force:
            print(f"\n  {kind}: already in {os.path.basename(path)}, skipping",
                  flush=True)
            continue
        print(f"\n  --- {kind} ---", flush=True)
        try:
            label, idx, build_s = load_or_build(
                kind, name, xb, d, args.index_dir)
        except Exception as e:
            print(f"    BUILD/LOAD FAILED — {e}", flush=True)
            traceback.print_exc()
            continue

        factory, _, knob = SEARCH_FACTORY[kind]
        entry = {"label": label, "knob": knob, "build_s": build_s,
                 "points": []}
        for k in args.k:
            for p in SWEEPS[knob]:
                search = factory(p)
                t0 = time.time()
                _, ids = search(idx, xq, k)
                dt = time.time() - t0
                m = eps_recall(ids, xb, xq, gt, gt_d2, k, args.eps)
                m.update({knob: p, "k": k,
                          "ms_per_query": dt * 1000.0 / xq.shape[0]})
                entry["points"].append(m)
                shown = [e for e in (0.0, 0.01, 0.1) if e in args.eps]
                print(f"    k={k:<4} {knob}={p:<8} R={m['recall']:.4f}  "
                      + "  ".join(f"eps{e}={m[f'eps_recall_{e}']:.4f}"
                                  for e in shown)
                      + f"  ({m['ms_per_query']:.3f} ms/q)", flush=True)
        out["indexes"][kind] = entry
        save()
        print(f"    -> {path}", flush=True)
        del idx
        gc.collect()

    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/"))
    ap.add_argument("--index-dir", default=os.environ.get("INDEX_DIR",
                                                          "indices/"),
                    help="where bench_router_paper.py's indexes live; they are "
                         "reloaded from here rather than rebuilt")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dataset", nargs="+", default=["uqv"])
    ap.add_argument("--index-type", nargs="+", default=DEFAULT_INDEX_TYPES,
                    choices=sorted(BUILDERS))
    ap.add_argument("--k", nargs="+", type=int, default=[10])
    ap.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    ap.add_argument("--nb", type=int, default=None)
    ap.add_argument("--plateau", action="store_true",
                    help="also count the base points inside each eps radius "
                         "(one brute-force pass over the base; O(nb*nq))")
    ap.add_argument("--force", action="store_true",
                    help="recompute entries already present in the output")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_router")
    args.eps = sorted(set(args.eps))

    for name in args.dataset:
        run_dataset(bd.canonical_name(name), args)


if __name__ == "__main__":
    main()
