#!/usr/bin/env python3
"""
Verify that every benchmark dataset resolves under a given --data-dir.

Run this on the login node before submitting jobs: it only reads file headers
(and, with --read, a couple of vectors), so it finishes in seconds and does not
need a compute allocation.

Usage:
    python benchs/check_datasets.py --data-dir $WORK/dhm/data
    python benchs/check_datasets.py --data-dir $WORK/dhm/data --read
    python benchs/check_datasets.py --data-dir $WORK/dhm/data sift1m sift10m

Exit code is 0 only if every requested dataset resolved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_datasets import (  # noqa: E402
    ALL_DATASETS,
    DatasetNotFound,
    get_dataset,
    gt_cache_path,
)

# The datasets each paper's benchmark suite needs.
SUITES = {
    "suco": ["sift1m", "sift10m", "deep1m", "deep10m", "gist1m", "spacev10m"],
    "cspg": ["sift1m", "sift10m", "deep1m", "gist1m", "uqv", "openai"],
    "shg":  ["openai", "enron", "gist1m", "msong", "uqv", "msturing10m"],
}

DEFAULT_ORDER = [
    "sift1m", "sift10m", "deep1m", "deep10m", "gist1m", "uqv", "enron",
    "msong", "openai", "msturing10m", "spacev10m",
]


def check(name, data_dir, read_sample=False):
    """Resolve one dataset; return True on success."""
    try:
        ds = get_dataset(name, data_dir)
    except (DatasetNotFound, ValueError) as e:
        print(f"[FAIL] {name}")
        for line in str(e).splitlines():
            print(f"       {line}")
        return False

    print(f"[ ok ] {ds.describe()}")

    warn = []
    if ds.nb > ds.n_available:
        warn.append(
            f"needs {ds.nb:,} base vectors but the file holds {ds.n_available:,}")
    if ds.gt_path is None:
        warn.append("groundtruth will be computed on first run (slow, then cached)")
    if ds.train_path is None:
        warn.append("no learn file; training vectors sampled from the base")
    for w in warn:
        print(f"       WARN: {w}")

    if read_sample:
        try:
            xq = ds.get_queries(nq=min(8, ds.nq))
            xb = ds._read(ds.base_path, ds.base_reader, min(8, ds.nb))
            print(f"       read xb{xb.shape} {xb.dtype} "
                  f"[{xb.min():.4g}, {xb.max():.4g}] | "
                  f"xq{xq.shape} [{xq.min():.4g}, {xq.max():.4g}]")
            if xb.shape[1] != xq.shape[1]:
                print(f"       FAIL: base d={xb.shape[1]} != query d={xq.shape[1]}")
                return False
            if ds.gt_path is not None:
                gt = ds.get_groundtruth(k=None)
                print(f"       read gt{gt.shape} {gt.dtype} "
                      f"max_id={int(gt.max()):,}")
                if gt.shape[0] != ds.nq:
                    print(f"       WARN: groundtruth has {gt.shape[0]:,} rows "
                          f"but there are {ds.nq:,} queries")
                if int(gt.max()) >= ds.nb:
                    print(f"       FAIL: groundtruth references id "
                          f"{int(gt.max()):,} outside a {ds.nb:,}-vector base "
                          f"-- wrong groundtruth file for this crop")
                    return False
        except Exception as e:  # noqa: BLE001 - report anything the readers raise
            print(f"       FAIL while reading: {type(e).__name__}: {e}")
            return False

    return True


def verify_gt(name, data_dir, nq_sample=32):
    """Brute-force a sample of queries and check the shipped groundtruth agrees.

    Groundtruth ids stay within range even when the file belongs to a different
    base ordering, so a range check cannot catch a mismatched pairing -- only
    recomputing can. Loads the full base, so run it on a compute node.
    """
    import faiss
    import numpy as np

    ds = get_dataset(name, data_dir)
    # A computed-and-cached groundtruth is worth verifying too: it pins down
    # that the crop and the query set it was built against still agree.
    gt_src = ds.gt_path or gt_cache_path(name, data_dir)
    if not os.path.exists(gt_src):
        print(f"[skip] {name}: no groundtruth on disk yet "
              f"(it is computed on the first benchmark run)")
        return True

    print(f"[....] {name}: verifying {gt_src}")
    xb = ds.get_database()
    xq = ds.get_queries()[:nq_sample]
    gt = ds.get_groundtruth(k=10)[:nq_sample]

    metric = faiss.METRIC_INNER_PRODUCT if ds.metric == "IP" else faiss.METRIC_L2
    _, ids = faiss.knn(xq, xb, 10, metric=metric)

    top1 = float((ids[:, 0] == gt[:, 0]).mean())
    # ties at equal distance are legitimate, so also allow top-1 in true top-10
    top1_in10 = float(np.mean([g in row for g, row in zip(gt[:, 0], ids)]))

    print(f"       top-1 exact agreement : {top1:.3f}")
    print(f"       shipped top-1 in true top-10: {top1_in10:.3f}")
    if top1_in10 >= 0.95:
        print(f"[ ok ] {name}: groundtruth matches the base")
        return True
    print(f"[FAIL] {name}: {os.path.basename(gt_src)} does NOT match "
          f"{os.path.basename(ds.base_path)} / "
          f"{os.path.basename(ds.query_path)}.")
    print( "       Recall measured against it would be meaningless. The base, "
           "query and")
    print( "       groundtruth files must come from the same release.")
    return False


def main():
    p = argparse.ArgumentParser(
        description="Check benchmark dataset resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("datasets", nargs="*", default=None,
                   help=f"datasets to check (default: all). "
                        f"Known: {', '.join(ALL_DATASETS)}")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/"),
                   help="root directory holding the dataset subdirectories")
    p.add_argument("--suite", choices=sorted(SUITES),
                   help="check only the datasets one paper's suite needs")
    p.add_argument("--read", action="store_true",
                   help="also read a few vectors and validate groundtruth ids")
    p.add_argument("--verify-gt", action="store_true",
                   help="brute-force a sample of queries to confirm the shipped "
                        "groundtruth really matches the base. Loads the full "
                        "base into RAM, so run it on a compute node.")
    p.add_argument("--verify-n", type=int, default=32,
                   help="queries to brute-force per dataset with --verify-gt")
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if args.datasets:
        names = args.datasets
    elif args.suite:
        names = SUITES[args.suite]
    else:
        names = DEFAULT_ORDER

    print(f"data-dir: {data_dir}")
    if not os.path.isdir(data_dir):
        sys.exit(f"ERROR: --data-dir {data_dir} is not a directory")
    print(f"contents: {', '.join(sorted(os.listdir(data_dir))[:20])}")
    print("-" * 72)

    failed = [n for n in names if not check(n, data_dir, args.read)]

    print("-" * 72)
    print(f"{len(names) - len(failed)}/{len(names)} datasets resolved")
    if failed:
        print(f"failed: {', '.join(failed)}")

    if args.verify_gt:
        print("-" * 72)
        print("verifying groundtruth against the base (brute force)")
        for n in names:
            if n in failed:
                continue
            try:
                if not verify_gt(n, data_dir, args.verify_n):
                    failed.append(n)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {n}: verification error: {type(e).__name__}: {e}")
                failed.append(n)
        print("-" * 72)

    if failed:
        print(f"failed: {', '.join(sorted(set(failed)))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
