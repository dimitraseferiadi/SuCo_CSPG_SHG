"""
Unified dataset resolution and loading for the SuCo / CSPG / SHG benchmarks.

Every benchmark script used to carry its own copy of the readers and its own
hard-coded file names.  The datasets have since been re-downloaded into a
different on-disk layout, so the path knowledge now lives here instead.

Layout currently on the HPC (--data-dir points at the directory holding these):

    sift100M/    bigann_base_100M.bvecs, bigann_query.bvecs, bigann_learn.bvecs,
                 gnd/idx_{1,2,5,10,20,50,100}M.ivecs
    deep10m/     base_00, learn_00   (raw Yandex Deep1B chunks, 27.4M vectors
                 each; see the DEEP note below)
    gist1m/      gist_{base,learn,query}.fvecs, gist_groundtruth.ivecs
    uqv/         uqv_{base,query}.fvecs, uqv_groundtruth.ivecs
    enron/       enron_{base,query}.fvecs, enron_groundtruth.ivecs
    msong/       msong_{base,query}.fvecs, msong_groundtruth.ivecs
    openai1m/    base.fbin, queries.fbin
    msturing10m/ base.10M.fbin, query.public.100K.fbin,
                 groundtruth.public.100K.ibin
    spacev10m/   spacev1b_base.i8bin.crop_nb_10000000, query.i8bin,
                 msspacev-gt-10M

SIFT and DEEP are served from a single large base file cropped to the requested
size: sift1m/sift10m are the first 1M/10M vectors of the BIGANN base, deep1m/
deep10m the first 1M/10M of the DEEP base.  For BIGANN the official groundtruth
for each crop ships in gnd/idx_<N>M.ivecs, so nothing is recomputed.

DEEP exists in three releases whose base orderings differ (big-ann-benchmarks
base.10M.fbin, faiss/Yandex base.fvecs, and the raw Yandex chunks base_00 /
learn_00).  A groundtruth from one release is meaningless against another's
base -- and silently so, since the ids stay in range either way -- so base,
query and groundtruth are only ever taken from the same release.

With only the raw chunks present there is no official query set, so the last
10 000 vectors of the chunk are held out as queries (written once to
deep_heldout_query_10k.fvecs) and excluded from every crop; groundtruth is then
computed and cached per crop.  Recall measured this way is self-consistent but
not comparable to published numbers that use the official deep1B_queries.fvecs.
Drop that file into the directory and it is preferred automatically.

Older layouts (sift1M/sift_base.fvecs, deep1b/base.fvecs, SIFT10Mfeatures.mat,
openai_xb.npy, …) are still accepted as fallbacks, so existing local checkouts
keep working.

Use `python benchs/check_datasets.py --data-dir DIR` to verify resolution
before submitting a job.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Low-level format readers
# ---------------------------------------------------------------------------


def sanitize(x):
    """Contiguous float32 view, as FAISS requires."""
    return np.ascontiguousarray(x, dtype="float32")


def ivecs_mmap(fname):
    """Memory-map a .ivecs file -> (n, d) int32 view (no copy).

    The row count comes from the file size rather than from reshape(-1, ...),
    so a truncated final record is dropped instead of failing the mapping --
    the raw Yandex Deep1B chunks end with a partial vector.
    """
    d = int(np.fromfile(fname, dtype="int32", count=1)[0])
    if d <= 0:
        raise ValueError(f"{fname}: implausible dimension {d} in header")
    n = os.path.getsize(fname) // ((d + 1) * 4)
    a = np.memmap(fname, dtype="int32", mode="r", shape=(n, d + 1))
    return a[:, 1:]


def fvecs_mmap(fname):
    """Memory-map a .fvecs file -> (n, d) float32 view (no copy)."""
    return ivecs_mmap(fname).view("float32")


def bvecs_mmap(fname):
    """Memory-map a .bvecs file -> (n, d) uint8 view (no copy).

    Row count from the file size, so a truncated final record is dropped
    rather than failing the reshape.
    """
    d = int(np.fromfile(fname, dtype="int32", count=1)[0])
    if d <= 0:
        raise ValueError(f"{fname}: implausible dimension {d} in header")
    n = os.path.getsize(fname) // (d + 4)
    x = np.memmap(fname, dtype="uint8", mode="r", shape=(n, d + 4))
    return x[:, 4:]


def read_ivecs(fname, n=None):
    """Read a .ivecs file into RAM as int32."""
    x = ivecs_mmap(fname)
    if n is not None:
        x = x[:n]
    return np.ascontiguousarray(x, dtype="int32")


def read_fvecs(fname, n=None):
    """Read a .fvecs file into RAM as float32."""
    x = fvecs_mmap(fname)
    if n is not None:
        x = x[:n]
    return sanitize(x)


def read_bvecs(fname, n=None, chunk=1_000_000):
    """Read (a prefix of) a .bvecs file into RAM as float32.

    Converted in chunks so a 10M x 128 crop never needs a second full-size
    temporary alongside the result.
    """
    x = bvecs_mmap(fname)
    n = x.shape[0] if n is None else min(n, x.shape[0])
    out = np.empty((n, x.shape[1]), dtype="float32")
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = x[start:end]
    return out


def _bin_header(fname, dtype):
    """Return (n, d) for a .fbin/.i8bin/.u8bin file, corrected for cropping.

    The `crop_nb_*` files keep the original 1B row count in the header, so the
    true row count has to come from the file size.
    """
    with open(fname, "rb") as f:
        hdr_n, d = struct.unpack("ii", f.read(8))
    itemsize = np.dtype(dtype).itemsize
    actual_n = (os.path.getsize(fname) - 8) // (d * itemsize)
    return min(hdr_n, actual_n), d


def read_bin(fname, dtype=np.float32, n=None, as_float32=True):
    """Read a .fbin / .i8bin / .u8bin file: int32 (n, d) header, then n*d values."""
    total_n, d = _bin_header(fname, dtype)
    n = total_n if n is None else min(n, total_n)
    x = np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(total_n, d))[:n]
    return sanitize(x) if as_float32 else np.ascontiguousarray(x)


def read_fbin(fname, dtype=np.float32, n=None):
    return read_bin(fname, dtype=dtype, n=n)


def read_i8bin(fname, n=None):
    return read_bin(fname, dtype=np.int8, n=n)


def read_ibin(fname, n=None):
    """Read groundtruth ids from a .ibin / big-ann-benchmarks GT file.

    Both formats start with an int32 (n, k) header followed by n*k int32 ids.
    The big-ann-benchmarks files (msspacev-gt-10M, msturing-gt-10M) append n*k
    float32 distances after the ids; those trailing bytes are ignored here.
    """
    with open(fname, "rb") as f:
        hdr_n, k = struct.unpack("ii", f.read(8))
    payload = (os.path.getsize(fname) - 8) // 4
    # ids only -> payload == n*k;  ids + distances -> payload == 2*n*k
    hdr_n = min(hdr_n, payload // k)
    ids = np.memmap(fname, dtype="int32", mode="r", offset=8, shape=(hdr_n, k))
    if n is not None:
        ids = ids[:n]
    return np.ascontiguousarray(ids, dtype="int32")


def read_enron_data(fname):
    """Read enron.data_new: 3-int32 header (version, n, d), then n*d float32."""
    with open(fname, "rb") as f:
        _version, n, d = np.fromfile(f, dtype=np.int32, count=3)
        data = np.fromfile(f, dtype=np.float32, count=int(n) * int(d))
    return data.reshape(int(n), int(d))


def read_npy(fname, n=None):
    x = np.load(fname, mmap_mode="r")
    if n is not None:
        x = x[:n]
    return sanitize(x)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class DatasetNotFound(FileNotFoundError):
    """Raised with the full list of locations that were tried."""


def _find_dir(data_dir, names, marker_files=None):
    """Locate a dataset directory under data_dir.

    data_dir itself is accepted too, so both `--data-dir data/` and
    `--data-dir data/spacev10m/` resolve. When marker_files is given, a
    candidate only counts if it holds one of them, which keeps an empty or
    half-populated directory from shadowing a good one.
    """
    for name in list(names) + [os.curdir]:
        p = os.path.normpath(os.path.join(data_dir, name))
        if not os.path.isdir(p):
            continue
        if marker_files and _find_file(p, marker_files) is None:
            continue
        return p
    return None


def _find_file(dirpath, names):
    if dirpath is None:
        return None
    for name in names:
        p = os.path.join(dirpath, name)
        if os.path.exists(p):
            return p
    return None


def _require(path, what, dirpath, names):
    if path is None:
        where = dirpath if dirpath else "<dataset dir not found>"
        raise DatasetNotFound(
            f"{what} not found.\n"
            f"  looked in : {where}\n"
            f"  tried     : {', '.join(names)}"
        )
    return path


def _require_dir(dirpath, what, data_dir, names):
    if dirpath is None:
        raise DatasetNotFound(
            f"{what} directory not found under {data_dir}\n"
            f"  tried: {', '.join(names)}"
        )
    return dirpath


# Directory names tried for each dataset family, new layout first.
_DIRS = {
    "bigann":      ["sift100M", "sift100m", "bigann", "sift1B", "sift1b"],
    "sift1m_old":  ["sift1M", "sift1m", "sift"],
    "deep":        ["deep10m", "deep10M", "deep1b", "deep1B", "deep"],
    "gist":        ["gist1m", "gist1M", "gist"],
    "uqv":         ["uqv", "uqv1m", "UQV"],
    "enron":       ["enron"],
    "msong":       ["msong"],
    "openai":      ["openai1m", "openai"],
    "msturing":    ["msturing10m", "msturing"],
    "spacev":      ["spacev10m", "spacev", "spacev1b"],
}

_BIGANN_BASE = [
    "bigann_base_100M.bvecs", "bigann_base.bvecs", "bigann_base_1B.bvecs",
    "bigann_base_10M.bvecs",
]
_BIGANN_QUERY = ["bigann_query.bvecs", "bigann_queries.bvecs"]
_BIGANN_LEARN = ["bigann_learn.bvecs"]

_DEEP_BASE = [
    "base.10M.fbin", "base.1B.fbin.crop_nb_10000000", "deep10M.fvecs",
    "base.fvecs", "base.100M.fbin", "base_00", "base00",
]
_DEEP_QUERY = [
    "query.public.10K.fbin", "queries.fbin", "deep1B_queries.fvecs",
    "deep10M_queries.fvecs",
]

# When only the raw Yandex chunks are on disk there is no official query set,
# so one is held out from the tail of the base chunk and written here once.
DEEP_HELDOUT_QUERY_FILE = "deep_heldout_query_10k.fvecs"
DEEP_HELDOUT_NQ = 10_000

_OPENAI_BASE = ["base.fbin", "openai_xb.npy"]
_OPENAI_QUERY = ["queries.fbin", "query.fbin", "openai_xq.npy"]

_MSTURING_BASE = ["base.10M.fbin", "base1b.fbin.crop_nb_10000000"]
_MSTURING_QUERY = ["query.public.100K.fbin", "testQuery10K.fbin", "query.fbin"]
_MSTURING_GT = ["groundtruth.public.100K.ibin", "msturing-gt-10M"]

_SPACEV_BASE = [
    "spacev1b_base.i8bin.crop_nb_10000000", "base.10M.i8bin", "base.100M.i8bin",
]
_SPACEV_QUERY = ["query.i8bin", "query.30K.i8bin", "private_query_30k.bin"]
_SPACEV_GT = ["msspacev-gt-10M", "gt100_private_query_30k.bin"]

# BIGANN ships official groundtruth only for these crop sizes.
_BIGANN_GT_SIZES = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


# ---------------------------------------------------------------------------
# Groundtruth
# ---------------------------------------------------------------------------


def compute_ground_truth(xb, xq, k=100, batch_size=4096, metric="L2"):
    """Exact k-NN groundtruth by brute force, batched over queries."""
    import faiss

    faiss_metric = (faiss.METRIC_INNER_PRODUCT if metric == "IP"
                    else faiss.METRIC_L2)
    nq = xq.shape[0]
    print(f"  computing groundtruth (nb={xb.shape[0]:,} nq={nq:,} k={k} "
          f"metric={metric}) ...", flush=True)
    gt = np.empty((nq, k), dtype="int32")
    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        _, ids = faiss.knn(sanitize(xq[start:end]), xb, k, metric=faiss_metric)
        gt[start:end] = ids
    return gt


# Groundtruth is always cached at this many neighbours and sliced down to the
# k a caller asks for, so a k=10 request reuses the k=100 file rather than
# paying for a second brute-force pass over the base.
GT_CACHE_K = 100


def _cache_k(k):
    return max(k or GT_CACHE_K, GT_CACHE_K)


def _fingerprint(base_path, query_path):
    """Short digest of the (base, query) pair a groundtruth was built from.

    Swapping in a different query set -- e.g. dropping in the official
    deep1B_queries.fvecs next to a held-out slice -- must not silently reuse a
    groundtruth computed against the old one, and ids alone would not reveal
    the mismatch. Query files are small enough to hash whole; the base is
    identified by name and size, since hashing 10 GB on every call is not.
    """
    h = hashlib.blake2b(digest_size=6)
    for path in (base_path, query_path):
        h.update(os.path.basename(path).encode())
        h.update(str(os.path.getsize(path)).encode())
    with open(query_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gt_cache_path(dirpath, tag, k=None, cache_dir=None, fingerprint=None):
    suffix = f"_{fingerprint}" if fingerprint else ""
    return os.path.join(cache_dir or dirpath,
                        f"gt_{tag}_k{_cache_k(k)}{suffix}.npy")


def _load_or_compute_gt(dirpath, tag, xb, xq, k, cache_dir=None,
                        fingerprint=None, metric="L2"):
    """Return cached groundtruth, else compute it and try to cache."""
    path = _gt_cache_path(dirpath, tag, k, cache_dir, fingerprint)
    if os.path.exists(path):
        print(f"  groundtruth from cache: {path}")
        gt = np.load(path).astype("int32", copy=False)
    else:
        gt = compute_ground_truth(xb, xq, k=_cache_k(k), metric=metric)
        try:
            np.save(path, gt)
            print(f"  groundtruth cached to {path}")
        except OSError as e:
            print(f"  could not cache groundtruth ({e}); "
                  f"it will be recomputed next run")
    return gt[:, :k] if k else gt


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# name -> (family, nb).  nb=None means "whole file".
_SIZED = {
    "sift1m":   ("sift", 1_000_000),
    "sift2m":   ("sift", 2_000_000),
    "sift5m":   ("sift", 5_000_000),
    "sift10m":  ("sift", 10_000_000),
    "sift20m":  ("sift", 20_000_000),
    "sift50m":  ("sift", 50_000_000),
    "sift100m": ("sift", 100_000_000),
    "sift0.1m": ("sift", 100_000),
    "sift0.2m": ("sift", 200_000),
    "sift0.5m": ("sift", 500_000),
    "deep1m":   ("deep", 1_000_000),
    "deep2m":   ("deep", 2_000_000),
    "deep5m":   ("deep", 5_000_000),
    "deep10m":  ("deep", 10_000_000),
}

_ALIASES = {
    "sift": "sift1m",
    # bench_router_paper's bigann-scaling family is the same BIGANN crops.
    "bigann100k": "sift0.1m",
    "bigann200k": "sift0.2m",
    "bigann500k": "sift0.5m",
    "bigann1m": "sift1m",
    "bigann2m": "sift2m",
    "bigann5m": "sift5m",
    "bigann10m": "sift10m",
    "bigann20m": "sift20m",
    "bigann50m": "sift50m",
    "bigann100m": "sift100m",
    "deep": "deep1m",
    "deep1b": "deep1m",
    "gist": "gist1m",
    "uqv1m": "uqv",
    "openai1m": "openai",
    "msturing": "msturing10m",
    "spacev": "spacev10m",
    "spacev10M": "spacev10m",
}

ALL_DATASETS = sorted(
    set(_SIZED) | {"gist1m", "uqv", "enron", "msong", "openai", "msturing10m",
                   "spacev10m"}
)


def canonical_name(name):
    name = name.lower()
    return _ALIASES.get(name, name)


class BenchDataset:
    """Lazily-loaded dataset with the faiss.contrib.datasets.Dataset interface.

    Attributes d / nb / nq / metric are filled in as soon as the dataset is
    resolved; vectors are only read when the get_* methods are called.
    """

    def __init__(self, name, data_dir, nb=None, gt_cache_dir=None):
        self.name = canonical_name(name)
        self.data_dir = os.path.expanduser(data_dir.rstrip("/") or ".")
        self.gt_cache_dir = gt_cache_dir
        self.metric = "L2"
        self._resolve(nb)

    # -- resolution ------------------------------------------------------

    def _resolve(self, nb_override):
        name = self.name
        if name in _SIZED:
            family, nb = _SIZED[name]
            self.nb = nb_override or nb
            getattr(self, f"_resolve_{family}")()
        elif name in ("gist1m", "uqv", "enron", "msong"):
            self._resolve_fvecs_triplet(name)
            self.nb = nb_override or self.nb
        elif name == "openai":
            self._resolve_openai()
        elif name == "msturing10m":
            self._resolve_msturing()
        elif name == "spacev10m":
            self._resolve_spacev()
        else:
            raise ValueError(
                f"Unknown dataset {name!r}. Known: {', '.join(ALL_DATASETS)}"
            )
        if nb_override:
            self.nb = min(self.nb, nb_override)

    def _resolve_sift(self):
        """SIFT/BIGANN: crop of the shared BIGANN base, official GT per crop."""
        d = _find_dir(self.data_dir, _DIRS["bigann"], _BIGANN_BASE)
        base = _find_file(d, _BIGANN_BASE)
        if base is not None:
            self.dir = d
            self.base_path = base
            self.base_reader = "bvecs"
            self.query_path = _require(
                _find_file(d, _BIGANN_QUERY), "BIGANN queries", d, _BIGANN_QUERY)
            self.query_reader = "bvecs"
            self.train_path = _find_file(d, _BIGANN_LEARN)
            self.train_reader = "bvecs"
            nb_m = self.nb // 1_000_000
            gt = None
            if nb_m in _BIGANN_GT_SIZES and nb_m * 1_000_000 == self.nb:
                gt = _find_file(d, [os.path.join("gnd", f"idx_{nb_m}M.ivecs")])
            if gt is None:
                # Crops below 1M have no published GT; bench_router_paper has
                # been caching them here, so reuse those rather than recompute.
                gt = _find_file(d, [
                    os.path.join("gnd", f"computed_gt_{self.nb}_k100.ivecs.npy"),
                ])
            self.gt_path = gt
            self.gt_reader = "npy" if (gt or "").endswith(".npy") else "ivecs"
            self.d = 128
            self.nq = bvecs_mmap(self.query_path).shape[0]
            self.n_available = bvecs_mmap(self.base_path).shape[0]
            return

        # Fallback: classic ANN_SIFT1M release.
        d = _find_dir(self.data_dir, _DIRS["sift1m_old"], ["sift_base.fvecs"])
        base = _find_file(d, ["sift_base.fvecs"])
        if base is None:
            raise DatasetNotFound(
                f"SIFT base not found under {self.data_dir}\n"
                f"  tried {'/, '.join(_DIRS['bigann'])}/ with "
                f"{', '.join(_BIGANN_BASE)}\n"
                f"  and   {'/, '.join(_DIRS['sift1m_old'])}/ with sift_base.fvecs"
            )
        if self.nb > 1_000_000:
            raise DatasetNotFound(
                f"{self.name} needs {self.nb:,} base vectors but only the 1M "
                f"ANN_SIFT1M release was found at {base}.\n"
                f"  Provide the BIGANN base (bigann_base_100M.bvecs) instead."
            )
        self.dir = d
        self.base_path, self.base_reader = base, "fvecs"
        self.query_path = _require(
            _find_file(d, ["sift_query.fvecs"]), "SIFT queries", d,
            ["sift_query.fvecs"])
        self.query_reader = "fvecs"
        self.train_path = _find_file(d, ["sift_learn.fvecs"])
        self.train_reader = "fvecs"
        self.gt_path = _find_file(d, ["sift_groundtruth.ivecs"])
        self.gt_reader = "ivecs"
        self.d = 128
        self.nq = fvecs_mmap(self.query_path).shape[0]
        self.n_available = fvecs_mmap(self.base_path).shape[0]

    def _resolve_deep(self):
        d = _require_dir(_find_dir(self.data_dir, _DIRS["deep"], _DEEP_BASE), "DEEP",
                         self.data_dir, _DIRS["deep"])
        self.dir = d
        # DEEP ships in two incompatible flavours. Their base orderings differ,
        # so a groundtruth from one is meaningless against the other's base --
        # and silently so, since the ids are in range either way. Pick the base
        # first, then take only the query/groundtruth files of that flavour.
        nb_m = self.nb // 1_000_000
        flavours = [
            # big-ann-benchmarks: base.10M.fbin + query.public.10K.fbin
            {"base":  ["base.10M.fbin", "base.1B.fbin.crop_nb_10000000",
                       "base.100M.fbin"],
             "query": ["query.public.10K.fbin", "queries.fbin"],
             "gt":    ["groundtruth.public.10K.ibin", f"deep{nb_m}M_gt.ibin"],
             "reader": "fbin",
             "name":  "big-ann-benchmarks"},
            # faiss/Yandex: base.fvecs + deep1B_queries.fvecs
            {"base":  ["deep10M.fvecs", "base.fvecs"],
             "query": ["deep1B_queries.fvecs", "deep10M_queries.fvecs"],
             "gt":    [f"deep{nb_m}M_groundtruth.ivecs",
                       f"deep{nb_m}M_groundtruth.npy"],
             "reader": "fvecs",
             "name":  "faiss/Yandex Deep1B"},
            # Raw Yandex Disk chunks: base_00 + learn_00, nothing else. The
            # official query set is not part of them, so one is held out from
            # the tail of the chunk (see _derive_deep_queries).
            # The raw chunks are fvecs despite carrying no extension.
            {"base":  ["base_00", "base00"],
             # Official query set wins; the held-out slice is only a fallback
             # for when it is absent.
             "query": ["deep1B_queries.fvecs", DEEP_HELDOUT_QUERY_FILE],
             "gt":    [f"deep{nb_m}M_groundtruth.ivecs",
                       f"deep{nb_m}M_groundtruth.npy"],
             "reader": "fvecs",
             "name":  "raw Deep1B chunks"},
        ]

        for f in flavours:
            base = _find_file(d, f["base"])
            if base is not None:
                break
        else:
            raise DatasetNotFound(
                f"DEEP base not found in {d}\n"
                f"  tried: {', '.join(_DEEP_BASE)}"
            )

        self.flavour = f["name"]
        self.base_path = base
        self.base_reader = f["reader"]
        self.derived_queries = False

        query = _find_file(d, f["query"])
        if query is None and self.flavour == "raw Deep1B chunks":
            # Materialised on first use by _derive_deep_queries().
            query = os.path.join(d, DEEP_HELDOUT_QUERY_FILE)
        # Flagged by which file this is, not by whether it exists yet: once
        # materialised it resolves like any other query file, but the base
        # crop must still stop short of the slice it was taken from.
        self.derived_queries = (
            os.path.basename(query) == DEEP_HELDOUT_QUERY_FILE)
        if query is None:
            raise DatasetNotFound(
                f"DEEP query vectors not found in {d}\n"
                f"  base is {os.path.basename(base)} ({f['name']} flavour), which\n"
                f"  pairs only with: {', '.join(f['query'])}\n"
                f"  Query sets are not interchangeable between flavours -- the\n"
                f"  groundtruth is indexed by position in its own query file."
            )
        self.query_path = query
        self.query_reader = "fbin" if query.endswith(".fbin") else "fvecs"

        self.gt_path = _find_file(d, f["gt"])
        self.gt_reader = (
            "ivecs" if (self.gt_path or "").endswith(".ivecs")
            else "npy" if (self.gt_path or "").endswith(".npy")
            else "ibin"
        )
        self.train_path = _find_file(
            d, ["learn.fvecs", "learn.10M.fbin", "learn_00", "learn00"])
        self.train_reader = (
            "fbin" if (self.train_path or "").endswith(".fbin") else "fvecs"
        )
        self.d = 96
        if self.base_reader == "fvecs":
            self.n_available = fvecs_mmap(base).shape[0]
        else:
            self.n_available, self.d = _bin_header(base, np.float32)

        if self.derived_queries and not os.path.exists(self.query_path):
            self.nq = DEEP_HELDOUT_NQ
        else:
            self.nq = self._probe_n(self.query_path, self.query_reader)

        if self.derived_queries:
            # The held-out slice sits at the tail of the same chunk, so the base
            # crop must stop short of it or queries would leak into the index.
            self.n_available -= DEEP_HELDOUT_NQ

    def _resolve_fvecs_triplet(self, name):
        """gist1m / uqv / enron / msong: <prefix>_{base,query,groundtruth}."""
        prefix = {"gist1m": "gist", "uqv": "uqv", "enron": "enron",
                  "msong": "msong"}[name]
        key = {"gist1m": "gist"}.get(name, name)
        base_names = [f"{prefix}_base.fvecs"]
        if name == "enron":
            base_names.append("enron.data_new")
        d = _require_dir(_find_dir(self.data_dir, _DIRS[key], base_names),
                         name.upper(), self.data_dir, _DIRS[key])
        self.dir = d
        self.base_path = _require(
            _find_file(d, base_names), f"{name} base", d, base_names)
        self.base_reader = (
            "enron" if self.base_path.endswith(".data_new") else "fvecs")
        self.query_path = _require(
            _find_file(d, [f"{prefix}_query.fvecs"]), f"{name} queries", d,
            [f"{prefix}_query.fvecs"])
        self.query_reader = "fvecs"
        self.train_path = _find_file(d, [f"{prefix}_learn.fvecs"])
        self.train_reader = "fvecs"
        self.gt_path = _find_file(d, [f"{prefix}_groundtruth.ivecs"])
        self.gt_reader = "ivecs"
        self.nq = fvecs_mmap(self.query_path).shape[0]
        self.d = fvecs_mmap(self.query_path).shape[1]
        self.n_available = (
            read_enron_data(self.base_path).shape[0]
            if self.base_reader == "enron"
            else fvecs_mmap(self.base_path).shape[0]
        )
        self.nb = self.n_available

    def _resolve_openai(self):
        d = _require_dir(_find_dir(self.data_dir, _DIRS["openai"], _OPENAI_BASE), "OpenAI",
                         self.data_dir, _DIRS["openai"])
        self.dir = d
        self.base_path = _require(
            _find_file(d, _OPENAI_BASE), "OpenAI base", d, _OPENAI_BASE)
        self.base_reader = "npy" if self.base_path.endswith(".npy") else "fbin"
        self.query_path = _require(
            _find_file(d, _OPENAI_QUERY), "OpenAI queries", d, _OPENAI_QUERY)
        self.query_reader = "npy" if self.query_path.endswith(".npy") else "fbin"
        self.train_path = None
        self.train_reader = None
        self.gt_path = _find_file(d, ["openai_gt100.npy", "gt_openai_k100.npy"])
        self.gt_reader = "npy"
        self.n_available, self.d = (
            np.load(self.base_path, mmap_mode="r").shape
            if self.base_reader == "npy" else _bin_header(self.base_path, np.float32)
        )
        self.nb = self.n_available
        self.nq = self._probe_n(self.query_path, self.query_reader)
        self.metric = "IP"

    def _resolve_msturing(self):
        d = _require_dir(_find_dir(self.data_dir, _DIRS["msturing"], _MSTURING_BASE), "MSTuring",
                         self.data_dir, _DIRS["msturing"])
        self.dir = d
        self.base_path = _require(
            _find_file(d, _MSTURING_BASE), "MSTuring base", d, _MSTURING_BASE)
        self.base_reader = "fbin"
        self.query_path = _require(
            _find_file(d, _MSTURING_QUERY), "MSTuring queries", d, _MSTURING_QUERY)
        self.query_reader = "fbin"
        self.train_path = None
        self.train_reader = None
        self.gt_path = _find_file(d, _MSTURING_GT)
        self.gt_reader = "ibin"
        self.n_available, self.d = _bin_header(self.base_path, np.float32)
        self.nb = self.n_available
        self.nq = self._probe_n(self.query_path, "fbin")

    def _resolve_spacev(self):
        d = _require_dir(_find_dir(self.data_dir, _DIRS["spacev"], _SPACEV_BASE), "SpaceV",
                         self.data_dir, _DIRS["spacev"])
        self.dir = d
        self.base_path = _require(
            _find_file(d, _SPACEV_BASE), "SpaceV base", d, _SPACEV_BASE)
        self.base_reader = "i8bin"
        self.query_path = _require(
            _find_file(d, _SPACEV_QUERY), "SpaceV queries", d, _SPACEV_QUERY)
        self.query_reader = "i8bin"
        self.train_path = None
        self.train_reader = None
        # gt100_private_query_30k.bin only matches the private query set.
        gt_names = (
            ["gt100_private_query_30k.bin"]
            if os.path.basename(self.query_path) == "private_query_30k.bin"
            else ["msspacev-gt-10M"]
        )
        self.gt_path = _find_file(d, gt_names)
        self.gt_reader = "ibin"
        self.n_available, self.d = _bin_header(self.base_path, np.int8)
        self.nb = self.n_available
        self.nq = _bin_header(self.query_path, np.int8)[0]

    @staticmethod
    def _probe_n(path, reader):
        if reader == "fvecs":
            return fvecs_mmap(path).shape[0]
        if reader == "bvecs":
            return bvecs_mmap(path).shape[0]
        if reader == "npy":
            return np.load(path, mmap_mode="r").shape[0]
        if reader == "i8bin":
            return _bin_header(path, np.int8)[0]
        return _bin_header(path, np.float32)[0]

    # -- loading ---------------------------------------------------------

    @staticmethod
    def _read(path, reader, n=None):
        if reader == "fvecs":
            return read_fvecs(path, n)
        if reader == "bvecs":
            return read_bvecs(path, n)
        if reader == "fbin":
            return read_fbin(path, n=n)
        if reader == "i8bin":
            return read_i8bin(path, n=n)
        if reader == "npy":
            return read_npy(path, n)
        if reader == "enron":
            x = read_enron_data(path)
            return sanitize(x[:n] if n else x)
        raise ValueError(f"Unknown reader {reader!r}")

    def _check_size(self):
        """Refuse to silently run on a short base file.

        Cropping to fewer vectors than the name implies would also invalidate
        the groundtruth, whose ids are relative to the full crop.
        """
        if self.nb > self.n_available:
            extra = ""
            if getattr(self, "derived_queries", False):
                extra = (f"\n  ({DEEP_HELDOUT_NQ:,} vectors at the tail are "
                         f"reserved as the held-out query set.)")
            raise DatasetNotFound(
                f"{self.name} needs {self.nb:,} base vectors but "
                f"{self.base_path} holds only {self.n_available:,} usable.{extra}\n"
                f"  The groundtruth is indexed against the full crop, so a "
                f"shorter base cannot be used."
            )

    def get_database(self):
        self._check_size()
        return self._read(self.base_path, self.base_reader, self.nb)

    def database_mmap(self):
        """Read-only view of the base crop, for when it must not enter RAM."""
        self._check_size()
        if self.base_reader == "fvecs":
            return fvecs_mmap(self.base_path)[: self.nb]
        if self.base_reader == "bvecs":
            return bvecs_mmap(self.base_path)[: self.nb]
        dtype = np.int8 if self.base_reader == "i8bin" else np.float32
        total_n, d = _bin_header(self.base_path, dtype)
        return np.memmap(self.base_path, dtype=dtype, mode="r", offset=8,
                         shape=(total_n, d))[: self.nb]

    def _derive_deep_queries(self):
        """Hold out a query set from the tail of the raw Deep1B base chunk.

        The chunk holds ~27.4M vectors, so the last 10 000 sit far beyond any
        1M/10M crop and are never indexed. Written once as a real .fvecs file
        so the split is reproducible and inspectable rather than implicit.
        """
        src = fvecs_mmap(self.base_path)
        start = src.shape[0] - DEEP_HELDOUT_NQ
        xq = np.ascontiguousarray(src[start:], dtype="float32")

        n, d = xq.shape
        rec = np.empty((n, d + 1), dtype="int32")
        rec[:, 0] = d
        rec[:, 1:] = xq.view("int32")
        tmp = self.query_path + ".tmp"
        rec.tofile(tmp)
        os.replace(tmp, self.query_path)   # atomic: no half-written query file
        print(f"  held out queries [{start:,}:{start + n:,}] from "
              f"{os.path.basename(self.base_path)} -> {self.query_path}")
        return xq

    def get_queries(self, nq=None):
        if getattr(self, "derived_queries", False) \
                and not os.path.exists(self.query_path):
            xq = self._derive_deep_queries()
            return xq[:nq] if nq else xq
        return self._read(self.query_path, self.query_reader, nq)

    def get_train(self, maxtrain=None, seed=42):
        """Training vectors: the learn file when present, else a base sample."""
        if self.train_path is not None:
            return self._read(self.train_path, self.train_reader, maxtrain)
        maxtrain = min(maxtrain or 500_000, self.nb)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.nb, maxtrain, replace=False))
        xb = self.database_mmap()
        return sanitize(xb[idx])

    def get_groundtruth(self, k=100, xb=None, xq=None):
        """Official groundtruth when available, otherwise computed and cached."""
        if self.gt_path is not None:
            if self.gt_reader == "ivecs":
                gt = read_ivecs(self.gt_path)
            elif self.gt_reader == "npy":
                gt = np.load(self.gt_path).astype("int32", copy=False)
            else:
                gt = read_ibin(self.gt_path)
            if gt.shape[0] > self.nq:
                gt = gt[: self.nq]
            if k is None or gt.shape[1] >= k:
                if k is not None:
                    gt = gt[:, :k]
                return np.ascontiguousarray(gt)
            # Some releases ship a shallow groundtruth -- enron and msong carry
            # only 20 neighbours per query. Truncating the request to what the
            # file holds would silently redefine recall@k, so compute the deeper
            # groundtruth instead and cache it alongside.
            print(f"  {os.path.basename(self.gt_path)} has only {gt.shape[1]} "
                  f"neighbours per query but k={k} was requested; computing a "
                  f"deeper groundtruth")
            shallow = gt
        else:
            shallow = None

        if xb is None:
            xb = self.get_database()
        if xq is None:
            xq = self.get_queries()
        deep = _load_or_compute_gt(self.dir, self.name, xb, xq, k,
                                   self.gt_cache_dir, self.gt_fingerprint(),
                                   metric=self.metric)

        if shallow is not None:
            # The shipped file is shallow but should still be correct as far as
            # it goes; disagreement means it belongs to a different base or
            # query set, which is worth knowing before it reaches a recall plot.
            agree = float((deep[:, 0] == shallow[:, 0]).mean())
            if agree < 0.95:
                print(f"  WARNING: the computed groundtruth agrees with "
                      f"{os.path.basename(self.gt_path)} on only {agree:.1%} of "
                      f"top-1 neighbours. That file may not match this base or "
                      f"query set; the computed one is being used.")
            else:
                print(f"  (computed groundtruth agrees with the shipped "
                      f"{shallow.shape[1]}-NN file on {agree:.1%} of top-1)")
        return deep

    def gt_fingerprint(self):
        """Identity of the (base, query) pair, for naming computed groundtruth."""
        if getattr(self, "_fp", None) is None:
            self._fp = _fingerprint(self.base_path, self.query_path)
        return self._fp

    def describe(self):
        query_note = " [held out from base]" if getattr(
            self, "derived_queries", False) else ""
        flavour = f" ({self.flavour})" if getattr(self, "flavour", None) else ""
        return (
            f"{self.name}: d={self.d} nb={self.nb:,} nq={self.nq:,} "
            f"metric={self.metric}{flavour}\n"
            f"    base  : {self.base_path} ({self.base_reader}, "
            f"{self.n_available:,} usable)\n"
            f"    query : {self.query_path} ({self.query_reader}){query_note}\n"
            f"    train : {self.train_path or '<sampled from base>'}\n"
            f"    gt    : {self.gt_path or '<computed and cached>'}"
        )


def get_dataset(name, data_dir, nb=None, gt_cache_dir=None):
    """Resolve a dataset without reading any vectors."""
    return BenchDataset(name, data_dir, nb=nb, gt_cache_dir=gt_cache_dir)


def load_dataset(name, data_dir, nb=None, k=100, gt_cache_dir=None):
    """Resolve and load a dataset. Returns (xb, xq, gt)."""
    ds = get_dataset(name, data_dir, nb=nb, gt_cache_dir=gt_cache_dir)
    xb = ds.get_database()
    xq = ds.get_queries()
    gt = ds.get_groundtruth(k=k, xb=xb, xq=xq)
    return xb, xq, gt


def load_dataset_queries_only(name, data_dir, nb=None, k=100, gt_cache_dir=None):
    """Load queries and groundtruth only. Fails if the groundtruth needs computing."""
    ds = get_dataset(name, data_dir, nb=nb, gt_cache_dir=gt_cache_dir)
    if ds.gt_path is None:
        cached = _gt_cache_path(ds.dir, ds.name, k, gt_cache_dir)
        if not os.path.exists(cached):
            raise DatasetNotFound(
                f"{name}: no groundtruth on disk and base vectors were not "
                f"loaded, so it cannot be computed. Run once without "
                f"--queries-only to build {cached}."
            )
    return ds.get_queries(), ds.get_groundtruth(k=k)


def gt_cache_path(name, data_dir, k=100, gt_cache_dir=None):
    """Path the groundtruth for this dataset lives at, official or cached."""
    ds = get_dataset(name, data_dir)
    if ds.gt_path is not None:
        return ds.gt_path
    return _gt_cache_path(ds.dir, ds.name, k, gt_cache_dir,
                          ds.gt_fingerprint())


def groundtruth_available(name, data_dir, k=100, gt_cache_dir=None):
    """True if groundtruth can be had without loading the base vectors."""
    try:
        return os.path.exists(gt_cache_path(name, data_dir, k, gt_cache_dir))
    except DatasetNotFound:
        return False


def dataset_size(name, data_dir=None):
    """Base-vector count for a dataset, without touching the disk when known."""
    name = canonical_name(name)
    if name in _SIZED:
        return _SIZED[name][1]
    if data_dir is None:
        raise ValueError(f"dataset_size({name!r}) needs data_dir")
    return get_dataset(name, data_dir).nb


if __name__ == "__main__":
    sys.exit("Use benchs/check_datasets.py to verify dataset resolution.")
