#!/usr/bin/env python3
"""
benchs/check_suco_load.py

Pre-flight for the SuCo deserialisation crash that killed the first M2 run.
Seconds to run, needs no dataset and no batch allocation; exits non-zero when
the run about to be submitted would hit the same wall.

What actually failed, and why it is not a Python bug
----------------------------------------------------
The 2026-08-13 job died at bench_m2_distcount.py's first faiss.read_index with

    IndexSuCo: d must be divisible by nsubspaces

on sift1m, whose d=128 is divisible by its nsubspaces=8.  The dataset had
nothing to do with it.  faiss/impl/index_read.cpp deserialised the IxSC branch
through

    auto idxs = std::make_unique<IndexSuCo>(1);   // placeholder, overwritten

and that placeholder ran the validating constructor with d=1 and the default
nsubspaces=8.  1 % 8 != 0, so the throw fired before a single byte of the index
was read.  Every SuCo index failed to load, on every dataset, always.

Commit 17f60df4 fixed it in C++, not in Python: it added an unvalidated
IndexSuCo::IndexSuCo() and pointed the reader at it.  So a `git pull` alone
changes nothing -- the installed swigfaiss*.so still carries the old compiled
constructor.  FAISS has to be rebuilt and reinstalled into the environment the
job actually imports.  That is what this script checks.

Checks
  1  provenance   which faiss is imported, and whether it postdates the source
  2  constructor  is IndexSuCo() (no arguments) exposed?  the discriminator:
                  present => the fix is compiled in; absent => it is not
  3  round trip   build a tiny SuCo index and read it back through the exact
                  IxSC branch that failed, in memory, no dataset, no disk
  4  real files   optionally faiss.read_index every prebuilt index the run will
                  touch, smallest first, under a size cap

Usage
  python benchs/check_suco_load.py
  python benchs/check_suco_load.py --index-dir $WORK/dhm/indices --datasets sift1m gist1m
  python benchs/check_suco_load.py --skip-files      # checks 1-3 only, ~2 s
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss at all -- nothing else here can be checked.")

OK = "PASS"
NO = "FAIL"


def check_provenance():
    """Which library is imported, and is it newer than the source that fixes it.

    Reported rather than judged: on a shared cluster the .so often lives in
    someone else's conda prefix, where mtimes say little.  Check 2 is the one
    that decides.
    """
    print("1. provenance")
    print(f"   faiss package : {faiss.__file__}")
    so = None
    pkg_dir = os.path.dirname(faiss.__file__ or "")
    for pat in ("_swigfaiss*.so", "*swigfaiss*.so"):
        hits = glob.glob(os.path.join(pkg_dir, pat))
        if hits:
            so = max(hits, key=os.path.getmtime)
            break
    if so:
        print(f"   compiled lib  : {so}")
        print(f"   built         : {time.strftime('%F %T', time.localtime(os.path.getmtime(so)))}")
    else:
        print("   compiled lib  : not found next to the package")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "faiss", "IndexSuCo.cpp")
    if os.path.exists(src) and so:
        s_t, o_t = os.path.getmtime(src), os.path.getmtime(so)
        print(f"   IndexSuCo.cpp : {time.strftime('%F %T', time.localtime(s_t))}")
        if s_t > o_t:
            print("   -> source is NEWER than the compiled library: likely stale build")
        else:
            print("   -> compiled library postdates the source")
    print()


def check_constructor():
    """Advisory only.  Do NOT read a failure here as the bug being present.

    It is tempting to treat a missing zero-argument IndexSuCo() as proof that
    the pre-17f60df4 library is installed, and that is wrong.  read_index is
    pure C++: the IxSC branch of faiss/impl/index_read.cpp constructs the
    placeholder itself and never crosses the Python boundary.  Whether SWIG
    exposes a zero-argument form depends only on when the wrapper was last
    regenerated from IndexSuCo.h.

    Those two move independently, and the combination that looks alarming is
    common: libfaiss rebuilt with the fix, swigfaiss wrapper not regenerated,
    so the C++ load path is correct while Python still demands `d`.  Check 3
    is the one that decides, because it runs the actual path.
    """
    print("2. SWIG binding freshness (advisory -- does not decide anything)")
    if not hasattr(faiss, "IndexSuCo"):
        print(f"   {NO}  this build has no IndexSuCo at all -- wrong FAISS")
        return False
    try:
        idx = faiss.IndexSuCo()
        print(f"   {OK}  IndexSuCo() constructs, d={idx.d}: the wrapper was "
              f"regenerated after 17f60df4")
    except TypeError:
        print("   note  IndexSuCo() needs `d`, so the SWIG wrapper predates")
        print("         17f60df4. This does NOT mean the load is broken --")
        print("         read_index never uses the Python constructor. Check 3")
        print("         settles it. Regenerate the wrapper only if you want to")
        print("         call IndexSuCo() from Python yourself.")
    except Exception as e:
        print(f"   note  IndexSuCo() raised {type(e).__name__}: {e}")
    return True


def check_round_trip():
    """The end-to-end test: through the IxSC reader branch that failed.

    Small enough to run anywhere, and it exercises write_index's dynamic_cast
    dispatch and read_index's fourcc branch rather than a stand-in for them.
    """
    print("\n3. IxSC round trip through read_index (AUTHORITATIVE)")
    d, ns, n = 32, 8, 2000          # 32/8 = 4 per subspace, halves of 2: legal
    try:
        rng = np.random.default_rng(0)
        xb = rng.random((n, d), dtype=np.float32)
        idx = faiss.IndexSuCo(d, ns, 4, 0.05, 0.005, 5)
        idx.train(xb)
        idx.add(xb)
        blob = faiss.serialize_index(idx)
    except Exception as e:
        print(f"   SKIP  could not build/serialise a probe index: "
              f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:120]}")
        return None
    try:
        back = faiss.deserialize_index(blob)
    except Exception as e:
        msg = str(e).strip().splitlines()[-1][:140]
        print(f"   {NO}  deserialise raised: {msg}")
        if "divisible by nsubspaces" in str(e):
            print("        => exactly the original failure, still present.")
        return False
    good = (back.d == d and back.ntotal == n)
    print(f"   {OK if good else NO}  read back d={back.d} ntotal={back.ntotal} "
          f"(expected d={d} ntotal={n})")
    return bool(good)


def check_real_files(index_dir, datasets, max_mb, kinds):
    """Load the prebuilt indexes the run will actually open.

    Smallest first, so a size cap or an out-of-memory login node still leaves a
    verdict for the cheap cases.  Every failure is caught and tabulated: the
    point is a complete picture, not the first traceback.
    """
    print(f"\n4. prebuilt indexes in {index_dir}")
    if not os.path.isdir(index_dir):
        print(f"   SKIP  no such directory")
        return None

    files = []
    for ds in datasets:
        for p in sorted(glob.glob(os.path.join(index_dir, f"{ds}_*.idx"))):
            base = os.path.basename(p)
            if kinds and not any(f"_{k}" in base for k in kinds):
                continue
            files.append((os.path.getsize(p), p))
    if not files:
        print("   SKIP  no matching *.idx files")
        return None

    files.sort()
    failures = 0
    for size, p in files:
        mb = size / 1e6
        name = os.path.basename(p)
        if mb > max_mb:
            print(f"   skip  {name:<42s} {mb:8.0f} MB  (over --max-mb {max_mb})")
            continue
        t0 = time.perf_counter()
        try:
            idx = faiss.read_index(p)
            print(f"   {OK}  {name:<42s} {mb:8.0f} MB  d={idx.d} "
                  f"ntotal={idx.ntotal}  {time.perf_counter()-t0:.1f}s")
            del idx
        except Exception as e:
            failures += 1
            msg = str(e).strip().splitlines()[-1][:100]
            print(f"   {NO}  {name:<42s} {mb:8.0f} MB  {msg}")
    return failures == 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir",
                    default=os.environ.get("INDEX_DIR", "indexes_router"))
    ap.add_argument("--datasets", nargs="+", default=["sift1m", "gist1m"])
    ap.add_argument("--kinds", nargs="+", default=None,
                    help="substring filter, e.g. suco cspg shg")
    ap.add_argument("--max-mb", type=float, default=4096.0,
                    help="skip index files larger than this")
    ap.add_argument("--skip-files", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("SuCo load pre-flight")
    print("=" * 68)
    check_provenance()
    ctor = check_constructor()
    trip = check_round_trip()
    files = None if args.skip_files else check_real_files(
        args.index_dir, args.datasets, args.max_mb, args.kinds)

    print("\n" + "=" * 68)
    # Only the checks that exercise the real C++ load path can block. `ctor`
    # blocks solely when IndexSuCo is absent entirely, which means the wrong
    # library rather than a stale one.
    blocking = (ctor is False) or (trip is False) or (files is False)
    if trip is None and files is None:
        print("VERDICT: INCONCLUSIVE -- neither the round trip nor a real index")
        print("         was loaded. Re-run with --index-dir pointing at the")
        print("         prebuilt indexes, on a node with room for them.")
        return 2
    if blocking:
        print("VERDICT: the SuCo load will fail again.")
        print("""
The fix in 17f60df4 is C++ (faiss/IndexSuCo.cpp, faiss/IndexSuCo.h,
faiss/impl/index_read.cpp).  Pulling the branch is not enough -- the
swigfaiss*.so the job imports has to be rebuilt and reinstalled:

    cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \\
          -DFAISS_OPT_LEVEL=avx512 -DCMAKE_BUILD_TYPE=Release .
    make -C build -j32 faiss swigfaiss_avx512
    make -C build/faiss/python install       # into the env the job activates

Then re-run this script.  Note the failing job imported faiss from
/leonardo/home/userexternal/afilippa/miniconda3/envs/anns -- confirm the
rebuild lands in the prefix `conda activate anns` actually resolves to, and
that you can write to it, before queueing anything.

Steps 1-5 of m2b_hpc.sbatch do not touch SuCo and are unaffected; only step 6
(distance counts) needs this.  Run with SKIP_DISTCOUNT=1 meanwhile.""")
        return 1

    print("VERDICT: SuCo indexes load. Step 6 is safe to run.")
    if trip is None:
        print("NOTE: the round trip was skipped, so checks 2 and 4 carry the")
        print("      verdict on their own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
