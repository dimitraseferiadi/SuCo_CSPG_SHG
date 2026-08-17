#!/usr/bin/env python3
"""
benchs/analyze_m2b.py

Assembles the M2b outputs into the numbers Sec. V-D needs, with each achieved
bandwidth divided by the ceiling that belongs to its access pattern rather than
by a sequential copy.

Reads from --results-dir:
  m2_ceilings.json / m2_ceilings_1t.json    STREAM + BW_rand(4d, C)
  m2_padding_<ds>.csv                       the ablation
  m2_threadsweep_<ds>.csv                   saturation
  m2_efsweep_<ds>.csv                       d(log t)/d(log ndis)
  m2_perf_<ds>_d<W>_p{1,11}.csv             measured DRAM traffic

Usage
  python benchs/analyze_m2b.py --results-dir benchs/results_router
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

BYTES_PER_FLOAT = 4


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def bw_rand_table(ceil, conc="C4"):
    if not ceil:
        return {}
    return {int(r["d"]): float(r[conc])
            for r in ceil.get("bw_rand_gbs", []) if conc in r}


def parse_perf(path):
    """perf stat -x, output -> {event: bytes}.

    The unit column matters.  Intel exposes cas_count_read with a .scale of
    1/16384 and a .unit of MiB, so perf reports MiB already and multiplying by
    64 would overcount by 2^14.  Raw core-PMU miss counts carry no unit and do
    need the 64 B cache line.
    """
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) < 3:
                continue
            raw, unit, event = f[0], f[1].strip(), f[2]
            try:
                val = float(raw)
            except ValueError:
                continue          # <not counted> / <not supported>
            key = event.split("/")[1] if "/" in event else event
            byts = val * (1024 ** 2) if unit.lower() in ("mib",) else val * 64.0
            out[key] = out.get(key, 0.0) + byts
    return out or None


def measured_bytes_per_pass(results_dir, ds, width):
    """(counts at 11 passes - counts at 1 pass) / 10.

    Build, dataset load and interpreter start-up appear identically in both and
    cancel, which is the only reason a counter wrapped around a whole process
    can say anything about one search pass.
    """
    a = parse_perf(os.path.join(results_dir, f"m2_perf_{ds}_d{width}_p1.csv"))
    b = parse_perf(os.path.join(results_dir, f"m2_perf_{ds}_d{width}_p11.csv"))
    if not a or not b:
        return None
    total = 0.0
    for k in b:
        if k in a:
            total += (b[k] - a[k]) / 10.0
    return total if total > 0 else None


def analyse_dataset(results_dir, ds, bwr, ceil):
    pad = read_csv(os.path.join(results_dir, f"m2_padding_{ds}.csv"))
    if not pad:
        return
    pad = [r for r in pad if r["invariant"] == "True"]
    ndis = float(pad[0]["ndis_per_query"])

    print(f"\n{'='*78}\n{ds}   (ndis/query = {ndis:.1f}, held fixed by construction)\n{'='*78}")

    # nq is needed to turn measured per-pass traffic into per-query traffic.
    nq = {"gist1m": 1000, "openai1m": 1000, "msong": 200, "enron": 200}.get(ds, 10000)

    print(f"{'d':>6} {'B=4d':>7} {'lines':>6} {'achieved':>9} {'BW_rand':>8} "
          f"{'%rand':>6} | {'measured':>9} {'x infer':>8} {'%rand':>6}")
    for r in pad:
        d = int(r["d"])
        ach = float(r["achieved_gbs"])
        ref = bwr.get(d)
        infer_bytes = ndis * d * BYTES_PER_FLOAT           # per query
        meas = measured_bytes_per_pass(results_dir, ds, d)
        line = (f"{d:6d} {d*4:7d} {d*4/64:6.1f} {ach:9.2f} "
                f"{(ref if ref else float('nan')):8.2f} "
                f"{(100*ach/ref if ref else float('nan')):5.0f}%")
        if meas:
            meas_q = meas / nq
            ratio = meas_q / infer_bytes
            ach_m = ach * ratio
            line += (f" | {meas_q/1e3:8.1f}kB {ratio:7.2f}x "
                     f"{(100*ach_m/ref if ref else float('nan')):5.0f}%")
        else:
            line += " | " + "  (no perf)".rjust(9)
        print(line)
    if any(measured_bytes_per_pass(results_dir, ds, int(r["d"])) for r in pad):
        print("  measured = uncore/LLC traffic per query; x infer = how much more")
        print("  than ndis*d*4, i.e. neighbour lists + visited table + heap")

    # --- saturation, anchored to the ablation so the nq bug cannot bite ------
    ts = read_csv(os.path.join(results_dir, f"m2_threadsweep_{ds}.csv"))
    if ts:
        print(f"\n  thread scaling (GB/s rebuilt from speedup x the ablation's "
              f"32-thread point,\n  which is immune to how the sweep normalised "
              f"its query count):")
        by_d = {}
        for r in ts:
            by_d.setdefault(int(r["d"]), []).append(r)
        for d, rows in sorted(by_d.items()):
            rows.sort(key=lambda r: int(r["threads"]))
            anchor = next((float(p["achieved_gbs"]) for p in pad
                           if int(p["d"]) == d), None)
            sp32 = float(rows[-1]["speedup_vs_1t"])
            series = []
            for r in rows:
                sp = float(r["speedup_vs_1t"])
                g = anchor * sp / sp32 if anchor else float("nan")
                series.append((int(r["threads"]), g, float(r["efficiency"])))
            ref = bwr.get(d)
            print(f"    d={d:5d}  " + "  ".join(
                f"t{t}:{g:.0f}" for t, g, _ in series) + " GB/s")
            last, prev = series[-1][1], series[-2][1]
            gain = (last - prev) / prev * 100 if prev else 0
            verdict = ("SATURATED (flat 16->32)" if gain < 3
                       else f"still climbing (+{gain:.0f}% at 16->32)")
            print(f"            {verdict}"
                  + (f", plateau = {100*last/ref:.0f}% of BW_rand({d*4} B)"
                     if ref and gain < 3 else ""))

    # --- the elasticity the conclusions rest on -----------------------------
    ef = read_csv(os.path.join(results_dir, f"m2_efsweep_{ds}.csv"))
    if ef:
        x = np.log([float(r["ndis_per_query"]) for r in ef])
        y = np.log([float(r["ms_per_query"]) for r in ef])
        rc = np.array([float(r["recall"]) for r in ef])
        s_all = np.polyfit(x, y, 1)[0]
        m = (rc >= 0.95) & (rc <= 0.995)
        s_band = np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 3 else float("nan")
        print(f"\n  d(log t)/d(log ndis) = {s_all:.3f} over the sweep, "
              f"{s_band:.3f} in the 0.95-0.995 band")
        print(f"    => halving the distance count buys {2**s_all:.2f}x, "
              f"not 'largely unchanged'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchs/results_router")
    ap.add_argument("--conc", default="C4",
                    help="memory-level parallelism column of BW_rand to use as "
                         "the denominator (C4 matches fvec_L2sqr_batch_4)")
    args = ap.parse_args()

    cpath = os.path.join(args.results_dir, "m2_ceilings.json")
    ceil = json.load(open(cpath)) if os.path.exists(cpath) else None
    if not ceil:
        print(f"no {cpath}; run m2b_hpc.sbatch first")
        return
    bwr = bw_rand_table(ceil, args.conc)

    st = ceil.get("stream_gbs", {})
    print("=" * 78)
    print("ceilings  (32 threads)")
    print("=" * 78)
    print(f"  STREAM copy {st.get('copy',0):7.1f}   triad {st.get('triad',0):7.1f}"
          f"   sequential read {ceil.get('sequential_read_gbs',0):7.1f} GB/s")
    print(f"  BW_rand(4d, {args.conc}): " + "  ".join(
        f"d{d}:{g:.0f}" for d, g in sorted(bwr.items())))
    print("\n  The paper's 97.0 GB/s was a numpy copy. Against the real copy "
          "ceiling it was\n  low by ~1.46x, and a copy ceiling is in any case "
          "the wrong reference for a\n  path that only reads and never writes.")

    for pth in sorted(glob.glob(os.path.join(args.results_dir, "m2_padding_*.csv"))):
        ds = os.path.basename(pth)[len("m2_padding_"):-len(".csv")]
        analyse_dataset(args.results_dir, ds, bwr, ceil)


if __name__ == "__main__":
    main()
