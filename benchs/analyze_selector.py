#!/usr/bin/env python3
"""
Index-selection analysis over the eleven-dataset cross-algorithm matrix.

Answers, in order:
  Q1  How much does N* actually vary across the matrix?
  Q2  Do the geometry descriptors (LID, rho_16, sigma_d/mean_d) predict
      log10(N*) better than n and d alone, under leave-one-dataset-out?
  Q3  What is the LODO regret of a fitted decision rule against the
      "always HNSW32" / "always most-frequent-winner" / random baselines?

Cost model (Section 12.2.9): total(N) = build_time_s + N * per_query_s,
where per_query_s is taken at the cheapest measured operating point that
reaches the recall floor tau at the given k ("matched recall").

Usage:  python3 benchs/analyze_selector.py [--results-dir benchs/results_router]
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

# The eleven-dataset matrix of Table 10.1. The bigann* files in the same
# directory are the SIFT cardinality sweep and are handled separately.
MATRIX = [
    "sift1m", "sift10m", "gist1m", "deep1m", "deep10m", "spacev10m",
    "msong", "enron", "openai1m", "msturing10m", "uqv",
]

GRAPHS = ["HNSW32", "HNSW48", "CSPG", "SHG"]
PARTITION = "SuCo"

TAUS = [0.90, 0.95, 0.99]
KS = [10, 50, 100]
NS = np.logspace(3, 9, 7)  # 1e3 .. 1e9 queries per index lifetime

FEATS_BASE = ["log_n", "log_d"]
FEATS_GEOM = ["lid_mle", "rho16", "cov_pdist"]


# ---------------------------------------------------------------- loading


def load_matrix(results_dir):
    out = {}
    for name in MATRIX:
        path = os.path.join(results_dir, f"results_{name}.json")
        if not os.path.exists(path):
            print(f"  [skip] {name}: no file at {path}")
            continue
        out[name] = json.load(open(path))
    return out


def features(rec):
    f = rec["features"]
    return {
        "n": float(f["n"]),
        "d": float(f["d"]),
        "log_n": np.log10(float(f["n"])),
        "log_d": np.log10(float(f["d"])),
        "lid_mle": float(f["lid_mle"]),
        "rho16": float(f["kmeans_inertia_ratio_16"]),
        "cov_pdist": float(f["pdist_std"]) / float(f["pdist_mean"]),
    }


def matched(rec, method, k, tau):
    """Cheapest measured operating point reaching recall >= tau. None if
    the method never reaches tau at this k (a topological ceiling)."""
    pts = rec.get(f"recall_k{k}", {}).get(method)
    if not pts:
        return None
    ok = [p for p in pts if p["recall"] >= tau]
    if not ok:
        return None
    return min(ok, key=lambda p: p["ms_per_query"])


def build_s(rec, method):
    c = rec.get("construction", {}).get(method)
    return None if c is None else float(c["build_time_s"])


def total_cost(rec, method, k, tau, N):
    """Build + N * per-query, in seconds. None if infeasible."""
    m = matched(rec, method, k, tau)
    b = build_s(rec, method)
    if m is None or b is None:
        return None
    return b + N * m["ms_per_query"] / 1000.0


# ---------------------------------------------------------------- Q1: N*


def crossover(rec, k, tau, graph):
    """N* where the graph's cumulative cost equals SuCo's.

    N* = (build_g - build_s) / (q_s - q_g).  Returns None when either method
    is infeasible, or when the graph is not both costlier to build and
    cheaper to query (no finite crossover)."""
    ms, mg = matched(rec, PARTITION, k, tau), matched(rec, graph, k, tau)
    bs, bg = build_s(rec, PARTITION), build_s(rec, graph)
    if ms is None or mg is None or bs is None or bg is None:
        return None
    dq = (ms["ms_per_query"] - mg["ms_per_query"]) / 1000.0
    db = bg - bs
    if dq <= 0 or db <= 0:
        return None
    return db / dq


def best_graph(rec, k, tau):
    """Graph with the lowest matched-recall per-query cost."""
    cand = [(g, matched(rec, g, k, tau)) for g in GRAPHS]
    cand = [(g, m) for g, m in cand if m is not None]
    if not cand:
        return None
    return min(cand, key=lambda t: t[1]["ms_per_query"])[0]


def collect_nstar(data):
    """One N* per (dataset, k, tau), against the best graph at that cell."""
    rows = []
    for name, rec in data.items():
        F = features(rec)
        for k, tau in itertools.product(KS, TAUS):
            g = best_graph(rec, k, tau)
            if g is None:
                continue
            ns = crossover(rec, k, tau, g)
            if ns is None or not np.isfinite(ns) or ns <= 0:
                continue
            rows.append(dict(dataset=name, k=k, tau=tau, graph=g,
                             nstar=ns, log_nstar=np.log10(ns), **F))
    return rows


# ---------------------------------------------------------------- regression


def ols_fit(X, y):
    Xa = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return beta


def ols_pred(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def lodo_regression(rows, feat_names):
    """Leave-one-dataset-out prediction of log10(N*). Returns (preds, truth,
    per-fold MAE in dex)."""
    datasets = sorted({r["dataset"] for r in rows})
    preds, truth, folds = [], [], {}
    for held in datasets:
        tr = [r for r in rows if r["dataset"] != held]
        te = [r for r in rows if r["dataset"] == held]
        if not te or len(tr) < len(feat_names) + 2:
            continue
        Xtr = np.array([[r[f] for f in feat_names] for r in tr])
        ytr = np.array([r["log_nstar"] for r in tr])
        Xte = np.array([[r[f] for f in feat_names] for r in te])
        yte = np.array([r["log_nstar"] for r in te])
        beta = ols_fit(Xtr, ytr)
        p = ols_pred(beta, Xte)
        preds.extend(p)
        truth.extend(yte)
        folds[held] = float(np.mean(np.abs(p - yte)))
    preds, truth = np.array(preds), np.array(truth)
    return preds, truth, folds


def r2(pred, truth):
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------- Q3: regret


def oracle_and_costs(rec, k, tau, N):
    """{method: cost} over feasible methods, plus the argmin."""
    costs = {}
    for m in [PARTITION] + GRAPHS:
        c = total_cost(rec, m, k, tau, N)
        if c is not None:
            costs[m] = c
    if not costs:
        return None, None
    return min(costs, key=costs.get), costs


def regret(costs, pick):
    """Relative regret of `pick` against the oracle. None if pick infeasible."""
    if pick not in costs:
        return None
    best = min(costs.values())
    return (costs[pick] - best) / best


def rule_constant(nstar_hat, N, graph_default):
    """The decision list, collapsed to its operative content."""
    return PARTITION if N < nstar_hat else graph_default


def evaluate_rules(data, nstar_rows, rng):
    datasets = sorted(data.keys())
    per_ds = {d: {r: [] for r in
                  ["fitted_rule", "always_hnsw32", "always_mode", "random", "oracle_nstar"]}
              for d in datasets}
    n_cells = n_infeasible = 0

    for held in datasets:
        tr_rows = [r for r in nstar_rows if r["dataset"] != held]
        if not tr_rows:
            continue
        # Threshold fitted on the training folds only.
        nstar_hat = 10 ** np.median([r["log_nstar"] for r in tr_rows])
        # Most frequent global winner on the training folds.
        wins = {}
        for name in datasets:
            if name == held:
                continue
            for k, tau, N in itertools.product(KS, TAUS, NS):
                o, _ = oracle_and_costs(data[name], k, tau, N)
                if o:
                    wins[o] = wins.get(o, 0) + 1
        mode_pick = max(wins, key=wins.get) if wins else "HNSW32"
        # Best graph default, also training-only (per (k,tau) majority).
        gwins = {}
        for name in datasets:
            if name == held:
                continue
            for k, tau in itertools.product(KS, TAUS):
                g = best_graph(data[name], k, tau)
                if g:
                    gwins[g] = gwins.get(g, 0) + 1
        graph_default = max(gwins, key=gwins.get) if gwins else "HNSW32"

        rec = data[held]
        for k, tau, N in itertools.product(KS, TAUS, NS):
            o, costs = oracle_and_costs(rec, k, tau, N)
            if o is None:
                n_infeasible += 1
                continue
            n_cells += 1
            picks = {
                "fitted_rule": rule_constant(nstar_hat, N, graph_default),
                "always_hnsw32": "HNSW32",
                "always_mode": mode_pick,
                "random": rng.choice(sorted(costs.keys())),
                "oracle_nstar": o,
            }
            for rule, p in picks.items():
                r = regret(costs, p)
                # An infeasible pick is a hard failure: charge it the worst
                # feasible regret in the cell rather than dropping it.
                if r is None:
                    r = max(regret(costs, m) for m in costs)
                per_ds[held][rule].append(r)
    return per_ds, n_cells, n_infeasible


# ---------------------------------------------------------------- reporting


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchs/results_router")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchs/selector_analysis.tsv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = load_matrix(args.results_dir)
    print(f"\nLoaded {len(data)} datasets: {', '.join(sorted(data))}\n")

    # ---- Q1 -----------------------------------------------------------
    print("=" * 72)
    print("Q1  Spread of the crossover query volume N*")
    print("=" * 72)
    rows = collect_nstar(data)
    if not rows:
        print("  no finite crossovers found -- check the cost model")
        return
    ns = np.array([r["nstar"] for r in rows])
    print(f"  cells with a finite N*: {len(rows)}")
    print(f"  N* range      : {ns.min():.3g} .. {ns.max():.3g}")
    print(f"  spread        : {np.log10(ns.max() / ns.min()):.2f} decades")
    print(f"  median        : {np.median(ns):.3g}")
    print(f"  IQR           : {pct(ns, 25):.3g} .. {pct(ns, 75):.3g}\n")

    print(f"  {'dataset':<14}{'k':>4}{'tau':>6}{'graph':>9}{'N*':>12}")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["k"], r["tau"])):
        print(f"  {r['dataset']:<14}{r['k']:>4}{r['tau']:>6.2f}"
              f"{r['graph']:>9}{r['nstar']:>12.3g}")

    # per-dataset spread, pooling k and tau
    print(f"\n  {'dataset':<14}{'median N*':>12}{'min':>12}{'max':>12}")
    for ds in sorted({r["dataset"] for r in rows}):
        v = np.array([r["nstar"] for r in rows if r["dataset"] == ds])
        print(f"  {ds:<14}{np.median(v):>12.3g}{v.min():>12.3g}{v.max():>12.3g}")

    # ---- Q2 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("Q2  Does geometry predict log10(N*) beyond n and d?  (LODO)")
    print("=" * 72)

    # marginal correlations
    print("\n  Pearson r of each feature with log10(N*):")
    y = np.array([r["log_nstar"] for r in rows])
    for f in FEATS_BASE + FEATS_GEOM:
        x = np.array([r[f] for r in rows])
        rr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else float("nan")
        print(f"    {f:<12} r = {rr:+.3f}")

    models = {
        "intercept only (global constant)": [],
        "n, d": FEATS_BASE,
        "geometry only": FEATS_GEOM,
        "n, d + geometry": FEATS_BASE + FEATS_GEOM,
    }
    print(f"\n  {'model':<34}{'MAE (dex)':>12}{'R^2':>10}")
    baseline_mae = None
    for label, feats in models.items():
        if not feats:
            # LODO constant: predict the training-fold median
            preds, truth = [], []
            for held in sorted({r["dataset"] for r in rows}):
                tr = [r["log_nstar"] for r in rows if r["dataset"] != held]
                te = [r["log_nstar"] for r in rows if r["dataset"] == held]
                preds.extend([np.median(tr)] * len(te))
                truth.extend(te)
            preds, truth = np.array(preds), np.array(truth)
            folds = {}
        else:
            preds, truth, folds = lodo_regression(rows, feats)
        mae = float(np.mean(np.abs(preds - truth)))
        if baseline_mae is None:
            baseline_mae = mae
        print(f"  {label:<34}{mae:>12.3f}{r2(preds, truth):>10.3f}")
    print(f"\n  (MAE in dex: 0.30 dex = a factor of 2 error in N*.)")

    # ---- Q3 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("Q3  LODO regret of the fitted rule vs. baselines")
    print("=" * 72)
    per_ds, n_cells, n_infeasible = evaluate_rules(data, rows, rng)
    print(f"\n  evaluated cells: {n_cells}   fully-infeasible cells: {n_infeasible}")
    print(f"  (grid: {len(KS)} k x {len(TAUS)} tau x {len(NS)} N x {len(data)} datasets)")

    pooled = {}
    for rule in ["oracle_nstar", "fitted_rule", "always_hnsw32", "always_mode", "random"]:
        vals = np.concatenate([per_ds[d][rule] for d in per_ds if per_ds[d][rule]])
        pooled[rule] = vals

    print(f"\n  {'rule':<18}{'mean':>10}{'median':>10}{'p90':>10}{'max':>12}")
    for rule, v in pooled.items():
        print(f"  {rule:<18}{np.mean(v):>10.3f}{np.median(v):>10.3f}"
              f"{pct(v, 90):>10.3f}{v.max():>12.1f}")

    print(f"\n  per-dataset mean regret:")
    print(f"  {'dataset':<14}{'fitted':>10}{'HNSW32':>10}{'mode':>10}{'random':>10}")
    for ds in sorted(per_ds):
        row = per_ds[ds]
        if not row["fitted_rule"]:
            continue
        print(f"  {ds:<14}"
              f"{np.mean(row['fitted_rule']):>10.3f}"
              f"{np.mean(row['always_hnsw32']):>10.3f}"
              f"{np.mean(row['always_mode']):>10.3f}"
              f"{np.mean(row['random']):>10.3f}")

    # ---- dump ---------------------------------------------------------
    with open(args.out, "w") as fh:
        cols = ["dataset", "k", "tau", "graph", "nstar", "log_nstar",
                "n", "d", "lid_mle", "rho16", "cov_pdist"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n  wrote per-cell N* table to {args.out}\n")


if __name__ == "__main__":
    main()
