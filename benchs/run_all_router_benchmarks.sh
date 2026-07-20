#!/bin/bash
#
# run_all_router_benchmarks.sh
#
# Runs the router-training benchmark suite (SuCo / SHG / CSPG with paper
# defaults) across all 11 datasets, writes one results_<dataset>.json per
# dataset, and saves indices to INDEX_DIR for reuse.
#
# Usage:
#   chmod +x benchs/run_all_router_benchmarks.sh
#   ./benchs/run_all_router_benchmarks.sh                    # all 11 datasets
#   ./benchs/run_all_router_benchmarks.sh sift1m             # one dataset
#   ./benchs/run_all_router_benchmarks.sh sift1m gist1m deep1m   # several
#
# Subset of benchmarks / indexes via env vars:
#   BENCHMARKS="construction recall_k10 recall_k20" \
#     ./benchs/run_all_router_benchmarks.sh sift1m
#
#   INDEX_TYPES="suco shg" \
#     ./benchs/run_all_router_benchmarks.sh openai1m
#
# Available datasets (positional args):
#   sift1m sift10m gist1m deep1m deep10m spacev10m
#   msong enron openai1m msturing10m uqv
#
# Available index types:
#   suco shg cspg hnsw32 hnsw48
#
# Environment variables:
#   DATA_DIR     — dataset root                  (default: $DATA_DIR, else <repo>/data)
#   INDEX_DIR    — saved indices                 (default: $INDEX_DIR, else <repo>/indices)
#   OUTPUT_DIR   — result JSONs                  (default: benchs/results_router)
#   BENCHMARKS   — space-separated               (default: all)
#   INDEX_TYPES  — suco/shg/cspg/hnsw32/hnsw48   (default: "suco shg cspg hnsw32 hnsw48")
#
# Datasets are resolved by benchs/bench_datasets.py, which reads the raw Deep1B
# chunks (base_00 / learn_00) directly -- no prepare step is needed. A dataset
# check runs first and aborts on any missing or mis-paired file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-${REPO_DIR:-$PWD}/data}"
INDEX_DIR="${INDEX_DIR:-${REPO_DIR:-$PWD}/indices}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_router}"
BENCHMARKS="${BENCHMARKS:-all}"
INDEX_TYPES="${INDEX_TYPES:-suco shg cspg hnsw32 hnsw48}"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_router_paper.py"

ALL_DATASETS=(
    sift1m sift10m gist1m
    deep1m deep10m
    spacev10m
    msong enron openai1m
    msturing10m uqv
)

read -r -a BENCHMARK_ARGS  <<< "${BENCHMARKS}"
read -r -a INDEX_TYPE_ARGS <<< "${INDEX_TYPES}"

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

# Validate dataset names early.
for ds in "${DATASETS[@]}"; do
    found=0
    for valid in "${ALL_DATASETS[@]}"; do
        if [ "$ds" = "$valid" ]; then found=1; break; fi
    done
    if [ $found -eq 0 ]; then
        echo "Unknown dataset: '$ds'"
        echo "Valid choices: ${ALL_DATASETS[*]}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Auto-prepare deep1b/base.fvecs + learn.fvecs when deep* datasets are selected.
# Idempotent: skips if files already exist with sufficient size.
# Each fvecs row for d=96 occupies 4 + 96*4 = 388 bytes.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Router Benchmark Suite (SuCo / SHG / CSPG, paper defaults)"
echo "============================================================"
echo "  Data dir:    ${DATA_DIR}"
echo "  Index dir:   ${INDEX_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Benchmarks:  ${BENCHMARK_ARGS[*]}"
echo "  Index types: ${INDEX_TYPE_ARGS[*]}"
echo "============================================================"

mkdir -p "${INDEX_DIR}" "${OUTPUT_DIR}"

# Fail fast if a dataset is missing or mis-pathed
python3 "${SCRIPT_DIR}/check_datasets.py" --data-dir "${DATA_DIR}" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Dataset: ${ds}"
    echo "############################################################"

    LOG_FILE="${OUTPUT_DIR}/log_router_${ds}.txt"

    python3 "${BENCH_SCRIPT}" \
        --data-dir   "${DATA_DIR}" \
        --index-dir  "${INDEX_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset    "${ds}" \
        --benchmark  "${BENCHMARK_ARGS[@]}" \
        --index-type "${INDEX_TYPE_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Done: ${ds}  (log: ${LOG_FILE})"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "  Results: ${OUTPUT_DIR}/results_*.json"
echo "  Logs:    ${OUTPUT_DIR}/log_router_*.txt"
echo "============================================================"
