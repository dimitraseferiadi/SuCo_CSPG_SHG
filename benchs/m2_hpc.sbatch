#!/bin/bash
#SBATCH --job-name=suco-m2
#SBATCH --account=EUHPC_D32_102
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.err
#
# Memory-bandwidth evidence for the claim that FAISS's distance path is already
# byte-bound, and that this is what compresses the margin available to CSPG's and
# SHG's distance-count reductions.
#
# Two parts:
#   bench_m2_bandwidth.py   zero-padding ablation + STREAM ceilings. Needs no
#                           FAISS changes and no prebuilt indexes; it builds one
#                           small HNSW per dataset and transplants its topology.
#   bench_m2_distcount.py   distance evaluations per query at matched recall.
#                           Reports HNSW32/HNSW48 from FAISS's built-in counter
#                           and skips SuCo/CSPG/SHG unless the counters from
#                           benchs/M2_DISTANCE_COUNTERS.md have been compiled in.
#
# Much cheaper than the router job: hours, not days. Run it on the same node type
# as the main results, since the whole point is the memory system of that node.
#
# Usage:
#   sbatch benchs/m2_hpc.sbatch                     # sift1m + gist1m
#   sbatch benchs/m2_hpc.sbatch sift1m deep1m gist1m openai1m
#   SKIP_DISTCOUNT=1 sbatch benchs/m2_hpc.sbatch    # bandwidth part only
#   PAD_DIMS="128 256 512 1024" sbatch benchs/m2_hpc.sbatch   # explicit widths

set -uo pipefail
mkdir -p logs/slurm

# One low-d and one high-d set is enough for the padding slope: the ablation
# controls for the graph, so extra datasets add confirmation rather than
# coverage. openai1m is worth adding if the high-d end looks anomalous.
DEFAULT_DATASETS=(sift1m gist1m)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

SKIP_DISTCOUNT=${SKIP_DISTCOUNT:-0}
# Widths for the zero-padding ablation. Zero columns leave every squared L2
# distance unchanged, so the graph, the visited set and the distance count are
# identical at every width and only the bytes per evaluation change. Left empty,
# the benchmark uses d, 2d, 4d, 8d from the dataset itself; set PAD_DIMS to a
# space-separated list of absolute widths to override (e.g. "128 256 512").
PAD_DIMS=${PAD_DIMS:-}
EF_SEARCH=${EF_SEARCH:-100}
STREAM_MB=${STREAM_MB:-4096}

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

# --- Environment (identical to the router / thread jobs) ------------------
module purge
module load profile/base
module load gcc/11.3.0

source ~/miniconda3/bin/activate
conda activate anns

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FAISS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

# No OMP_PROC_BIND / OMP_PLACES — see the note in thread_hpc.sbatch. Achieved
# bandwidth is meaningless if the team is confined to one core.

echo "--- CPU affinity ---"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
taskset -cp $$ 2>/dev/null || echo "taskset unavailable"
python -c "import os; m=os.sched_getaffinity(0); print(f'sched_getaffinity: {len(m)} CPUs')"
echo "--------------------"

cd "$SLURM_SUBMIT_DIR"

DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
mkdir -p "$OUT_DIR"

python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

echo "=== M2 run start $(date '+%F %T') ==="
echo "    datasets  : ${DATASETS[*]}"
echo "    pad dims  : ${PAD_DIMS:-auto (d, 2d, 4d, 8d)}"
echo "    efSearch  : ${EF_SEARCH}"

# --- Part 1: bandwidth ceilings + zero-padding ablation -------------------
for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  bandwidth / padding: $DS"
    echo "############################################################"

    # Padded widths are left to the benchmark, which defaults to d, 2d, 4d and 8d
    # from the dataset's own d. Deriving them here would mean a second dataset
    # load just to read a shape, and PAD_DIMS below overrides when needed.
    python benchs/bench_m2_bandwidth.py \
        --dataset "$DS" \
        --data-dir  "$DATA_DIR" \
        --index-dir "$INDEX_DIR" \
        --out-dir   "$OUT_DIR" \
        ${PAD_DIMS:+--pad-dims $PAD_DIMS} \
        --ef ${EF_SEARCH} \
        --threads ${SLURM_CPUS_PER_TASK} \
        --stream-mb ${STREAM_MB} \
        2>&1 | tee "$OUT_DIR/log_m2_bandwidth_${DS}.txt" \
        || echo "!!! $DS bandwidth FAILED (continuing) !!!"
done

# --- Part 2: distance evaluations per query at matched recall -------------
if [ "${SKIP_DISTCOUNT}" != "1" ]; then
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  distance counts"
    echo "############################################################"
    echo "NOTE: SuCo/CSPG/SHG rows appear only if the counters described in"
    echo "      benchs/M2_DISTANCE_COUNTERS.md are compiled into this build."

    python benchs/bench_m2_distcount.py \
        --dataset "${DATASETS[@]}" \
        --data-dir    "$DATA_DIR" \
        --index-dir   "$INDEX_DIR" \
        --results-dir "$OUT_DIR" \
        --out-dir     "$OUT_DIR" \
        --targets 0.95 0.99 \
        2>&1 | tee "$OUT_DIR/log_m2_distcount.txt" \
        || echo "!!! distcount FAILED !!!"
fi

echo ""
echo "=== M2 run done $(date '+%F %T') ==="
echo "Key numbers:"
echo "  d(log t)/d(log d) from the padding fit  — 1.0 means byte-bound"
echo "  peak achieved GB/s as a fraction of the STREAM ceiling"
echo "  distance evals/query at matched recall, CSPG/SHG vs their HNSW baseline"
echo "Results: $OUT_DIR/m2_bandwidth.json, m2_padding_*.csv, m2_distcount_*.json"
