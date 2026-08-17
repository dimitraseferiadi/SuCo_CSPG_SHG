#!/bin/bash
#SBATCH --job-name=suco-eps-recall
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
# eps-recall: recall that credits a returned point by distance rather than by
# id -- a point counts when d(q,r) <= (1+eps) d_k(q).
#
# The measurement this exists for is UQ-V, where every graph index saturates at
# an id-recall of 0.82-0.84 and SuCo at 0.879 (§V-C). UQ-V is full of duplicate
# base vectors, so a query's k-th neighbour is one of many points at the same
# distance and an index that returns a different copy is scored as a miss. If
# eps-recall closes that gap, the ceiling is a tie-breaking artefact rather
# than a search failure, and §V-C's "topological ceiling of greedy descent"
# reading has to change accordingly.
#
# Cheap: no index is built if the router's are on disk (--index-dir), and the
# metric is one exact-distance recomputation per search. Hours, not days --
# most of the wall clock is the searches themselves. The --plateau pass is the
# exception: it is a full brute-force scan per query block, so it is on only
# for the 10^6-scale sets by default.
#
# Usage:
#   sbatch benchs/eps_recall_hpc.sbatch                    # uqv (+ plateau)
#   sbatch benchs/eps_recall_hpc.sbatch uqv gist1m sift1m
#   KS="1 10 100" sbatch benchs/eps_recall_hpc.sbatch uqv
#   PLATEAU=0 sbatch benchs/eps_recall_hpc.sbatch uqv      # skip brute force
#   INDEX_TYPES="suco hnsw32" sbatch benchs/eps_recall_hpc.sbatch uqv

set -uo pipefail
mkdir -p logs/slurm

# UQ-V is the dataset the metric was written for. The others are the control:
# if eps-recall also lifts recall where no ceiling was reported, the metric is
# measuring its own looseness rather than UQ-V's duplicates.
DEFAULT_DATASETS=(uqv)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# k=10 is the paper's headline. k=1 and k=100 bracket it: at k=1 the tie
# plateau is at its widest relative to k, at k=100 its narrowest.
KS=${KS:-"10"}
EPS=${EPS:-"0.0 0.001 0.005 0.01 0.02 0.05 0.1"}
INDEX_TYPES=${INDEX_TYPES:-"suco shg cspg hnsw32 hnsw48"}
PLATEAU=${PLATEAU:-1}

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

# --- Environment (identical to the router / m2 jobs) ----------------------
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

echo "--- CPU affinity ---"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
python -c "import os; m=os.sched_getaffinity(0); print(f'sched_getaffinity: {len(m)} CPUs')"
echo "--------------------"

cd "$SLURM_SUBMIT_DIR"

DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices          # the router's prebuilt indexes
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
mkdir -p "$OUT_DIR"

python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

PLATEAU_ARG=""
[ "$PLATEAU" = "1" ] && PLATEAU_ARG="--plateau"

echo "=== eps-recall run start $(date '+%F %T') ==="
echo "    datasets    : ${DATASETS[*]}"
echo "    index types : ${INDEX_TYPES}"
echo "    k           : ${KS}"
echo "    eps         : ${EPS}"
echo "    plateau     : ${PLATEAU}"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  eps-recall: $DS"
    echo "############################################################"

    # Missing indexes are built and saved into $INDEX_DIR under the router's
    # own names, so a rebuild here is reused by the next job rather than
    # thrown away.
    python benchs/bench_eps_recall.py \
        --dataset    "$DS" \
        --data-dir   "$DATA_DIR" \
        --index-dir  "$INDEX_DIR" \
        --output-dir "$OUT_DIR" \
        --index-type ${INDEX_TYPES} \
        --k ${KS} \
        --eps ${EPS} \
        ${PLATEAU_ARG} \
        2>&1 | tee "$OUT_DIR/log_eps_recall_${DS}.txt" \
        || echo "!!! $DS eps-recall FAILED (continuing) !!!"
done

echo ""
echo "=== eps-recall run done $(date '+%F %T') ==="
echo "    results: $OUT_DIR/eps_recall_<dataset>.json"
