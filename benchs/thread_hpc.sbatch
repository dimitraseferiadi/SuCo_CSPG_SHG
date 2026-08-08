#!/bin/bash
#SBATCH --job-name=suco-threads
#SBATCH --account=EUHPC_D32_102
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.err
#
# Thread-scaling experiment: QPS / speedup / parallel efficiency vs
# faiss.omp_set_num_threads at recall ≥ 0.90 and 0.95 (k=10).
#
# Loads the indexes the router job already built in $WORK/dhm/indices — nothing
# is rebuilt. Merges a "thread_scaling" node into benchs/results_router/
# results_<ds>.json, leaving every other key in those files untouched, so
# plot_router_paper.py picks it up with no changes.
#
# Usage:
#   sbatch benchs/thread_hpc.sbatch                       # the 9 indexed datasets
#   sbatch benchs/thread_hpc.sbatch sift1m gist1m         # subset
#   THREAD_COUNTS="1 2 4 8 16" sbatch benchs/thread_hpc.sbatch   # match the old tables
#   INDEX_TYPES="suco hnsw32" sbatch benchs/thread_hpc.sbatch    # subset of indexes
#
# Run the 1M datasets first: they are cheap and confirm the pipeline before the
# 10M ones (where SuCo at 1 thread is the long pole — hours, not minutes).

set -uo pipefail
mkdir -p logs/slurm

# Cheap-to-expensive, so a walltime kill loses the least. Enron and MSTuring10M
# are deliberately absent: the router run never produced their indexes, so they
# would load the (large) base vectors and then skip every index. Add them back
# here — or pass them on the submit line — once router_hpc.sbatch has built them.
DEFAULT_DATASETS=(sift1m gist1m deep1m msong openai1m uqv sift10m deep10m spacev10m)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Read env knobs BEFORE `set --` wipes the positional args.
THREAD_COUNTS=${THREAD_COUNTS:-"1 2 4 8 16 32"}
# All six indexes plot_router_paper.py draws (INDICES), not the five in
# DEFAULT_INDEX_TYPES — omitting opq_ivfpq leaves an OPQ-IVFPQ-shaped hole in
# the thread-scaling panels while every other figure in the paper has that line.
INDEX_TYPES=${INDEX_TYPES:-"suco shg cspg hnsw32 hnsw48 opq_ivfpq"}
RECALL_TARGETS=${RECALL_TARGETS:-"0.90 0.95"}

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

# --- Environment (identical to the router / per-algo jobs) ----------------
module purge
module load profile/base
module load gcc/11.3.0
# (no CUDA; CPU-only FAISS)

source ~/miniconda3/bin/activate
conda activate anns

# Ceiling for the sweep. The benchmark calls faiss.omp_set_num_threads() per
# point and drops any count above the cores actually available, so this must
# match --cpus-per-task or the top of the sweep is silently truncated.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FAISS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

# Pin the OMP team so thread counts map to distinct cores; without this the
# 8/16/32-thread points drift as the runtime migrates threads and the
# efficiency numbers pick up scheduler noise rather than scaling behaviour.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd "$SLURM_SUBMIT_DIR"   # = your SuCo checkout

DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices          # router's prebuilt indexes live here
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
mkdir -p "$OUT_DIR"

python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

echo "=== thread-scaling run start $(date '+%F %T') ==="
echo "    datasets : ${DATASETS[*]}"
echo "    indexes  : ${INDEX_TYPES}"
echo "    threads  : ${THREAD_COUNTS}  (cpus-per-task=${SLURM_CPUS_PER_TASK})"
echo "    targets  : ${RECALL_TARGETS}"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  dataset: $DS"
    echo "############################################################"

    # Own log file per dataset — deliberately NOT log_router_<ds>.txt, which
    # belongs to the full router run and would be truncated on reopen.
    python benchs/bench_thread_scaling.py \
        --dataset "$DS" \
        --data-dir  "$DATA_DIR" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$OUT_DIR" \
        --index-type ${INDEX_TYPES} \
        --threads ${THREAD_COUNTS} \
        --recall-targets ${RECALL_TARGETS} \
        2>&1 | tee "$OUT_DIR/log_threads_${DS}.txt" \
        || echo "!!! $DS FAILED (continuing to next dataset) !!!"
done

echo ""
echo "=== thread-scaling run done $(date '+%F %T') ==="
echo "Regenerate figures + tables with:"
echo "  python benchs/plot_router_paper.py --results-dir benchs/results_router --out-dir benchs/figures_router"