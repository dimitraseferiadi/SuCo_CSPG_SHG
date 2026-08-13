#!/bin/bash
#SBATCH --job-name=suco-m6
#SBATCH --account=EUHPC_D32_102
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.err
#
# Two baselines the original suite leaves out, added to bench_router_paper.py
# behind the "m6" family gate:
#
#   IVFFlat         the full-precision *partition* index. Without it SuCo -- a
#                   clustered-partition method -- is only ever measured against
#                   graphs and against LSH, so the family it actually belongs to
#                   has no representative on any axis.
#   OPQ-IVFPQ+SQ8   quantisation with an 8-bit scalar-quantised re-ranking stage.
#                   Plain OPQ-IVFPQ reaches Recall@10=0.95 on no dataset in the
#                   suite, which makes the quantisation family incomparable on
#                   the recall--throughput plane rather than merely worse. SQ8
#                   refine rather than the more usual RFlat: an exact fp32 refine
#                   stores the raw matrix a second time and yields an index
#                   larger than IndexFlat, forfeiting the footprint axis that is
#                   the family's only reason for inclusion.
#
# Writes into the same results_<ds>.json the router job produced, under the same
# keys, so plot_router_paper.py and the tables pick the new indexes up with no
# changes. Existing entries for the five original indexes are left untouched.
#
# Usage:
#   sbatch benchs/m6_hpc.sbatch                        # all 11 datasets
#   sbatch benchs/m6_hpc.sbatch sift1m gist1m          # subset
#   BENCHMARKS="recall_k10 construction" sbatch benchs/m6_hpc.sbatch
#   INDEX_TYPES=ivfflat sbatch benchs/m6_hpc.sbatch    # one of the two

set -uo pipefail
mkdir -p logs/slurm

# Cheap-to-expensive, so a walltime kill loses the least.
DEFAULT_DATASETS=(sift1m gist1m deep1m msong enron openai1m uqv sift10m deep10m spacev10m msturing10m)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Read env knobs BEFORE `set --` wipes the positional args.
INDEX_TYPES=${INDEX_TYPES:-"ivfflat opq_ivfpq_sq8"}
# "all" runs every per-index benchmark, which is what makes the two new indexes
# comparable on the non-Pareto axes (footprint, tails, cold/warm, MRE) as well as
# on recall--throughput. Narrow it if only the frontier is wanted.
BENCHMARKS=${BENCHMARKS:-"all"}

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

# No OMP_PROC_BIND / OMP_PLACES — see the note in thread_hpc.sbatch.

echo "--- CPU affinity ---"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
taskset -cp $$ 2>/dev/null || echo "taskset unavailable"
python -c "import os; m=os.sched_getaffinity(0); print(f'sched_getaffinity: {len(m)} CPUs')"
echo "--------------------"

cd "$SLURM_SUBMIT_DIR"

DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices          # new indexes are added alongside the old
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
mkdir -p "$OUT_DIR" "$INDEX_DIR"

python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

echo "=== M6 baselines run start $(date '+%F %T') ==="
echo "    datasets   : ${DATASETS[*]}"
echo "    indexes    : ${INDEX_TYPES}"
echo "    benchmarks : ${BENCHMARKS}"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  dataset: $DS"
    echo "############################################################"

    # Own log per dataset — not log_router_<ds>.txt, which belongs to the
    # original router run and would be truncated on reopen.
    python benchs/bench_router_paper.py \
        --dataset "$DS" \
        --data-dir   "$DATA_DIR" \
        --index-dir  "$INDEX_DIR" \
        --output-dir "$OUT_DIR" \
        --index-type ${INDEX_TYPES} \
        --benchmark  ${BENCHMARKS} \
        2>&1 | tee "$OUT_DIR/log_m6_${DS}.txt" \
        || echo "!!! $DS FAILED (continuing to next dataset) !!!"
done

echo ""
echo "=== M6 baselines run done $(date '+%F %T') ==="
echo "Then, with no cluster time needed:"
echo "  python benchs/analyze_ci.py --results-dir benchs/results_router \\"
echo "                             --out-dir benchs/figures_router/tables --paired-baseline"
echo "  python benchs/plot_paper_figures.py --out-dir benchs/figures_paper"
echo "  python benchs/plot_router_paper.py --results-dir benchs/results_router \\"
echo "                                    --out-dir benchs/figures_router"
