#!/bin/bash
#SBATCH --job-name=suco-router
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

set -uo pipefail
mkdir -p logs/slurm

DEFAULT_DATASETS=(spacev10m sift10m msturing10m deep10m openai1m deep1m uqv sift1m gist1m msong enron)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# --- Environment (same as your bigann100m job) ---------------------------
module purge
module load profile/base
module load gcc/11.3.0
# (no CUDA; CPU-only FAISS)

source ~/miniconda3/bin/activate
conda activate suco

# Thread budgets - match what we asked Slurm for
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FAISS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

cd "$SLURM_SUBMIT_DIR"   # = your SuCo checkout

DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
mkdir -p "$INDEX_DIR" "$OUT_DIR"

# Fail fast (once) if any dataset's files are missing / mis-paired
python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

echo "=== router run start $(date '+%F %T')  |  NRUNS=$NRUNS  |  ${#DATASETS[@]} datasets ==="

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  dataset: $DS"
    echo "############################################################"

    python benchs/bench_router_paper.py \
        --dataset "$DS" --benchmark all --families all \
        --data-dir  "$DATA_DIR" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$OUT_DIR" \
        || echo "!!! $DS FAILED (continuing to next dataset) !!!"
done

echo ""
echo "=== router run done $(date '+%F %T') ==="
