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
#SBATCH --output=logs/slurm/%x-%A_%a.log
#SBATCH --error=logs/slurm/%x-%A_%a.err

# One array task per dataset (11 datasets => indices 0..10)
#SBATCH --array=0-10

set -euo pipefail
mkdir -p logs/slurm

# --- Dataset for this array task ---
DATASETS=(sift1m sift10m gist1m deep1m deep10m spacev10m msong enron openai1m msturing10m uqv)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# --- Environment (same as your bigann100m job) ---
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

echo "=== router: dataset=$DS  (array task $SLURM_ARRAY_TASK_ID) ==="

# Fail fast if this dataset's files are missing / mis-paired
python benchs/check_datasets.py --data-dir "$DATA_DIR" "$DS"

python benchs/bench_router_paper.py \
  --dataset "$DS" --benchmark all \
  --families all \
  --data-dir  "$DATA_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-dir "$OUT_DIR"