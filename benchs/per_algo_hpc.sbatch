#!/bin/bash
#SBATCH --job-name=suco-per-algo
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
# Per-algorithm reproductions (SuCo / CSPG / SHG), each vs its own HNSW
# (+ LSH for SuCo). Reuses the indexes already built by bench_router_paper.py
# (the cross-algorithm run) via symlinks — nothing is rebuilt except SuCo's own
# HNSW (efC200, by design) and the fresh LSH indexes SuCo adds.
#
# Usage:
#   sbatch benchs/per_algo_hpc.sbatch                 # all three
#   sbatch benchs/per_algo_hpc.sbatch cspg shg        # subset
#   SUCO_SWEEP=1 sbatch benchs/per_algo_hpc.sbatch suco   # + SuCo curve (rebuilds SuCo variants)

set -uo pipefail
mkdir -p logs/slurm

# Which algorithms to run (default: all three).
DEFAULT_ALGOS=(suco cspg shg)
if [ "$#" -gt 0 ]; then
    ALGOS=("$@")
else
    ALGOS=("${DEFAULT_ALGOS[@]}")
fi

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

# --- Environment (same as the router job) --------------------------------
module purge
module load profile/base
module load gcc/11.3.0
# (no CUDA; CPU-only FAISS)

source ~/miniconda3/bin/activate
conda activate anns

# Thread budgets - match what we asked Slurm for (THE fix for "runs on 1 thread")
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FAISS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

cd "$SLURM_SUBMIT_DIR"   # = your SuCo checkout

# --- Paths ---------------------------------------------------------------
DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices          # router's prebuilt indexes live here
CSPG_IDX=$WORK/dhm/indices_cspg      # symlink views onto $INDEX_DIR
SHG_IDX=$WORK/dhm/indices_shg
SUCO_IDX=$WORK/dhm/indices_suco
mkdir -p "$CSPG_IDX" "$SHG_IDX" "$SUCO_IDX"

SUCO_OUT=$SLURM_SUBMIT_DIR/benchs/results_suco_repro
CSPG_OUT=$SLURM_SUBMIT_DIR/benchs/results_cspg
SHG_OUT=$SLURM_SUBMIT_DIR/benchs/results_shg
mkdir -p "$SUCO_OUT" "$CSPG_OUT" "$SHG_OUT"

# Set SUCO_SWEEP=1 to also trace the SuCo recall/QPS curve. The reused router
# index only yields SuCo's default operating point; the curve needs --sweep,
# which REBUILDS SuCo parameter variants (SuCo-only collision indexes, not the
# shared cross-algo indexes).
SUCO_SWEEP=${SUCO_SWEEP:-0}
SUCO_EXTRA=""
[ "$SUCO_SWEEP" = "1" ] && SUCO_EXTRA="--sweep"

want() { for a in "${ALGOS[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }

link_or_warn() {  # link_or_warn <router-target> <expected-linkname>
    if [ -e "$1" ]; then
        ln -sf "$1" "$2"
    else
        echo "  !! MISSING router index: $1  ->  would be REBUILT for $(basename "$2")"
    fi
}

echo "=== per-algo run start $(date '+%F %T')  |  algos: ${ALGOS[*]}  |  threads=${OMP_NUM_THREADS} ==="

# =========================================================================
# 1. Bridge router index filenames -> per-algorithm script filenames
# =========================================================================
if want cspg; then
    echo "--- linking CSPG indexes (cspg + hnsw32) ---"
    # cspg-name -> router-name  (router used 'uqv', CSPG script uses 'uqv1m')
    declare -A CSPG_MAP=( [sift1m]=sift1m [deep1m]=deep1m [gist1m]=gist1m \
                          [uqv1m]=uqv [openai1m]=openai1m [sift10m]=sift10m )
    for name in "${!CSPG_MAP[@]}"; do r=${CSPG_MAP[$name]}
        link_or_warn "$INDEX_DIR/${r}_cspg.idx"   "$CSPG_IDX/${name}_cspg_default.idx"
        link_or_warn "$INDEX_DIR/${r}_hnsw32.idx" "$CSPG_IDX/${name}_hnsw.idx"
    done
fi

if want shg; then
    echo "--- linking SHG indexes (shg + hnsw48) ---"
    # shg-name -> router-name  (router used 'openai1m', SHG script uses 'openai')
    declare -A SHG_MAP=( [openai]=openai1m [enron]=enron [gist1m]=gist1m \
                         [msong]=msong [uqv]=uqv [msturing10m]=msturing10m )
    for name in "${!SHG_MAP[@]}"; do r=${SHG_MAP[$name]}
        link_or_warn "$INDEX_DIR/${r}_shg.idx"    "$SHG_IDX/${name}_shg.idx"
        link_or_warn "$INDEX_DIR/${r}_hnsw48.idx" "$SHG_IDX/${name}_hnsw.idx"
    done
fi

if want suco; then
    echo "--- preparing SuCo indexes (convert router format -> native SuCo stream) ---"
    # bench_suco_*.py load via IndexSuCo.read_index(), which needs the NATIVE
    # SuCo stream. The router saved them with faiss.write_index(), which prepends
    # a 4-byte "IxSC" container tag -> instance read fails with "bad magic". So
    # convert once into $SUCO_IDX as real (native) files. (HNSW/LSH build fresh.)
    declare -A SUCO_SRC=( [sift1m]=sift1m_suco_ns8  [gist1m]=gist1m_suco_ns40 \
        [deep_nb1000000]=deep1m_suco_ns8  [deep_nb10000000]=deep10m_suco_ns8 \
        [sift10m]=sift10m_suco_ns8  [spacev10m]=spacev10m_suco_ns10 )
    for name in "${!SUCO_SRC[@]}"; do
        src="$INDEX_DIR/${SUCO_SRC[$name]}.idx"
        dst="$SUCO_IDX/${name}.idx"
        [ -L "$dst" ] && rm -f "$dst"                 # drop stale symlink from older runs
        if [ -f "$dst" ]; then
            echo "  native SuCo already present: $dst"
        elif [ -e "$src" ]; then
            python benchs/convert_suco_index.py "$src" "$dst" \
                || echo "  !! SuCo convert FAILED: $src"
        else
            echo "  !! MISSING router SuCo index: $src"
        fi
    done
fi

# =========================================================================
# 2. SuCo  —  SuCo (reused) + HNSW (efC200, fresh) + LSH (fresh)
# =========================================================================
if want suco; then
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  SuCo reproduction  (sweep=${SUCO_SWEEP})"
    echo "############################################################"
    python benchs/check_datasets.py --data-dir "$DATA_DIR" --suite suco \
        || echo "  (suco dataset check warning — continuing)"

    python benchs/bench_suco_sift1m.py --data-dir "$DATA_DIR" \
        --index-path "$SUCO_IDX/sift1m.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_sift1m.txt" || echo "!!! suco sift1m FAILED"

    python benchs/bench_suco_gist1m.py --data-dir "$DATA_DIR" \
        --index-path "$SUCO_IDX/gist1m.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_gist1m.txt" || echo "!!! suco gist1m FAILED"

    python benchs/bench_suco_deep1b.py --data-dir "$DATA_DIR" --nb 1000000 \
        --index-path "$SUCO_IDX/deep_nb1000000.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_deep1m.txt" || echo "!!! suco deep1m FAILED"

    python benchs/bench_suco_deep1b.py --data-dir "$DATA_DIR" --nb 10000000 \
        --index-path "$SUCO_IDX/deep_nb10000000.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_deep10m.txt" || echo "!!! suco deep10m FAILED"

    python benchs/bench_suco_sift10m.py --data-dir "$DATA_DIR" \
        --gt-path "$INDEX_DIR/sift10m_gt.npy" \
        --index-path "$SUCO_IDX/sift10m.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_sift10m.txt" || echo "!!! suco sift10m FAILED"

    python benchs/bench_suco_spacev10m.py --data-dir "$DATA_DIR" \
        --gt-path "$INDEX_DIR/spacev10m_gt.npy" \
        --index-path "$SUCO_IDX/spacev10m.idx" --hnsw-sweep --lsh-sweep $SUCO_EXTRA \
        2>&1 | tee "$SUCO_OUT/log_suco_spacev10m.txt" || echo "!!! suco spacev10m FAILED"
fi

# =========================================================================
# 3. CSPG  —  CSPG + HNSW  (both reused)
# =========================================================================
if want cspg; then
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  CSPG reproduction"
    echo "############################################################"
    python benchs/check_datasets.py --data-dir "$DATA_DIR" --suite cspg \
        || echo "  (cspg dataset check warning — continuing)"

    for DS in sift1m deep1m gist1m uqv1m openai1m sift10m; do
        echo "--- CSPG: $DS ---"
        python benchs/bench_cspg_paper.py \
            --data-dir "$DATA_DIR" --index-dir "$CSPG_IDX" --output-dir "$CSPG_OUT" \
            --dataset "$DS" --benchmark all --index-type cspg hnsw \
            2>&1 | tee "$CSPG_OUT/log_cspg_${DS}.txt" || echo "!!! cspg $DS FAILED"
    done
fi

# =========================================================================
# 4. SHG  —  SHG + HNSW  (both reused; needs --index-type patch)
# =========================================================================
if want shg; then
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  SHG reproduction"
    echo "############################################################"
    python benchs/check_datasets.py --data-dir "$DATA_DIR" --suite shg \
        || echo "  (shg dataset check warning — continuing)"

    for DS in openai enron gist1m msong uqv msturing10m; do
        echo "--- SHG: $DS ---"
        python benchs/bench_shg_paper.py \
            --data-dir "$DATA_DIR" --index-dir "$SHG_IDX" --output-dir "$SHG_OUT" \
            --dataset "$DS" --benchmark all --index-type shg hnsw \
            2>&1 | tee "$SHG_OUT/log_shg_${DS}.txt" || echo "!!! shg $DS FAILED"
    done
fi

echo ""
echo "=== per-algo run done $(date '+%F %T') ==="

