#!/bin/bash
#SBATCH --job-name=suco-distcount
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
# Distance evaluations per query at matched recall 
#
# Usage
#   sbatch benchs/distcount_hpc.sbatch                      # all 11 datasets
#   sbatch benchs/distcount_hpc.sbatch sift1m gist1m
#   INDEX_TYPES="cspg shg hnsw32 hnsw48 suco" sbatch benchs/distcount_hpc.sbatch
#   FORCE_BUILD=1 sbatch benchs/distcount_hpc.sbatch        # build inside the job
#   TARGETS="0.95" sbatch benchs/distcount_hpc.sbatch
#


set -uo pipefail
mkdir -p logs/slurm

DEFAULT_DATASETS=(sift1m sift10m gist1m deep1m deep10m spacev10m \
                  msong enron openai1m msturing10m uqv)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

INDEX_TYPES=${INDEX_TYPES:-"cspg shg hnsw32 hnsw48"}

TARGETS=${TARGETS:-"0.80 0.95 0.99"}
K=${K:-10}

FORCE_BUILD=${FORCE_BUILD:-0}
BUILD_JOBS=${BUILD_JOBS:-${SLURM_CPUS_PER_TASK}}
OPT_LEVEL=${OPT_LEVEL:-avx512}

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

module purge
module load profile/base
module load gcc/11.3.0
module load cmake 2>/dev/null || true

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
BUILD_DIR=$SLURM_SUBMIT_DIR/build_distcount
mkdir -p "$OUT_DIR"

echo "=== distcount run start $(date '+%F %T') ==="
echo "    datasets    : ${DATASETS[*]}"
echo "    index types : ${INDEX_TYPES}"
echo "    k           : ${K}"
echo "    targets     : ${TARGETS}"

probe_counters() {
    python - <<'PY'
import sys
try:
    import faiss
except Exception as e:
    print(f"import faiss failed: {e}")
    sys.exit(1)
missing = [n for n in ("cspg_stats_get", "shg_stats_get", "suco_stats_get")
           if not hasattr(faiss, n)]
cvar = getattr(faiss, "cvar", None)
if not hasattr(cvar, "hnsw_stats") and not hasattr(faiss, "hnsw_stats_get"):
    missing.append("hnsw_stats")
if missing:
    print("missing counters: " + ", ".join(missing))
    sys.exit(1)
print(f"counters present; faiss {faiss.__version__} from {faiss.__file__}")
PY
}

NEED_BUILD=0
if [ "$FORCE_BUILD" = "1" ]; then
    NEED_BUILD=1
    echo "step 1: FORCE_BUILD=1, building in-job"
else
    echo "step 1: probing the installed FAISS for distance counters"
    if ! probe_counters; then
        cat <<'EOF'

!!! The FAISS on PYTHONPATH has no CSPG/SHG/SuCo distance counters, so this
!!! job would produce HNSW rows and nothing else -- exactly the outcome of the
!!! M2b run (job 52451844, step 6). Aborting instead of wasting the slot.
!!!
!!! Rebuild and reinstall first, from the repo root on a login node:
!!!
!!!   module purge && module load profile/base gcc/11.3.0
!!!   source ~/miniconda3/bin/activate && conda activate anns
!!!   cmake --build build -j 8 --target faiss faiss_avx512 swigfaiss swigfaiss_avx512
!!!   ( cd build/faiss/python && pip install --force-reinstall --no-deps . )
!!!
!!! then verify:
!!!
!!!   python -c "import faiss; print(faiss.shg_stats_get().n_dis_full, \
!!!                                  faiss.suco_stats_get().n_dis_rerank, \
!!!                                  faiss.cspg_stats_get().n_dis)"
!!!   python benchs/check_suco_load.py --max-mb 1024
!!!
!!! and resubmit. FORCE_BUILD=1 makes this job configure a fresh build tree in
!!! build_distcount/ instead, but that recipe is a reconstruction and has not
!!! been validated on this cluster.
EOF
        exit 1
    fi
fi

if [ "$NEED_BUILD" = "1" ]; then
    echo ""
    echo "############ 1. building instrumented FAISS ############"
    echo "into $BUILD_DIR — the conda env's FAISS is left alone, so every"
    echo "timing number in the paper stays attributable to the build that"
    echo "produced it."

    # Same optimisation flags as the build that produced the timings. A
    # different -march would change the SIMD kernels' floating-point
    # association and with it the recall at a given parameter, which is
    # exactly what the recall-agreement check downstream would catch.
    cmake -B "$BUILD_DIR" -S "$SLURM_SUBMIT_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_ENABLE_C_API=OFF \
        -DFAISS_ENABLE_EXTRAS=OFF \
        -DFAISS_OPT_LEVEL="${OPT_LEVEL}" \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DPython_EXECUTABLE="$(which python)" \
        || { echo "!!! cmake configure FAILED !!!"; exit 1; }

    TARGETS_BUILD="faiss swigfaiss"
    case "$OPT_LEVEL" in
        avx512)      TARGETS_BUILD="$TARGETS_BUILD faiss_avx512 swigfaiss_avx512" ;;
        avx512_spr)  TARGETS_BUILD="$TARGETS_BUILD faiss_avx512_spr swigfaiss_avx512_spr" ;;
        avx2)        TARGETS_BUILD="$TARGETS_BUILD faiss_avx2 swigfaiss_avx2" ;;
    esac

    # shellcheck disable=SC2086
    cmake --build "$BUILD_DIR" -j "$BUILD_JOBS" --target $TARGETS_BUILD \
        || { echo "!!! FAISS build FAILED !!!"; exit 1; }

    ( cd "$BUILD_DIR/faiss/python" && python setup.py build ) \
        || { echo "!!! python package build FAILED !!!"; exit 1; }

    PYLIB=$(ls -d "$BUILD_DIR"/faiss/python/build/lib* 2>/dev/null | head -1)
    if [ -z "$PYLIB" ]; then
        echo "!!! no build/lib* under $BUILD_DIR/faiss/python !!!"
        exit 1
    fi
    export PYTHONPATH="$PYLIB:${PYTHONPATH:-}"
    echo "PYTHONPATH now leads with $PYLIB"

    probe_counters || { echo "!!! counters still absent after build !!!"; exit 1; }
fi

echo ""
echo "--- FAISS in use ---"
probe_counters
echo "--------------------"

python benchs/check_datasets.py --data-dir "$DATA_DIR" "${DATASETS[@]}" \
    || { echo "Dataset check failed - fix paths before running."; exit 1; }

# --- Step 2: the counting pass, one dataset per invocation ----------------
# Per-dataset invocations so that a dataset that fails to load, or an index
# missing from $INDEX_DIR, costs one dataset rather than the whole run. The
# table across datasets is assembled in step 3 from the JSONs on disk.
for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  distance counts: $DS"
    echo "############################################################"

    python benchs/bench_m2_distcount.py \
        --dataset      "$DS" \
        --data-dir     "$DATA_DIR" \
        --index-dir    "$INDEX_DIR" \
        --results-dir  "$OUT_DIR" \
        --out-dir      "$OUT_DIR" \
        --index-types  ${INDEX_TYPES} \
        --k            "$K" \
        --targets      ${TARGETS} \
        2>&1 | tee "$OUT_DIR/log_distcount_${DS}.txt" \
        || echo "!!! $DS distcount FAILED (continuing) !!!"
done

# --- Step 3: one table across every dataset that produced a JSON ----------
echo ""
echo "############ 3. summary table ############"
python benchs/bench_m2_distcount.py \
    --summarize --latex --latex-target 0.95 \
    --dataset "${DATASETS[@]}" \
    --out-dir "$OUT_DIR" \
    --targets ${TARGETS} \
    2>&1 | tee "$OUT_DIR/log_distcount_summary.txt"

echo ""
echo "=== distcount run done $(date '+%F %T') ==="
cat <<'EOF'

What to read out of this:

  m2_distcount_<dataset>.json   per index: the full count sweep, and at each
                                recall target the interpolated full-width
                                distance count, narrow-width count, float
                                traffic, and the ms/query from the timing run.
  m2_distcount_table.tex        the paper table, ready to \input.
  log_distcount_summary.txt     the ratio block: dis, float and QPS ratios of
                                each method against its own HNSW baseline.

The finding the paper needs is a row where the dis ratio is materially below 1
and the QPS ratio is not above it — a distance reduction that reproduced and
did not convert into time. Read the two together, per dataset:

  dis < 1, QPS < 1   the claim. Fewer evaluations, no faster, so the per-
                     distance path is not what bounds the query.
  dis ~ 1, QPS < 1   the reduction did not reproduce; the gap is the
                     reimplementation and the paper's reading has to change.
  dis < 1, QPS > 1   the reduction reproduced and did pay, at a smaller
                     factor than published; report the elasticity instead.

For SHG read the float ratio alongside the dis ratio: SHG's compressed upper
levels and its cross-level lower bound trade full-width evaluations for narrow
ones, so a dis ratio below 1 with a float ratio near 1 means the work moved
rather than disappeared. lb_pruned_per_query in the JSON is the count the
cross-level bound actually removed, which the SHG ablation predicts is small.

Check recall_agreement_vs_timing in each JSON before quoting anything: it is
the largest recall gap between this pass and the timing run at the same sweep
point, and the interpolated ratios are only comparable while it is small.
EOF
