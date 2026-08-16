#!/bin/bash
#SBATCH --job-name=suco-m2b
#SBATCH --account=EUHPC_D32_102
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.err
#
# M2, second pass.  The first pass (m2_hpc.sbatch) established that the padding
# ablation is sound -- the transplanted graph holds recall, the returned
# neighbour sets and the distance count identical at every width -- and then
# drew three conclusions the measurement does not support:
#
#   1. It read gamma < 1 as "query time is set by bytes read".  The affine fit
#      says 46% of SIFT1M's query time at d=128 is d-independent.  Worse, a
#      fixed per-query cost and an achievable bandwidth that rises with the
#      width of each read are the same function -- t = c*d/BW(d) with
#      BW(d) = BW_inf*d/(d+d0) IS t = a + b*d -- so padding cannot separate
#      them and gamma settles nothing either way.  What padding does identify
#      is the slope: marginal bandwidth, 57 GB/s at SIFT's native segment
#      rising to 80 GB/s on GIST.  Step 2 now reports it.
#
#   2. It quoted "80% of ceiling" from d=7680, where a vector is 30720 B = 480
#      cache lines = 7.5 pages and the "random" access has become a sequential
#      burst.  Reaching a sequential ceiling there is close to tautological,
#      and no benchmark in the suite exceeds d=1536.  Step 1 measures
#      BW_rand(4d) per width so each achieved figure has the right denominator.
#
#   3. It used the ceiling to explain why distance-count reductions do not
#      convert into speed.  That is d(log t)/d(log ndis); padding measures
#      d(log t)/d(log d).  A bandwidth-bound path predicts the first is 1 --
#      fewer evaluations read proportionally fewer bytes -- so the argument
#      runs the wrong way.  Step 4 measures it, and step 6 finally gets the
#      distance counts themselves, which the previous run never produced (it
#      died on the SuCo d % nsubspaces assertion at its first dataset).
#
# Steps
#   0  machine facts:            lscpu, DIMM population, perf availability
#   1  m2_ceilings.c:            real STREAM + BW_rand(B, C) per width
#   2  padding ablation:         unchanged, now with the affine/marginal fit
#   3  thread sweep:             does achieved bandwidth saturate in threads?
#   4  efSearch sweep:           d(log t)/d(log ndis), the elasticity in use
#   5  perf:                     measured DRAM traffic vs the inferred figure
#   6  distcount:                distance evaluations per query at matched recall
#
# Steps 1-4 are the ones the paper cannot go to review without.  Step 5 is
# best-effort: uncore counters usually need privileges a shared cluster does
# not grant, and the run continues without them.
#
# Usage
#   sbatch benchs/m2b_hpc.sbatch                    # sift1m + gist1m
#   sbatch benchs/m2b_hpc.sbatch sift1m gist1m openai1m
#   SKIP_PERF=1 SKIP_DISTCOUNT=1 sbatch benchs/m2b_hpc.sbatch

set -uo pipefail
mkdir -p logs/slurm

DEFAULT_DATASETS=(sift1m gist1m)
if [ "$#" -gt 0 ]; then DATASETS=("$@"); else DATASETS=("${DEFAULT_DATASETS[@]}"); fi

SKIP_CEILINGS=${SKIP_CEILINGS:-0}
SKIP_PADDING=${SKIP_PADDING:-0}
SKIP_THREADS=${SKIP_THREADS:-0}
SKIP_EFSWEEP=${SKIP_EFSWEEP:-0}
SKIP_PERF=${SKIP_PERF:-0}
SKIP_DISTCOUNT=${SKIP_DISTCOUNT:-0}

EF_SEARCH=${EF_SEARCH:-100}
THREAD_LIST=${THREAD_LIST:-"1 2 4 8 16 32"}
EF_LIST=${EF_LIST:-"10 20 40 60 80 100 150 200 300 400 600 800 1000 1500 2000"}
GATHER_GIB=${GATHER_GIB:-16}
STREAM_GIB=${STREAM_GIB:-4}

# CRITICAL: clear positional args before sourcing conda (see router_hpc.sbatch).
set --

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

# No OMP_PROC_BIND / OMP_PLACES: achieved bandwidth is meaningless if the team
# is confined to one core, and the main results were taken without it.

cd "$SLURM_SUBMIT_DIR"
DATA_DIR=$WORK/dhm/data
INDEX_DIR=$WORK/dhm/indices
OUT_DIR=$SLURM_SUBMIT_DIR/benchs/results_router
SCRATCH=${TMPDIR:-/tmp}/m2b_$$
mkdir -p "$OUT_DIR" "$SCRATCH"

echo "=== M2b run start $(date '+%F %T') ==="
echo "    datasets : ${DATASETS[*]}"
echo "    threads  : ${SLURM_CPUS_PER_TASK}"

# --- Pre-flight: will step 6 load SuCo, or die the way it did last time? ---
# Checked here, at the top, rather than discovered eight hours in. Steps 1-5
# never touch SuCo, so a failure disables step 6 alone instead of the job.
if [ "$SKIP_DISTCOUNT" != "1" ]; then
    echo ""
    echo "############ pre-flight: SuCo load ############"
    python benchs/check_suco_load.py \
        --index-dir "$WORK/dhm/indices" --datasets "${DATASETS[@]}" \
        2>&1 | tee "$OUT_DIR/log_m2_preflight.txt"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!!! SuCo indexes will not load; step 6 disabled for this run. !!!"
        echo "!!! Steps 1-5 are unaffected and continue.                    !!!"
        SKIP_DISTCOUNT=1
    fi
fi

# --- Step 0: machine facts ------------------------------------------------
# The ceiling is a property of this node.  Record what the node actually is, so
# a reviewer asking "204.8 GB/s theoretical, why is STREAM half that?" has the
# DIMM population in the artifact rather than in a reply.
echo ""
echo "############ 0. machine ############"
{
    lscpu
    echo "--- caches ---"; lscpu -C 2>/dev/null || true
    echo "--- numa ---"; numactl -H 2>/dev/null || echo "numactl unavailable"
    echo "--- memory ---"; free -g
    echo "--- DIMMs (needs privileges; absence is itself worth recording) ---"
    dmidecode -t memory 2>/dev/null | grep -E "Size|Speed|Locator" || \
        echo "dmidecode unavailable"
    echo "--- perf ---"
    echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
    command -v perf >/dev/null && perf --version || echo "perf not on PATH"
    echo "--- affinity ---"
    python -c "import os; print('sched_getaffinity:', len(os.sched_getaffinity(0)), 'CPUs')"
} 2>&1 | tee "$OUT_DIR/m2_machine.txt"

# --- Step 1: the ceilings -------------------------------------------------
# Real STREAM, and BW_rand(B, C) at B = 4d for every width in the suite and
# every padded width.  This is the denominator; everything downstream is a
# fraction of it.
if [ "$SKIP_CEILINGS" != "1" ]; then
    echo ""
    echo "############ 1. ceilings (STREAM + BW_rand) ############"
    gcc -O3 -march=native -fopenmp -o "$SCRATCH/m2_ceilings" \
        benchs/m2_ceilings.c -lm \
        && OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} "$SCRATCH/m2_ceilings" \
            --gather-gib ${GATHER_GIB} --stream-gib ${STREAM_GIB} \
            --json "$OUT_DIR/m2_ceilings.json" \
            2>&1 | tee "$OUT_DIR/log_m2_ceilings.txt" \
        || echo "!!! ceilings FAILED (continuing) !!!"

    # Single thread too: it separates a per-core limit from a shared one, and
    # the ratio between them is what step 3 then confirms against the search.
    OMP_NUM_THREADS=1 "$SCRATCH/m2_ceilings" \
        --gather-gib ${GATHER_GIB} --stream-gib ${STREAM_GIB} \
        --json "$OUT_DIR/m2_ceilings_1t.json" \
        2>&1 | tee "$OUT_DIR/log_m2_ceilings_1t.txt" \
        || echo "!!! 1-thread ceilings FAILED (continuing) !!!"
fi

# --- Steps 2-4: per dataset ----------------------------------------------
for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $(date '+%F %T')  $DS"
    echo "############################################################"

    ARGS=(--dataset "$DS" --data-dir "$DATA_DIR" --index-dir "$INDEX_DIR"
          --out-dir "$OUT_DIR" --ef ${EF_SEARCH}
          --threads ${SLURM_CPUS_PER_TASK} --skip-stream)

    # The numpy copy proxy is skipped throughout: step 1 measured the real
    # thing, and leaving the proxy in the output invites it back into the paper.
    [ "$SKIP_PADDING" = "1" ] && ARGS+=(--skip-padding)
    [ "$SKIP_THREADS" != "1" ] && ARGS+=(--thread-sweep ${THREAD_LIST})
    [ "$SKIP_EFSWEEP" != "1" ] && ARGS+=(--ef-sweep ${EF_LIST})

    python benchs/bench_m2_bandwidth.py "${ARGS[@]}" \
        2>&1 | tee "$OUT_DIR/log_m2b_${DS}.txt" \
        || echo "!!! $DS bandwidth FAILED (continuing) !!!"
done

# --- Step 5: measured traffic --------------------------------------------
# Two invocations per width, differenced, so build and start-up cancel.  Uncore
# first; core PMU if the cluster will not grant it; skip if neither works.
if [ "$SKIP_PERF" != "1" ] && command -v perf >/dev/null 2>&1; then
    echo ""
    echo "############ 5. measured DRAM traffic ############"

    # A probe must fail closed.  `grep -qv` succeeds as soon as any single line
    # of perf's output does not match, which is nearly always, so it would
    # report every event as available and step 5 would emit <not supported>
    # rows that look like measurements.
    probe_events() {
        local out
        out=$(perf stat $2 -e "$1" -x, -- true 2>&1) || return 1
        echo "$out" | grep -qiE "not supported|no permission|invalid event|not found|<not counted>" && return 1
        return 0
    }

    EVENTS=""; PERF_MODE=""
    # Uncore PMUs are system-wide only, so -a is required, and -a needs
    # perf_event_paranoid <= 0 or CAP_PERFMON.  These count every CAS the
    # memory controllers issue: the actual traffic, prefetch included.
    if probe_events "uncore_imc/cas_count_read/,uncore_imc/cas_count_write/" "-a"; then
        EVENTS="uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"
        PERF_MODE="-a"
        echo "uncore IMC counters available (x64 B per CAS): true DRAM traffic."
        echo "NOTE: -a is system-wide, so run this on an exclusive node or the"
        echo "      counts include whatever else is resident."
    # Core-PMU fallback: works unprivileged, but retired-load misses exclude
    # hardware prefetch, so treat the result as a LOWER BOUND on traffic. It is
    # still decisive for a 2x question, which is what is being asked here.
    elif probe_events "mem_load_retired.l3_miss" ""; then
        EVENTS="mem_load_retired.l3_miss,LLC-load-misses,LLC-store-misses"
        echo "uncore unavailable; core-PMU L3 misses (x64 B) as a LOWER BOUND"
        echo "      (excludes hardware-prefetched lines)"
    elif probe_events "cache-misses" ""; then
        EVENTS="cache-misses,cache-references"
        echo "using generic cache-misses (x64 B) as a coarse LOWER BOUND"
    else
        echo "no usable perf events; skipping step 5"
    fi

    if [ -n "$EVENTS" ]; then
        for DS in "${DATASETS[@]}"; do
            case "$DS" in
                sift1m) WIDTHS="128 256 512 1024" ;;
                gist1m) WIDTHS="960 1920" ;;
                *)      WIDTHS="" ;;
            esac
            GC="$SCRATCH/graph_${DS}.npz"
            for W in $WIDTHS; do
                for P in 1 11; do
                    echo "--- perf $DS d=$W passes=$P ---"
                    perf stat $PERF_MODE -x, -e "$EVENTS" \
                        -o "$OUT_DIR/m2_perf_${DS}_d${W}_p${P}.csv" \
                        -- python benchs/bench_m2_perf.py \
                            --dataset "$DS" --data-dir "$DATA_DIR" \
                            --index-dir "$INDEX_DIR" --out-dir "$OUT_DIR" \
                            --pad-dim "$W" --passes "$P" --ef ${EF_SEARCH} \
                            --threads ${SLURM_CPUS_PER_TASK} \
                            --graph-cache "$GC" \
                        2>&1 | tail -5
                done
            done
            rm -f "$GC"
        done
        echo "traffic per pass = (counts at passes=11 - counts at passes=1) / 10, x 64 B"
        echo "compare against ndis * nq * d * 4 from m2_padding_<ds>.csv:"
        echo "  the gap is neighbour lists and the visited table, and it is"
        echo "  d-independent, so it inflates the apparent bandwidth rise with d"
    fi
else
    echo "step 5 skipped (SKIP_PERF=$SKIP_PERF, perf present: $(command -v perf >/dev/null && echo yes || echo no))"
fi

# --- Step 6: distance counts ---------------------------------------------
# The quantity CSPG and SHG claim to reduce, and the one the previous run never
# reached.  Without it, Sec. VII's headline has no measurement behind it.
if [ "$SKIP_DISTCOUNT" != "1" ]; then
    echo ""
    echo "############ 6. distance counts ############"
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

rm -rf "$SCRATCH"

echo ""
echo "=== M2b run done $(date '+%F %T') ==="
cat <<'EOF'

What to read out of this, in the order the section needs it:

  m2_ceilings.json          BW_rand(4d, C=4) per width. THE denominator.
                            Also real STREAM triad, to replace the 97.0 GB/s
                            numpy copy figure, and the sequential read ceiling,
                            which is the right sequential reference for a path
                            that never writes.

  m2_padding_<ds>.csv       unchanged ablation; the invariance columns are what
  m2_bandwidth_<ds>.json    make it worth anything. In `affine`:
                              marginal_gbs_global and segments[]  -> the
                                bandwidth-bound quantity, to be divided by
                                BW_rand(4d), NOT by STREAM
                              fixed_fraction_at_d                 -> how much of
                                query time never was about bytes

  m2_threadsweep_<ds>.csv   achieved_gbs vs threads at fixed width. Flat tail =>
                            a shared-resource ceiling, and "% of ceiling" is
                            licensed. Still climbing at 32 => the limit is
                            memory-level parallelism, and the section must say
                            so instead.

  m2_efsweep_<ds>.csv       d(log t)/d(log ndis). If this is ~1, Sec. VII's
                            "fewer distances leave the wall-clock largely
                            unchanged" is false and the shortfalls belong to
                            CSPG's m=2 partitions and SHG's upper-level share --
                            both of which the paper already measures.

  m2_perf_*.csv             measured traffic vs the vector-bytes-only figure.
                            The gap is neighbour lists and the visited table.

  m2_distcount_<ds>.json    the reductions themselves, at matched recall.
EOF
