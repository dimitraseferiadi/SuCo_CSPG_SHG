/* benchs/m2_ceilings.c
 *
 * The two memory ceilings the M2 argument needs, neither of which the numpy
 * proxy in bench_m2_bandwidth.py measures correctly.
 *
 * A) STREAM.  McCalpin Copy/Scale/Add/Triad plus a pure-read reduction.
 *    Two reasons the numpy proxy is not a substitute.  It is a *copy*, and a
 *    copy counts 2n bytes while moving 3n unless the C library elects
 *    non-temporal stores, so the reported figure depends on a libc heuristic.
 *    And the graph search never writes: Read is the reference a read-only path
 *    must be measured against, not Copy.
 *
 * B) BW_rand(B, C).  Achievable bandwidth for the access pattern graph
 *    traversal actually has: chunks of B bytes read at random chunk-aligned
 *    offsets from an array far larger than LLC, C of them independent before
 *    the next C depend on bytes just read.  That is the shape of HNSW's
 *    base-layer loop -- fvec_L2sqr_batch_4 issues four independent vector
 *    reads per group, and the next hop cannot be chosen until the current
 *    group's distances are known.
 *
 *    B = 4d, so this yields one ceiling per dataset width and per padded
 *    width.  That is the denominator "% of ceiling" must use: a sequential
 *    STREAM number is the ceiling for a pattern the search only approaches at
 *    the widest padded vectors, where a 30 KB "random" access has become a
 *    7.5-page sequential burst.
 *
 *    C=1 is a pure pointer chase (memory-level parallelism 1); C=4 matches the
 *    batch-4 kernel; C=16 and C=64 bracket what out-of-order execution and the
 *    visited-table prefetch may add on top.  Reporting the sweep rather than a
 *    single C keeps the ceiling honest about an assumption the search does not
 *    let us observe directly.
 *
 * Offsets are chunk-index * B, exactly as FAISS lays out a flat node-major
 * array, so widths whose byte size is not a multiple of 64 (d=100, d=420)
 * carry their real line-splitting behaviour rather than an idealised one.
 *
 * Build
 *   gcc -O3 -march=native -fopenmp -o m2_ceilings benchs/m2_ceilings.c
 *
 * Run
 *   OMP_NUM_THREADS=32 ./m2_ceilings --json out/m2_ceilings.json
 *   ./m2_ceilings --gather-gib 16 --dims 96,128,960,7680
 */

#define _GNU_SOURCE
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* STREAM arrays: 3 x this many doubles.  256 Mi doubles = 2 GiB each, 6 GiB
 * total, ~128x the 48 MB LLC of the node these numbers are for. */
static size_t stream_n = 256UL * 1024 * 1024;
static int stream_reps = 10;

/* Random-gather array.  Sized so that even the largest chunk leaves enough
 * distinct chunks for the power-of-two mask below to cover most of it. */
static size_t gather_gib = 8;
static int gather_reps = 3;
static size_t gather_bytes_per_thread = 256UL * 1024 * 1024;

/* 4d for every width in the suite (96,100,128,200,256,420,512,960,1369,1536)
 * and every padded width the ablation uses (2d, 4d, 8d for 128 and 960). */
static int default_dims[] = {96,  100, 128,  200,  256,  420,  512, 960,
                             1024, 1369, 1536, 1920, 3840, 7680};
static int n_default_dims = sizeof(default_dims) / sizeof(int);

static int default_conc[] = {1, 4, 16, 64};
static int n_default_conc = 4;

/* ------------------------------------------------------------------ STREAM */

static double now(void) {
    return omp_get_wtime();
}

/* Somewhere the compiler cannot see through.  A reduction whose result is only
 * compared against a constant is dead code to an optimiser, and the whole loop
 * goes with it -- which shows up as a divide-by-zero-time "inf GB/s" rather
 * than as a wrong number, but only if you look. */
volatile double m2_sink = 0.0;

/* GB/s from bytes moved and elapsed time, refusing to report a figure from a
 * timer interval too short to mean anything. */
static double gbs(double bytes, double dt) {
    return (dt > 1e-6) ? bytes / dt / 1e9 : 0.0;
}

static void stream_bench(FILE *js) {
    double *a, *b, *c;
    size_t n = stream_n;
    if (posix_memalign((void **)&a, 4096, n * sizeof(double)) ||
        posix_memalign((void **)&b, 4096, n * sizeof(double)) ||
        posix_memalign((void **)&c, 4096, n * sizeof(double))) {
        fprintf(stderr, "STREAM allocation failed (%.1f GiB needed)\n",
                3.0 * n * sizeof(double) / (1024.0 * 1024 * 1024));
        exit(1);
    }

    /* First touch under the same schedule the kernels use, so pages land on
     * the socket that will read them. */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    /* The four McCalpin kernels only.  A read kernel does not belong here: a
     * `double` sum is a non-associative reduction, so the compiler must keep it
     * as one dependent add chain and the loop measures FP-add latency, not
     * bandwidth.  The read ceiling is measured in gather_bench() instead, as an
     * integer XOR that vectorises legally. */
    const double scalar = 3.0;
    double best[4];
    for (int j = 0; j < 4; j++) best[j] = 0.0;

    const double nb = (double)n * sizeof(double);
    for (int r = 0; r < stream_reps; r++) {
        double t0, g;

        /* Copy: 2n moved */
        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        g = gbs(2.0 * nb, now() - t0);
        if (g > best[0]) best[0] = g;

        /* Scale: 2n */
        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) b[i] = scalar * c[i];
        g = gbs(2.0 * nb, now() - t0);
        if (g > best[1]) best[1] = g;

        /* Add: 3n */
        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
        g = gbs(3.0 * nb, now() - t0);
        if (g > best[2]) best[2] = g;

        /* Triad: 3n */
        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
        g = gbs(3.0 * nb, now() - t0);
        if (g > best[3]) best[3] = g;
    }
    m2_sink += a[0] + b[0] + c[0];

    const char *names[4] = {"copy", "scale", "add", "triad"};
    printf("\nA) STREAM  (%d threads, %.1f GiB per array, best of %d)\n",
           omp_get_max_threads(),
           (double)n * sizeof(double) / (1024.0 * 1024 * 1024), stream_reps);
    for (int j = 0; j < 4; j++)
        printf("   %-6s %8.2f GB/s\n", names[j], best[j]);
    printf("   copy is what the numpy proxy in bench_m2_bandwidth.py stood in "
           "for;\n   the read ceiling a read-only search needs is in B).\n");

    fprintf(js, "  \"stream_gbs\": {");
    for (int j = 0; j < 4; j++)
        fprintf(js, "%s\"%s\": %.4f", j ? ", " : "", names[j], best[j]);
    fprintf(js, "},\n");
    fprintf(js, "  \"stream_array_bytes\": %zu,\n", n * sizeof(double));

    free(a);
    free(b);
    free(c);
}

/* ----------------------------------------------------- random chunk gather */

/* One hop: C chunk reads whose offsets are all derivable from `state` before
 * any load returns, so the C reads are mutually independent; the next hop's
 * state is mixed with bytes just read, so hops are dependent.  Eight XOR
 * accumulators keep the reduction off the critical path and let gcc vectorise
 * it without -ffast-math. */
static double gather_bench_one(const uint32_t *base, size_t mask, int words,
                               int conc, size_t hops, uint64_t seed) {
    double t0 = now();
    double bytes = 0.0;

#pragma omp parallel reduction(+ : bytes)
    {
        uint64_t state = seed * 0x9E3779B97F4A7C15ULL +
                         (uint64_t)omp_get_thread_num() * 0xBF58476D1CE4E5B9ULL;
        uint32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        size_t moved = 0;

        for (size_t h = 0; h < hops; h++) {
            uint64_t s = state;
            for (int cc = 0; cc < conc; cc++) {
                /* splitmix64 step: cheap, and off the load critical path */
                s += 0x9E3779B97F4A7C15ULL;
                uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                z ^= z >> 31;

                const uint32_t *p = base + (size_t)(z & mask) * (size_t)words;
                for (int w = 0; w < words; w += 8) {
                    int lim = (w + 8 <= words) ? 8 : words - w;
                    for (int q = 0; q < lim; q++) acc[q] ^= p[w + q];
                }
                moved += (size_t)words * 4;
            }
            /* the next hop depends on the bytes this hop read */
            state ^= ((uint64_t)acc[0] << 32) | acc[7];
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        }

        uint32_t fold = 0;
        for (int q = 0; q < 8; q++) fold ^= acc[q];
        /* consume `fold` so nothing is dead-code eliminated */
        bytes = (double)moved + (fold == 0xFFFFFFFFu ? 1e-9 : 0.0);
    }

    return gbs(bytes, now() - t0);
}

static void gather_bench(FILE *js, int *dims, int n_dims, int *conc,
                         int n_conc) {
    size_t nbytes = gather_gib * 1024UL * 1024 * 1024;
    uint32_t *arr;
    if (posix_memalign((void **)&arr, 4096, nbytes)) {
        fprintf(stderr, "gather allocation failed (%zu GiB)\n", gather_gib);
        exit(1);
    }
    size_t nwords = nbytes / 4;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nwords; i++) arr[i] = (uint32_t)(i * 2654435761u);

    printf("\nB) BW_rand(B, C)  (%d threads, %zu GiB array, best of %d)\n"
           "   B = 4d bytes per chunk at a random chunk-aligned offset;\n"
           "   C chunks independent per hop, hops dependent on bytes read.\n",
           omp_get_max_threads(), gather_gib, gather_reps);

    printf("\n   %6s %8s %8s", "d", "B(byte)", "lines");
    for (int c = 0; c < n_conc; c++) printf("   C=%-3d", conc[c]);
    printf("      (GB/s)\n");

    fprintf(js, "  \"gather_array_bytes\": %zu,\n", nbytes);
    fprintf(js, "  \"bw_rand_gbs\": [\n");

    for (int i = 0; i < n_dims; i++) {
        int d = dims[i];
        size_t B = (size_t)d * 4;
        size_t nchunks = nbytes / B;
        /* largest power of two <= nchunks, so the offset is a mask not a mod */
        size_t pow2 = 1;
        while (pow2 * 2 <= nchunks) pow2 *= 2;
        size_t mask = pow2 - 1;

        printf("   %6d %8zu %8.1f", d, B, (double)B / 64.0);
        fprintf(js, "    {\"d\": %d, \"chunk_bytes\": %zu, "
                    "\"chunks_addressed\": %zu, \"covered_bytes\": %zu",
                d, B, pow2, pow2 * B);

        for (int c = 0; c < n_conc; c++) {
            size_t hops = gather_bytes_per_thread / (B * (size_t)conc[c]);
            if (hops < 8) hops = 8;
            double best = 0.0;
            for (int r = 0; r < gather_reps; r++) {
                double g = gather_bench_one(arr, mask, d, conc[c], hops,
                                            0x243F6A8885A308D3ULL + r);
                if (g > best) best = g;
            }
            printf(" %7.2f", best);
            fprintf(js, ", \"C%d\": %.4f", conc[c], best);
        }
        printf("\n");
        fprintf(js, "}%s\n", (i + 1 < n_dims) ? "," : "");
    }
    fprintf(js, "  ],\n");

    /* Sequential read over the same array, as a calibration point against the
     * STREAM read figure above. */
    double s_best = 0.0;
    for (int r = 0; r < gather_reps; r++) {
        uint32_t g = 0;
        double t0 = now();
#pragma omp parallel for schedule(static) reduction(^ : g)
        for (size_t i = 0; i < nwords; i++) g ^= arr[i];
        double sg = gbs((double)nbytes, now() - t0) +
                    (g == 0xFFFFFFFFu ? 1e-9 : 0.0);
        if (sg > s_best) s_best = sg;
    }
    printf("\n   sequential read, same array, no writes: %7.2f GB/s\n"
           "   This is the read-only sequential ceiling: the upper bound the\n"
           "   search approaches only once a vector is wide enough that a\n"
           "   'random' access has become a multi-page burst.\n", s_best);
    fprintf(js, "  \"sequential_read_gbs\": %.4f,\n", s_best);

    free(arr);
}

/* -------------------------------------------------------------------- main */

static int parse_int_list(const char *s, int *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        out[n++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return n;
}

int main(int argc, char **argv) {
    const char *json_path = NULL;
    int dims[64], conc[16];
    int n_dims = 0, n_conc = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_path = argv[++i];
        else if (!strcmp(argv[i], "--dims") && i + 1 < argc)
            n_dims = parse_int_list(argv[++i], dims, 64);
        else if (!strcmp(argv[i], "--conc") && i + 1 < argc)
            n_conc = parse_int_list(argv[++i], conc, 16);
        else if (!strcmp(argv[i], "--gather-gib") && i + 1 < argc)
            gather_gib = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--stream-gib") && i + 1 < argc)
            stream_n = (size_t)atoll(argv[++i]) * 1024 * 1024 * 1024 /
                       sizeof(double);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            gather_reps = stream_reps = atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [--json f] [--dims 96,128,...] [--conc 1,4,16]\n"
                    "          [--gather-gib N] [--stream-gib N] [--reps N]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!n_dims) {
        memcpy(dims, default_dims, sizeof(default_dims));
        n_dims = n_default_dims;
    }
    if (!n_conc) {
        memcpy(conc, default_conc, sizeof(default_conc));
        n_conc = n_default_conc;
    }

    FILE *js = json_path ? fopen(json_path, "w") : fopen("/dev/null", "w");
    if (!js) {
        fprintf(stderr, "cannot write %s\n", json_path);
        return 1;
    }
    fprintf(js, "{\n  \"threads\": %d,\n", omp_get_max_threads());

    printf("m2_ceilings: threads=%d\n", omp_get_max_threads());
    stream_bench(js);
    gather_bench(js, dims, n_dims, conc, n_conc);

    fprintf(js, "  \"note\": \"bw_rand is the denominator for a dependent "
                "random-read pattern; stream.read is the sequential upper "
                "bound for a read-only path\"\n}\n");
    fclose(js);
    if (json_path) printf("\nwrote %s\n", json_path);
    return 0;
}
