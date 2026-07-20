/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
#pragma once

/**
 * IndexSHG: Shortcut-enabled Hierarchical Graph index for FAISS.
 *
 * Implements the SHG/HEDS algorithm from:
 *   Gong, Zeng, Chen. "Accelerating Approximate Nearest Neighbor Search in
 *   Hierarchical Graphs: Efficient Level Navigation with Shortcuts."
 *   PVLDB 18(10): 3518-3530, 2025.
 *
 * Two core innovations over HNSW:
 *   1. Hierarchical vector compression (progressive mean aggregation, eta=2).
 *      Upper-level distances use compressed low-dimensional representations,
 *      cutting per-level computation cost.  Compression levels are computed
 *      independently of the HNSW graph levels: maxFixLevel_ is determined by
 *      repeatedly dividing d by eta until (dim/eta < eta), and each HNSW graph
 *      level l uses compressed level min(l, maxFixLevel_).
 *   2. Learned shortcut: a piecewise linear model mapping (approximate
 *      distance -> skip count), fitted with the PGM-index over the training
 *      samples.  Given the approximate distance between the query and the
 *      current entry point, the shortcut returns the number of HNSW levels
 *      that can safely be skipped.  Training samples come from kNN density
 *      estimation (Lemma 2 in the paper).
 *
 * Usage
 * -----
 *   // Build:
 *   IndexSHG idx(d, M);
 *   idx.add(n, data);        // HNSW graph with compressed upper-level distances
 *   idx.build_shortcut();    // one-time: compress vectors + train shortcut
 *
 *   // Search (uses shortcuts automatically):
 *   idx.search(nq, queries, k, distances, labels);
 */

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/pgm/pgm_index.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace faiss {

// ---------------------------------------------------------------------------
// ShortcutMap — the learned shortcut f(dis) of Definition 4
// ---------------------------------------------------------------------------

/**
 * Learned shortcut mapping approximate-distance -> skip count.
 *
 * Definition 4 of the paper trains a set of piecewise linear functions
 *   f(dis) = {(dis_1, slope_1, intercept_1), (dis_2, slope_2, intercept_2), ...}
 * over the distance-level tuples S = {(dis_0, h_0), (dis_1, h_1), ...} produced
 * by Algorithm 2 lines 13-18.  Queries evaluate f via lower_bound(dis).
 *
 * We back this with the PGM-index, as the SHG authors do (their heds.h holds a
 * `pgm::DynamicPGMIndex<dist_t, int> Shortcuts`).  The PGM-index *is* a set of
 * epsilon-bounded piecewise linear models over the sorted keys, which is
 * exactly the f(dis) of Definition 4 — and it delivers the space-efficiency
 * the paper argues for in Section 4.1: a few hundred bytes of model replaces
 * the ~48 bytes/entry of red-black tree nodes a std::map would need.
 *
 * We use the *static* pgm::PGMIndex rather than the dynamic variant: the
 * shortcut is trained once by IndexSHG::build_shortcut() and is read-only for
 * the lifetime of the index, so the dynamic variant's insert path is dead
 * weight.  Training is therefore two-phase — insert_or_assign() accumulates
 * samples, finalize() sorts them and fits the model.
 *
 * Layout after finalize(): keys[] is sorted and deduplicated, values[i] is the
 * skip count for keys[i], and pgm indexes keys[].
 */
struct ShortcutMap {
    /// Epsilon for the piecewise linear fit: search narrows to a window of
    /// 2*Epsilon+1 keys. Matches the DynamicPGMIndex default the authors use.
    static constexpr size_t pgm_epsilon = 16;

    using pgm_index_t = pgm::PGMIndex<float, pgm_epsilon>;

    std::vector<float> keys;   ///< approximate distances (sorted after finalize)
    std::vector<int> values;   ///< skip counts, parallel to keys
    pgm_index_t pgm;           ///< piecewise linear model over keys
    bool trained = false;      ///< true once finalize() has fitted the model

    /// Record a (distance -> skip) training sample.  Samples may arrive in any
    /// order and may repeat keys; finalize() resolves both.  Invalidates the
    /// model, so finalize() must be called before predict().
    void insert_or_assign(float dist, int skip) {
        keys.push_back(dist);
        values.push_back(skip);
        trained = false;
    }

    /// Sort and deduplicate the samples, then fit the piecewise linear model.
    /// For repeated keys the last-inserted value wins, matching the
    /// std::map::operator[] semantics this replaces.
    void finalize() {
        size_t n = keys.size();
        if (n > 0) {
            std::vector<uint32_t> ord(n);
            for (size_t i = 0; i < n; ++i) {
                ord[i] = (uint32_t)i;
            }
            // Sort by key, breaking ties by insertion order so that the last
            // sample for a duplicate key is the one we keep below.
            std::sort(ord.begin(), ord.end(), [this](uint32_t a, uint32_t b) {
                return keys[a] != keys[b] ? keys[a] < keys[b] : a < b;
            });

            std::vector<float> sorted_keys;
            std::vector<int> sorted_values;
            sorted_keys.reserve(n);
            sorted_values.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                float k = keys[ord[i]];
                int v = values[ord[i]];
                if (!sorted_keys.empty() && sorted_keys.back() == k) {
                    sorted_values.back() = v; // later sample wins
                } else {
                    sorted_keys.push_back(k);
                    sorted_values.push_back(v);
                }
            }
            keys.swap(sorted_keys);
            values.swap(sorted_values);
        }
        build_model();
    }

    /// (Re)fit the model over keys[], which must already be sorted and unique.
    /// Used by finalize() and when deserializing an index.
    void build_model() {
        pgm = keys.empty() ? pgm_index_t()
                           : pgm_index_t(keys.begin(), keys.end());
        trained = !keys.empty();
    }

    /// Return the predicted skip count for an approximate distance, i.e. the
    /// value of the first sample whose distance is >= dist.  Returns 1 (skip a
    /// single level, the HNSW default) when dist is past the last sample.
    int predict(float dist) const {
        if (!trained) return 1;
        // The model narrows the search to a window of 2*Epsilon+1 keys that is
        // guaranteed to bracket the true lower_bound.
        auto range = pgm.search(dist);
        size_t hi = std::min(range.hi, keys.size());
        auto it = std::lower_bound(
                keys.begin() + range.lo, keys.begin() + hi, dist);
        size_t pos = (size_t)(it - keys.begin());
        if (pos >= keys.size()) return 1;
        return values[pos];
    }

    bool is_trained() const {
        return trained;
    }

    int size() const {
        return (int)keys.size();
    }

    void clear() {
        keys.clear();
        values.clear();
        pgm = pgm_index_t();
        trained = false;
    }

    /// Bytes held by the fitted model, excluding the keys/values samples.
    size_t model_size_in_bytes() const {
        return trained ? pgm.size_in_bytes() : 0;
    }
};

// ---------------------------------------------------------------------------
// Search parameters
// ---------------------------------------------------------------------------

struct SearchParametersSHG : SearchParametersHNSW {
    /// When true, use the shortcut for level skipping.
    bool use_shortcut = true;

    /// When true, apply compressed-vector lower-bound pruning at base level.
    bool use_lb_pruning = true;
};

// ---------------------------------------------------------------------------
// IndexSHG
// ---------------------------------------------------------------------------

/**
 * HNSW-based index augmented with hierarchical vector compression and a
 * learned shortcut for level navigation.
 *
 * Inherits from IndexHNSWFlat: the flat vector storage is used for exact
 * distance computation at the base level.  All upper-level distances use
 * the compressed representations stored in compressed_vecs.
 *
 * The compression hierarchy is independent of the HNSW levels:
 *   - maxFixLevel_ compression levels are computed from d and eta (=2),
 *     where each level l has dimension ceil(d / eta^l).
 *   - HNSW level l uses compression level min(l, maxFixLevel_).
 */
struct IndexSHG : IndexHNSWFlat {
    using storage_idx_t = HNSW::storage_idx_t;

    // --- compression ---

    /// Compression branching factor. Paper (Section 3.1) uses η=2.
    int eta = 2;

    /// Maximum compression level (computed from d and eta).
    int maxFixLevel_ = 0;

    /**
     * Compressed representations for all ntotal vectors.
     * Layout: compressed_vecs[node_id * data_rep_size_ + offset]
     * where offset is the start of the compressed data for a given level.
     * Level 0 = full dims (stored in flat storage, NOT here).
     * Levels 1..maxFixLevel_ are stored here concatenated.
     */
    std::vector<float> compressed_vecs;

    /// Total compressed representation size per node
    /// (sum of dims for levels 1..maxFixLevel_).
    size_t data_rep_size_ = 0;

    /// Dimension at each compression level (0..maxFixLevel_).
    std::vector<int> dim_at_level;

    /// Cumulative offset into per-node compressed data for each level.
    /// offset_at_level[l] = sum of dim_at_level[i] for i in 1..l-1.
    std::vector<size_t> offset_at_level;

    // --- shortcut ---

    ShortcutMap shortcut;

    // --- lifecycle ---

    explicit IndexSHG(
            int d = 0,
            int M = 32,
            MetricType metric = METRIC_L2);

    /**
     * Add vectors, building the HNSW graph with compressed distances
     * at upper levels (Algorithm 2 line 7 from the paper).
     */
    void add(idx_t n, const float* x) override;

    /**
     * Build compressed vectors for all nodes and train the shortcut.
     * Must be called once after all vectors have been added via add().
     */
    void build_shortcut();

    // --- core interface ---

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // --- public helpers ---

    /// Return dimension at compression level l.
    int get_dim_at_level(int l) const;

    /// Compress a d-dimensional vector to compression level l.
    static void compress_vector(
            const float* vec,
            int d,
            int l,
            int eta,
            float* out);

    /// Squared L2 distance between two compressed vectors of length dim.
    static float compressed_l2sqr(
            const float* a,
            const float* b,
            int dim);

    /// Get pointer to compressed data for a node at a given compression level.
    /// Level 0 returns nullptr (use flat storage for level 0).
    const float* get_compressed_data(idx_t node_id, int comp_level) const;

    /// Compute compressed squared L2 distance between two nodes
    /// at a given HNSW level (capped at maxFixLevel_).
    float get_dis_by_level(idx_t id1, idx_t id2, int hnsw_level) const;

    /// Compute compressed squared L2 distance between a pre-built
    /// query compressed representation and a node at a given HNSW level.
    float get_dis_by_level_q(
            const std::vector<float>& query_rep,
            idx_t node_id,
            int hnsw_level) const;

    /// Distance cache with epoch-based invalidation: avoids an O(ntotal)
    /// clear per query by only incrementing a counter. Reserved for
    /// cross-level distance reuse; NOT instantiated by the current search
    /// path, which uses a per-candidate compressed lower bound instead.
    struct DisCache {
        std::vector<float> values;
        std::vector<uint32_t> stamps;
        uint32_t cur_stamp = 0;

        void resize(size_t n) {
            values.resize(n);
            stamps.resize(n, 0);
        }
        void new_query() {
            if (++cur_stamp == 0) {
                std::fill(stamps.begin(), stamps.end(), 0u);
                cur_stamp = 1;
            }
        }
        float get(size_t idx) const {
            return stamps[idx] == cur_stamp ? values[idx] : -1.0f;
        }
        void set(size_t idx, float val) {
            values[idx] = val;
            stamps[idx] = cur_stamp;
        }
    };
    using dis_cache_t = DisCache;

   private:
    /// Shared distance computation for two compressed data pointers at a
    /// given HNSW level. Both get_dis_by_level and get_dis_by_level_q
    /// delegate here for levels > 0, ensuring a single code path.
    float compressed_dis_at_level(
            const float* a_data,
            const float* b_data,
            int hnsw_level) const;

    void compute_compression_params();
    void build_all_compressed();
    void compress_node(idx_t node_id);

    std::pair<float, idx_t> get_nearest_by_level(
            idx_t node_id, int hnsw_level) const;

    void build_shortcuts_density();

    /// Build compressed query representation into query_rep.
    /// full_rep is a per-thread scratch buffer.
    void build_compressed_query_rep(
            const float* query,
            std::vector<float>& full_rep,
            std::vector<float>& query_rep) const;

    /// Navigate upper levels using compressed distances + optional shortcuts.
    /// Uses a forked version of FAISS's greedy_update_nearest with inline
    /// compressed distance computation (batch-4 pattern).
    storage_idx_t navigate_upper_levels(
            const std::vector<float>& query_rep,
            bool use_shortcut) const;
};

} // namespace faiss
