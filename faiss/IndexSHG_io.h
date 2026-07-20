/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/**
 * IndexSHG I/O support.
 *
 * Provides write/read functions for the SHG-specific data
 * (compressed vectors, shortcut map, compression parameters).
 * The base HNSW index is written/read by the standard FAISS I/O.
 */

#pragma once

#include <faiss/IndexSHG.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include <cstdio>
#include <vector>

namespace faiss {

// -------------------------------------------------------------------------
// Write
// -------------------------------------------------------------------------

inline void write_index_shg_extra(const IndexSHG* idx, IOWriter* f) {
    // compression parameters
    WRITE1(idx->eta);
    WRITE1(idx->maxFixLevel_);

    // dim_at_level
    int n_levels = (int)idx->dim_at_level.size();
    WRITE1(n_levels);
    if (n_levels > 0) {
        WRITEVECTOR(idx->dim_at_level);
    }

    // offset_at_level
    int n_offsets = (int)idx->offset_at_level.size();
    WRITE1(n_offsets);
    if (n_offsets > 0) {
        for (int i = 0; i < n_offsets; ++i) {
            uint64_t off = (uint64_t)idx->offset_at_level[i];
            WRITE1(off);
        }
    }

    // data_rep_size_
    uint64_t rep_size = (uint64_t)idx->data_rep_size_;
    WRITE1(rep_size);

    // compressed_vecs (flat array)
    uint64_t cv_size = (uint64_t)idx->compressed_vecs.size();
    WRITE1(cv_size);
    if (cv_size > 0) {
        f->operator()(
                idx->compressed_vecs.data(),
                sizeof(float),
                cv_size);
    }

    // Learned shortcut: only the training samples are written. They are
    // already sorted and deduplicated by ShortcutMap::finalize(), so the
    // on-disk layout is unchanged from the previous std::map-backed version.
    // The piecewise linear model is refitted on read rather than serialized —
    // fitting is O(samples) and the sample count is small.
    int sc_size = idx->shortcut.size();
    WRITE1(sc_size);
    for (int i = 0; i < sc_size; ++i) {
        WRITE1(idx->shortcut.keys[i]);
        WRITE1(idx->shortcut.values[i]);
    }
}

// -------------------------------------------------------------------------
// Read
// -------------------------------------------------------------------------

inline void read_index_shg_extra(IndexSHG* idx, IOReader* f) {
    READ1(idx->eta);
    READ1(idx->maxFixLevel_);

    int n_levels = 0;
    READ1(n_levels);
    idx->dim_at_level.resize(n_levels);
    if (n_levels > 0) {
        READVECTOR(idx->dim_at_level);
    }

    int n_offsets = 0;
    READ1(n_offsets);
    idx->offset_at_level.resize(n_offsets);
    for (int i = 0; i < n_offsets; ++i) {
        uint64_t off = 0;
        READ1(off);
        idx->offset_at_level[i] = (size_t)off;
    }

    uint64_t rep_size = 0;
    READ1(rep_size);
    idx->data_rep_size_ = (size_t)rep_size;

    uint64_t cv_size = 0;
    READ1(cv_size);
    idx->compressed_vecs.resize(cv_size);
    if (cv_size > 0) {
        f->operator()(
                idx->compressed_vecs.data(),
                sizeof(float),
                cv_size);
    }

    int sc_size = 0;
    READ1(sc_size);
    FAISS_THROW_IF_NOT_MSG(sc_size >= 0, "IndexSHG: corrupt shortcut size");
    idx->shortcut.clear();
    idx->shortcut.keys.resize(sc_size);
    idx->shortcut.values.resize(sc_size);
    for (int i = 0; i < sc_size; ++i) {
        READ1(idx->shortcut.keys[i]);
        READ1(idx->shortcut.values[i]);
    }
    // Samples were written sorted and deduplicated; refit the model over them.
    idx->shortcut.build_model();
}

} // namespace faiss
