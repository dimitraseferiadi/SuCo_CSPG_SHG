# Vendored PGM-index

`pgm_index.hpp` and `piecewise_linear_model.hpp` are copied **verbatim** from the
PGM-index reference implementation:

- Upstream: <https://github.com/gvinciguerra/PGM-index>
- Copyright (c) 2018 Giorgio Vinciguerra
- Licensed under the Apache License, Version 2.0
  (<http://www.apache.org/licenses/LICENSE-2.0>) — see the header of each file.

They are used by `faiss/IndexSHG.h` to implement the *learned shortcut* of
Definition 4 in:

> Gong, Zeng, Chen. "Accelerating Approximate Nearest Neighbor Search in
> Hierarchical Graphs: Efficient Level Navigation with Shortcuts."
> PVLDB 18(10): 3518-3530, 2025.

The SHG-Index authors' own implementation uses the same library
(`pgm::DynamicPGMIndex<dist_t, int>`). We use the static `pgm::PGMIndex`
instead, because the shortcut is built once by `IndexSHG::build_shortcut()`
and is read-only thereafter — the dynamic variant's insert path is not needed.

These two headers are self-contained; the rest of the upstream distribution
(`pgm_index_variants.hpp`, `pgm_index_dynamic.hpp`, `sdsl.hpp`, `morton_nd.hpp`)
is not vendored.

**Do not modify these files.** Keeping them byte-identical to upstream makes it
possible to diff against a new release. Any SHG-specific adaptation belongs in
`faiss/IndexSHG.h`.
