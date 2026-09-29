# SuCo, CSPG and SHG in FAISS

This repository extends [FAISS](https://github.com/facebookresearch/faiss) with native implementations of three recent approximate nearest neighbour search (ANNS) methods: SuCo, CSPG and SHG. Each method is implemented as a first-class FAISS index. All three expose C++ and Python interfaces, support serialisation, and use the FAISS SIMD distance kernels and OpenMP parallelism. The repository also provides a benchmark suite that evaluates the three methods against standard FAISS baselines.

## Overview

| Index | Method | Reference |
|---|---|---|
| [`IndexSuCo`](faiss/IndexSuCo.h) | Subspace Collision | Wei et al., SIGMOD 2025 |
| [`IndexCSPG`](faiss/IndexCSPG.h) | Crossing Sparse Proximity Graphs | Yang et al., NeurIPS 2024 |
| [`IndexSHG`](faiss/IndexSHG.h) | Shortcut-enabled Hierarchical Graph | Gong et al., PVLDB 2025 |

- **SuCo** selects candidates by counting collisions with the query instead of ranking by distance. Each vector is split into Ns subspaces, and a point collides with the query in a subspace if it is among the query's αn nearest neighbours there. The βn points that collide in the most subspaces are re-ranked by exact L2 distance. Per-subspace neighbours are retrieved from an inverted multi-index built with k-means.
- **CSPG** randomly partitions the dataset into m subsets and replicates a fraction λ of the vectors, the routing vectors, in every subset. A proximity graph (HNSW in this implementation) is built on each subset. A search first approaches the query on one graph, then expands across partitions: when it reaches a routing vector, that vector's copies in the other graphs join the candidate set.
- **SHG** augments HNSW with two mechanisms. Distances on the upper levels are computed on progressively coarser vectors, obtained by repeatedly averaging pairs of adjacent coordinates. A learned shortcut, a piecewise-linear model fitted with a PGM-index, predicts from the approximate distance to the entry point how many upper levels the descent can skip. The compressed vectors also yield a lower bound that prunes base-level candidates before their exact distance is computed.

## Installation

The build follows the standard FAISS CMake procedure; see [INSTALL.md](INSTALL.md) for prerequisites. For a CPU-only build with Python bindings:

```bash
cmake -B build . \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DFAISS_OPT_LEVEL=avx512
make -C build -j faiss faiss_avx512 swigfaiss swigfaiss_avx512
(cd build/faiss/python && python setup.py install)
```

On processors without AVX-512 support, use `avx2` in place of `avx512`. On ARM platforms, including Apple silicon, set `-DFAISS_OPT_LEVEL=generic` and build only the `faiss` and `swigfaiss` targets.

## Usage

```python
import faiss
import numpy as np

d, k = 128, 10
xb = np.random.rand(100_000, d).astype("float32")
xq = np.random.rand(1_000, d).astype("float32")

# SuCo (d must be divisible by the number of subspaces)
suco = faiss.IndexSuCo(d, 8, 50, 0.05, 0.005)   # d, Ns, sqrt(K), alpha, beta
suco.train(xb)
suco.add(xb)
D, I = suco.search(xq, k, params=faiss.SearchParametersSuCo(candidate_ratio=0.01))

# CSPG (all vectors must be added in a single call)
cspg = faiss.IndexCSPG(d, 32, 2, 0.5)           # d, M, partitions, lambda
cspg.efConstruction = 128
cspg.add(xb)
D, I = cspg.search(xq, k, params=faiss.SearchParametersCSPG(efSearch=128))

# SHG (the shortcut is built once, after all vectors are added)
shg = faiss.IndexSHG(d, 48)
shg.hnsw.efConstruction = 80
shg.add(xb)
shg.build_shortcut()
D, I = shg.search(xq, k, params=faiss.SearchParametersSHG(efSearch=128))

# Serialisation
faiss.write_index(shg, "shg.index")
shg = faiss.read_index("shg.index")
```

SuCo has two search-time parameters: `collision_ratio` (α) and `candidate_ratio` (β). The benchmarks fix α = 0.05 and vary β. CSPG and SHG are tuned through `efSearch`.

## Benchmarks

The [benchs/](benchs/) directory contains the benchmark suite. It compares SuCo, CSPG and SHG with HNSW, IVFFlat and OPQ-IVFPQ baselines on eleven public datasets: SIFT, GIST, Deep, SpaceV, MSong, Enron, OpenAI, MSTuring and UQ-V. It also includes a scaling study on SIFT up to 100 million vectors.

### Results

All raw results are included in the repository. The figures and tables can be regenerated without re-running the experiments:

```bash
python benchs/plot_router_paper.py
```

| Directory | Contents |
|---|---|
| `benchs/results_router/` | Cross-method comparison |
| `benchs/bigann_results/` | SIFT scaling study |
| `benchs/results_suco_repro/`, `benchs/results_cspg/`, `benchs/results_shg/` | Reproductions of each method's original experiments |
| `logs/slurm/` | Raw SLURM job logs |

### Running the experiments

The main entry point is `benchs/bench_router_paper.py`:

```bash
python benchs/bench_router_paper.py --dataset sift1m --benchmark all \
    --data-dir /path/to/data --index-dir /path/to/indices
```

The expected layout of each dataset is documented in [bench_datasets.py](benchs/bench_datasets.py). The `benchs/*.sbatch` scripts are the SLURM jobs used on the LEONARDO supercomputer. Their account, module and path settings must be adapted before use on other systems.

## References

- J. Wei, X. Lee, Z. Liao, T. Palpanas, and B. Peng, "Subspace collision: An efficient and accurate framework for high-dimensional approximate nearest neighbor search," Proc. ACM Manag. Data, vol. 3, no. 1, Feb. 2025. [Online]. Available: https://doi.org/10.1145/3709729
- M. Yang, Y. Cai, and W. Zheng, "Cspg: Crossing sparse proximity graphs for approximate nearest neighbor search," in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 103 076–103 100.[Online]. Available: https://proceedings.neurips.cc/paper files/paper/2024/file/bab1486cec466c980b40e7d633dd4bbc-Paper-Conference.pdf
- Z. Gong, Y. Zeng, and L. Chen, "Accelerating approximate nearest neighbor search in hierarchical graphs: Efficient level navigation with shortcuts," Proc. VLDB Endow., vol. 18, no. 10, p. 3518–3530, Jun. 2025. [Online]. Available: https://doi.org/10.14778/3748191.3748212
- M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou, "The faiss library," 2024.

## License

This project is released under the MIT License, as is FAISS; see [LICENSE](LICENSE). The vendored [PGM-index](faiss/impl/pgm/) used by SHG is licensed under the Apache License 2.0.

## Acknowledgements

The experiments were run on the LEONARDO supercomputer hosted by CINECA, Italy, through EuroHPC Joint Undertaking (JU) project EHPC-DEV-2026D01-102.
