## t‑NeRV: Heavy‑Tailed Neighborhood Retrieval Visualizer on TorchDR

This repository provides a **PyTorch / TorchDR implementation of t‑NeRV**, a heavy‑tailed variant of the Neighborhood Retrieval Visualizer (NeRV) designed for dimensionality reduction.

The core implementation lives in the `torchdr` package (adapted and extended from the original TorchDR project), with t‑NeRV exposed as `torchdr.neighbor_embedding.TNERV`. A lightweight `datasetloader` module and example scripts make it easy to reproduce standard benchmarks (e.g. MNIST, single‑cell datasets, COIL‑100).

This codebase is meant to support the **t‑NeRV paper** and to serve as a clean, reusable library for follow‑up work.

---

## 1. Features at a Glance

- **t‑NeRV implementation (`TNERV`)**
  - Information‑theoretic loss combining forward and reverse KL.
  - Heavy‑tailed Student‑t kernel in the low‑dimensional space.
  - Controls the trade‑off between recall and precision via `lambda_param`.
- **Shared TorchDR infrastructure**
  - `DRModule` base class with `fit` / `fit_transform` and sklearn‑style API.
  - `SparseNeighborEmbedding` for efficient k‑NN based methods.
  - `EntropicAffinity` for perplexity‑controlled input affinities.
  - Optional **KeOps** and **FAISS** backends for large‑scale problems.
- **Multiple DR methods in a unified framework**
  - Neighbor embeddings: `TSNE`, `SNE`, `NERV`, `TNERV`.
  - Spectral embeddings: `PCA`, `KernelPCA`, `IncrementalPCA`, `PHATE`, etc.
- **Evaluation utilities**
  - K‑NN classification (`torchdr.eval.KNN`).
  - Neighborhood preservation metrics and NX curves.
  - Silhouette scores and clustering helpers.
- **Minimal, path‑agnostic dataset loader**
  - `DatasetLoader` for MNIST, Fashion‑MNIST, scanpy benchmarks, Tasic, COIL‑100.
  - No hard‑coded machine paths; all data paths are explicit.

---

## 2. Installation

### 2.1. Basic installation

The project is packaged as `t-nerv` with a local `torchdr` module. From the `t-NeRV` directory:

```bash
pip install -e .
```

This installs:

- core dependencies (`torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`);
- the `torchdr` package and its submodules.

### 2.2. Optional backends

For large‑scale experiments, you will typically want:

- **KeOps** (symbolic GPU computations):

```bash
pip install "t-nerv[keops]"
```

- **FAISS** (fast k‑NN search):

```bash
pip install "t-nerv[faiss]"
```

These extras are configured in `pyproject.toml` and pull platform‑appropriate wheels where available.

---

## 3. Quick Start

### 3.1. MNIST + t‑NeRV example

The repository includes an example in `Example_MNIST.py`. It:

1. loads MNIST via `DatasetLoader`,
2. fits a 2‑D t‑NeRV embedding on CUDA using the KeOps backend,
3. prints shapes and runtime.

```python
import time

from datasetloader.datasetloader import DatasetLoader
from torchdr.neighbor_embedding.t_nerv import TNERV


def main():
    X, _ = DatasetLoader.load_mnist()

    model = TNERV(n_components=2, perplexity=30.0, device="cuda", backend="keops")

    t0 = time.perf_counter()
    embedding = model.fit_transform(X)
    runtime = time.perf_counter() - t0

    print(f"X shape: {X.shape}, embedding shape: {embedding.shape}, runtime: {runtime:.3f}s")
    return X.shape, embedding.shape, runtime


if __name__ == "__main__":
    main()
```

Run:

```bash
python Example_MNIST.py
```

> Note: `time` is part of the Python standard library; **no change to `pyproject.toml` is needed** for timing.

### 3.2. Using t‑NeRV directly from `torchdr`

After installation, you can import `TNERV` wherever you like:

```python
from torchdr.neighbor_embedding import TNERV

model = TNERV(
    perplexity=50.0,
    n_components=2,
    lambda_param=0.5,
    degrees_of_freedom=1.0,
    device="cuda",
    backend="keops",
)
Z = model.fit_transform(X) 
```

---

## 4. Codebase Overview

High‑level layout:

- **`torchdr/`** – core dimensionality‑reduction library
  - `base.py` – `DRModule` base class with `fit` / `fit_transform` logic.
  - `neighbor_embedding/` – neighbor‑embedding methods:
    - `base.py` – `NeighborEmbedding`, `SparseNeighborEmbedding`.
    - `tsne.py` – t‑SNE implementation.
    - `sne.py` – SNE variants.
    - `nerv.py` – NeRV implementation.
    - `t_nerv.py` – t‑NeRV implementation (`TNERV`).
  - `spectral_embedding/` – PCA, KernelPCA, PHATE, etc.
  - `affinity/` – affinity models:
    - `entropic.py` – `EntropicAffinity` (perplexity‑driven affinities).
    - others: quadratic, self‑tuning, UMAP, PHATE, etc.
  - `distance/` – distance computations with optional KeOps/FAISS backends.
  - `eval/` – evaluation utilities: KNN, NX curves, neighborhood preservation.
  - `utils/` – numerical helpers, manifold tools, wrappers, visualization, etc.
- **`datasetloader/`**
  - `datasetloader.py` – minimal, path‑agnostic dataset loaders (`DatasetLoader`).
- **Top‑level utilities**
  - `Example_MNIST.py` – small runnable demo script.
  - `pyproject.toml` – packaging and dependency configuration.

This structure is already close to what is expected of a NeurIPS/ICLR/ICML‑grade library; the main work is in tightening documentation, tests, examples, and experiment scripts.

---

## 5. API Highlights (t‑NeRV)

The central class is `torchdr.neighbor_embedding.TNERV.TNERV`, a subclass of `SparseNeighborEmbedding` and ultimately `DRModule`. Key parameters:

- **Geometry / optimization**
  - `perplexity: float = 30.0` – controls effective neighborhood size in input space.
  - `n_components: int = 2` – embedding dimension.
  - `lr: float | "auto" = "auto"` – learning rate (sklearn‑style `"auto"` heuristic).
  - `optimizer: str | torch.optim.Optimizer = "SGD"` – optimizer choice.
  - `max_iter: int = 2000` – max optimization steps.
  - `init: str = "pca"` – initialization (`"pca"` or random normal).
  - `device: Optional[str] = None` – `"cuda"`, `"cpu"`, or `"auto"`.
  - `backend: {"keops", "faiss", None} | FaissConfig = None` – backend for distances/affinities.
- **Loss / model hyper‑parameters**
  - `lambda_param: float = 0.5` – trade‑off between forward and reverse KL.
  - `degrees_of_freedom: float = 1.0` – Student‑t df for low‑dimensional kernel.
  - `p_smoothing: float = 1e-12` – smoothing for input distribution \(P\).
  - `early_exaggeration_coeff: float = 12.0`, `early_exaggeration_iter: int = 250`.
- **Practical controls**
  - `sparsity: bool = True` – sparse input affinities (k‑NN‑style).
  - `max_iter_affinity: int = 100` – iterations for entropic affinity solver.
  - `check_interval: int = 50` – convergence checking frequency.
  - `compile: bool = False` – optional `torch.compile` acceleration.

The main entry point is:

```python
embedding = model.fit_transform(X)
```

where `X` can be a NumPy array or a PyTorch tensor of shape `(n_samples, n_features)`. All DR modules follow this sklearn‑style pattern.

---

## 6. Dataset Loader

The `datasetloader/datasetloader.py` module provides a small `DatasetLoader` class and helper for Tasic:

- **OpenML datasets**
  - `load_mnist()` – returns `(X, y)` with shape `(60000, 784)` and `(60000,)`.
  - `load_fashion_mnist()` – analogous for Fashion‑MNIST.
- **Scanpy benchmarks**
  - `load_macosko()`, `load_10x_zheng()` – via hosted `.pkl.gz` files with PCA50 and labels.
- **Single‑cell (Tasic)**
  - `load_tasic_preprocessed(data_dir, ...)` – loads PCA50 and transcriptomic types.
  - `DatasetLoader.load_tasic(data_dir)` – user‑facing wrapper returning `(X, y)`.
- **COIL‑100**
  - `DatasetLoader.load_coil100(data_dir, ..., backend options)` – image loading, optional download, and flattening for DR.

All loaders are **explicitly path‑based** (no machine‑specific defaults). When running experiments, pass your data root as `data_dir` or set it via your own scripting logic.

---

## 7. License

The repository is distributed under a **BSD 3‑Clause License** (see `LICENSE.txt`).  
If you adapt or extend this code, please preserve existing copyright notices and
provide clear attribution to both the original TorchDR authors and the t‑NeRV authors.

