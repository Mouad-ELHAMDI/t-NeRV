"""
t-NeRV implementation inspired by TorchDR.

This mirrors TorchDR’s neighbor-embedding style:
- Sparse input affinity via EntropicAffinity (k-NN with FAISS if requested).
- Full pairwise terms done with KeOps LazyTensors when backend="keops" (no NxN allocation).
- No manual chunking: we rely on LazyTensor algebra + TorchDR reducers.

Loss = lambda * KL(P || Q) + (1-lambda) * KL(Q || P)
See docstring in the user prompt for details.
"""

from __future__ import annotations

from typing import Dict, Optional, Union, Type

import torch

from torchdr.affinity import EntropicAffinity
from torchdr.distance import (
    FaissConfig,
    pairwise_distances,
    pairwise_distances_indexed,
)
from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.utils import (
    cross_entropy_loss,
    logsumexp_red,
    sum_red,
    identity_matrix,
)

try:
    # Optional import; we only use this to *detect* LazyTensor objects
    from pykeops.torch import LazyTensor as _LazyTensor  # type: ignore
except Exception:  # pragma: no cover
    _LazyTensor = ()  # sentinel for isinstance checks


def _is_lazy(x) -> bool:
    """Return True if x is a KeOps LazyTensor."""
    return bool(_LazyTensor) and isinstance(x, _LazyTensor)


class TNERV(SparseNeighborEmbedding):
    r"""t-distributed Neighborhood Retrieval Visualizer (Venna et al., 2010)."""

    def __init__(
        self,
        perplexity: float = 30.0,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        device: Optional[str] = None,
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: int = 250,
        max_iter_affinity: int = 100,
        metric: str = "sqeuclidean",
        sparsity: bool = True,
        lambda_param: float = 0.5,
        degrees_of_freedom: float = 1.0,
        p_smoothing: float = 1e-12,
        check_interval: int = 50,
        compile: bool = False,
        **kwargs: Dict,
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must lie in [0, 1].")
        if degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive.")

        self.metric = metric
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.sparsity = bool(sparsity)
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.lambda_param = float(lambda_param)
        self._p_smoothing = float(p_smoothing)

        # Rely on autograd for gradients:
        self._use_direct_gradients = False

        # Input affinity: FAISS/KeOps selection flows through `backend`.
        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,   # FAISS for k-NN if FaissConfig or "faiss"; KeOps if "keops"
            verbose=verbose,
            sparsity=sparsity,
        )

        super().__init__(
            affinity_in=affinity_in,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            check_interval=check_interval,
            compile=compile,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #                            t-NeRV loss                             #
    # ------------------------------------------------------------------ #
    def _compute_loss(self) -> torch.Tensor:
        device = self.embedding_.device
        dtype = self.embedding_.dtype
        n = self.n_samples_in_
        nu = torch.as_tensor(self.degrees_of_freedom, device=device, dtype=dtype)

        # ---------- Attractive term: neighbors only (dense [n,k]) ----------
        D2_nn = pairwise_distances_indexed(
            self.embedding_,
            key_indices=self.NN_indices_,   # [n, k]
            metric="sqeuclidean",
            backend=self.backend,
        )
        if _is_lazy(D2_nn):
            # (rare; typically returns Tensor)
            logW_nn = (-0.5 * (nu + 1.0)) * (1 + D2_nn / nu).log()
            # Materialize small (n,k) for CE if needed:
            D2_nn_t = (D2_nn).torch()  # LazyTensor -> torch.Tensor
            logW_nn = (-0.5 * (nu + 1.0)) * torch.log1p(D2_nn_t / nu)
        else:
            logW_nn = (-0.5 * (nu + 1.0)) * torch.log1p(D2_nn / nu)

        # CE over neighbors, unnormalized by Z
        ce_loss = cross_entropy_loss(self.affinity_in_, logW_nn, log=True)

        # ---------- Repulsive + reverse-KL terms: all pairs ----------
        D2_full = pairwise_distances(
            self.embedding_, metric="sqeuclidean", backend=self.backend
        )

        if _is_lazy(D2_full):
            # KeOps path: use LazyTensor-friendly ops
            logW_full = (-0.5 * (nu + 1.0)) * (1 + D2_full / nu).log()

            # Mask diagonal without indexing: add a large negative on identity
            logW_full = logW_full + identity_matrix(
                n, keops=True, device=device, dtype=dtype
            ) * (-1e6)

            # Partition function and log Q
            logZ = logsumexp_red(logW_full, dim=(0, 1))
            logQ_full = logW_full - logZ
            Q_full = logQ_full.exp()

            # Entropy of Q: sum(Q log Q)
            q_log_q_sum = sum_red(Q_full * logQ_full, dim=(0, 1))

        else:
            # Dense PyTorch path (small N)
            logW_full = (-0.5 * (nu + 1.0)) * torch.log1p(D2_full / nu)
            diag = torch.arange(n, device=device)
            logW_full[diag, diag] = float("-inf")
            logZ = torch.logsumexp(logW_full.reshape(-1), dim=0)
            logQ_full = logW_full - logZ
            Q_full = logQ_full.exp()
            q_log_q_sum = (Q_full * logQ_full).sum()

        # Attractive part: lambda * (EE * CE + logZ + H(P))
        attractive = self.lambda_param * (self.early_exaggeration_coeff_ * ce_loss + logZ)

        # ---------- H(P): entropy of input affinity without dense P ----------
        values = self.affinity_in_.to(device=device, dtype=dtype)  # [n, k]
        smoothing = torch.as_tensor(self._p_smoothing, device=device, dtype=dtype)
        total_entries = n * n
        num_neighbors = values.numel()
        total_mass = total_entries * smoothing - num_neighbors * smoothing + values.sum()
        p_base = smoothing / total_mass
        log_p_base = torch.log(p_base)
        p_neighbors = values / total_mass
        log_p_neighbors = torch.log(p_neighbors.clamp_min(torch.finfo(dtype).tiny))
        p_log_p = (p_neighbors * log_p_neighbors).sum() + (total_entries - num_neighbors) * p_base * log_p_base

        attractive = attractive + self.lambda_param * p_log_p

        # ---------- Reverse KL: KL(Q || P) = sum Q log(Q/P) ----------
        # Neighbor contributions of Q log P:
        # logQ_nn is dense [n,k] since logW_nn is dense [n,k]
        logQ_nn = logW_nn - logZ
        q_nn = logQ_nn.exp()
        q_nn_sum = q_nn.sum()
        S2 = (q_nn * log_p_neighbors).sum()

        # Non-neighbor contributions: remaining probability mass times log p_base
        S3 = (1.0 - q_nn_sum) * log_p_base

        reverse_kl = q_log_q_sum - S2 - S3

        # ---------- Final loss ----------
        loss = attractive + (1.0 - self.lambda_param) * reverse_kl
        return loss
