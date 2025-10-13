"""t-distributed Neighborhood Retrieval Visualizer (t-NeRV)."""

from typing import Dict, Optional, Union, Type

import torch

from torchdr.affinity import EntropicAffinity
from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.utils import cross_entropy_loss, logsumexp_red


class TNERV(SparseNeighborEmbedding):
    r"""t-NeRV embedding introduced in :cite:`venna2010neighbor`."""

    def __init__(
        self,
        perplexity: float = 30,
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
        check_interval: int = 50,
        compile: bool = False,
        degrees_of_freedom: float = 1.0,
        lambda_param: float = 0.5,
        **kwargs,
    ):
        if degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive.")
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must lie in the interval [0, 1].")

        self.metric = metric
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.sparsity = sparsity
        self.degrees_of_freedom = degrees_of_freedom
        self.lambda_param = lambda_param
        self._use_direct_gradients = True
        self._p_smoothing = 1e-12

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
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

    @torch.no_grad()
    def _build_dense_affinity(self, device: torch.device, dtype: torch.dtype):
        n = self.n_samples_in_

        if hasattr(self, "NN_indices_"):
            values = self.affinity_in_.to(device=device, dtype=dtype)
            indices = self.NN_indices_.to(device)
            dense = torch.full(
                (n, n), self._p_smoothing, device=device, dtype=dtype
            )
            dense.scatter_(1, indices, values)
        else:
            dense = self.affinity_in_.to(device=device, dtype=dtype).clone()
            dense = dense.clamp_min(self._p_smoothing)

        diag_idx = torch.arange(n, device=device)
        dense[diag_idx, diag_idx] = self._p_smoothing
        total_mass = dense.sum()
        if total_mass <= 0:
            raise ValueError("Dense affinity must have positive total mass.")
        dense /= total_mass
        log_dense = dense.clamp_min(self._p_smoothing).log()

        self._affinity_dense_cache = dense
        self._log_affinity_dense_cache = log_dense

    def _get_dense_affinity(self, device: torch.device, dtype: torch.dtype):
        cache = getattr(self, "_affinity_dense_cache", None)
        if (
            cache is None
            or cache.device != device
            or cache.dtype != dtype
        ):
            self._build_dense_affinity(device, dtype)
        return self._affinity_dense_cache, self._log_affinity_dense_cache

    @torch.no_grad()
    def _compute_common_terms(self):
        device = self.embedding_.device
        dtype = self.embedding_.dtype
        P_dense, log_P_dense = self._get_dense_affinity(device, dtype)

        distances_sq = pairwise_distances(
            self.embedding_, metric="sqeuclidean", backend=self.backend
        )
        nu = self.degrees_of_freedom
        log_w_full = -0.5 * (nu + 1.0) * torch.log1p(distances_sq / nu)
        diag_idx = torch.arange(distances_sq.shape[0], device=device)
        log_w_full[diag_idx, diag_idx] = float("-inf")

        logZ = logsumexp_red(log_w_full, dim=(0, 1))
        log_q_full = log_w_full - logZ
        q_full = torch.exp(log_q_full)
        q_full[diag_idx, diag_idx] = 0.0
        log_q_safe = torch.where(q_full > 0, log_q_full, torch.zeros_like(log_q_full))

        student_factor = nu / (nu + distances_sq)
        student_factor[diag_idx, diag_idx] = 0.0

        if hasattr(self, "NN_indices_"):
            log_w_neighbors = log_w_full.gather(
                1, self.NN_indices_.to(device)
            )
        else:
            log_w_neighbors = log_w_full

        log_ratio = log_q_safe - log_P_dense
        avg_log_ratio = (q_full * log_ratio).sum()

        p_log_p = (P_dense * log_P_dense).sum()

        return {
            "P_dense": P_dense,
            "log_P_dense": log_P_dense,
            "log_w_neighbors": log_w_neighbors,
            "logZ": logZ,
            "q_full": q_full,
            "log_ratio": log_ratio,
            "avg_log_ratio": avg_log_ratio,
            "student_factor": student_factor,
            "p_log_p": p_log_p,
        }

    def _compute_loss(self):
        with torch.no_grad():
            terms = self._compute_common_terms()
            ce_term = cross_entropy_loss(
                self.affinity_in_.to(
                    device=self.embedding_.device, dtype=self.embedding_.dtype
                ),
                terms["log_w_neighbors"],
                log=True,
            )
            loss = (
                self.lambda_param
                * (
                    self.early_exaggeration_coeff_ * ce_term
                    + terms["logZ"]
                    + terms["p_log_p"]
                )
                + (1.0 - self.lambda_param) * terms["avg_log_ratio"]
            )
        return loss

    @torch.no_grad()
    def _compute_gradients(self):
        terms = self._compute_common_terms()
        q_full = terms["q_full"]
        log_ratio = terms["log_ratio"]
        avg_log_ratio = terms["avg_log_ratio"]
        student_factor = terms["student_factor"]
        P_dense = terms["P_dense"]

        nu = self.degrees_of_freedom
        prefactor = 2.0 * (nu + 1.0) / nu
        diff = self.embedding_.unsqueeze(1) - self.embedding_.unsqueeze(0)

        weighted = (
            self.lambda_param
            * (self.early_exaggeration_coeff_ * P_dense - q_full)
            + (1.0 - self.lambda_param)
            * q_full
            * (log_ratio - avg_log_ratio)
        )
        forces = weighted * student_factor
        gradients = prefactor * (forces.unsqueeze(-1) * diff).sum(dim=1)
        return gradients

    def clear_memory(self):
        for attr in ("_affinity_dense_cache", "_log_affinity_dense_cache"):
            if hasattr(self, attr):
                delattr(self, attr)
        return super().clear_memory()