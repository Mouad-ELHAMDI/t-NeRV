"""K-nearest neighbour recall evaluation for embeddings."""

from typing import Optional, Union
import numpy as np
import torch
from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.utils import to_torch


def knn_recall(
    X: Union[torch.Tensor, np.ndarray],
    Z: Union[torch.Tensor, np.ndarray],
    k: int = 10,
    metric_hd: str = "euclidean",
    metric_ld: Optional[str] = None,
    backend: Optional[Union[str, FaissConfig]] = None,
    device: Optional[str] = None,
) -> Union[float, torch.Tensor]:
    """Compute the k-nearest neighbour recall between high- and low-dimensional spaces.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Original high-dimensional data.
    Z : torch.Tensor or np.ndarray of shape (n_samples, n_features_reduced)
        Reduced low-dimensional embeddings.
    k : int, default=10
        Number of nearest neighbours to consider.
    metric_hd : str, default='euclidean'
        Distance metric to compute neighbourhoods in the original space.
    metric_ld : str, optional
        Distance metric to compute neighbourhoods in the embedding space. If ``None``
        the same metric as ``metric_hd`` is used.
    backend : {'keops', 'faiss', None} or FaissConfig, optional
        Backend configuration passed to :func:`torchdr.distance.pairwise_distances`.
    device : str, optional
        Device used for computation. If ``None`` the device of the converted tensors
        is used.

    Returns
    -------
    score : float or torch.Tensor
        Average fraction of preserved neighbours across all samples. The result is
        returned as a numpy float when the inputs are numpy arrays, otherwise as a
        ``torch.Tensor`` on the requested device.
    """

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    input_is_numpy = not isinstance(X, torch.Tensor) or not isinstance(Z, torch.Tensor)

    X_t = to_torch(X)
    Z_t = to_torch(Z)

    if X_t.shape[0] != Z_t.shape[0]:
        raise ValueError(
            f"X and Z must have same number of samples, got {X_t.shape[0]} and {Z_t.shape[0]}"
        )

    n_samples = X_t.shape[0]

    if k >= n_samples:
        raise ValueError(f"k ({k}) must be less than number of samples ({n_samples})")

    if device is None:
        device_t = X_t.device
    else:
        device_t = torch.device(device)
        X_t = X_t.to(device_t)
        Z_t = Z_t.to(device_t)

    metric_ld = metric_hd if metric_ld is None else metric_ld

    _, neighbors_hd = pairwise_distances(
        X_t,
        metric=metric_hd,
        backend=backend,
        k=k,
        exclude_diag=True,
        return_indices=True,
        device=device_t,
    )

    _, neighbors_ld = pairwise_distances(
        Z_t,
        metric=metric_ld,
        backend=backend,
        k=k,
        exclude_diag=True,
        return_indices=True,
        device=device_t,
    )

    neighbors_hd_expanded = neighbors_hd.unsqueeze(2)
    neighbors_ld_expanded = neighbors_ld.unsqueeze(1)

    matches = (neighbors_hd_expanded == neighbors_ld_expanded).any(dim=2)
    recall_per_point = matches.float().sum(dim=1) / k
    score = recall_per_point.mean()

    if input_is_numpy:
        score = score.detach().cpu().numpy().item()

    return score