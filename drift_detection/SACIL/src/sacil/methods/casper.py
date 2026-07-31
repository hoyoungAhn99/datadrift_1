from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_feature_distance(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    return torch.cdist(features, features, p=2).square()


def _log_one_minus_exp(log_probability: Tensor) -> Tensor:
    split = log_probability < math.log(0.5)
    result = torch.empty_like(log_probability)
    result[split] = torch.log1p(-log_probability[split].exp())
    result[~split] = torch.log(-torch.expm1(log_probability[~split]))
    return result


def neural_knn_weights(
    negative_distances: Tensor,
    k: int,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Differentiable k-nearest-neighbour relaxation used by CaSpeR-IL."""

    if negative_distances.ndim != 2:
        raise ValueError("distance scores must be a square matrix")
    if negative_distances.shape[0] != negative_distances.shape[1]:
        raise ValueError("distance scores must be square")
    if k <= 0 or k >= negative_distances.shape[0]:
        raise ValueError("k must be in [1, number_of_samples)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = negative_distances / float(temperature)
    selections = []
    for _ in range(k):
        log_weights = F.log_softmax(logits, dim=1)
        selections.append(log_weights.exp())
        logits = logits + _log_one_minus_exp(log_weights)
    return torch.stack(selections, dim=-1).sum(dim=-1)


def casper_spectral_loss(
    features: Tensor,
    *,
    num_classes: int,
    k: int = 10,
    temperature: float = 1.0,
) -> Tensor:
    """CaSpeR-IL Laplacian eigengap objective on a replay minibatch."""

    sample_count = features.shape[0]
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if sample_count < 3:
        return features.sum() * 0.0
    effective_k = min(int(k), sample_count - 1)
    distances = pairwise_feature_distance(features)
    masked = distances.clone()
    masked.fill_diagonal_(torch.inf)

    hard = torch.zeros_like(masked)
    indices = masked.topk(effective_k, largest=False, dim=1).indices
    hard.scatter_(1, indices, 1.0)
    hard = ((hard + hard.T) > 0).to(features.dtype)

    soft = neural_knn_weights(
        -masked, effective_k, temperature=temperature
    )
    soft = soft + soft.T
    affinity = hard.detach() + soft - soft.detach()
    degree = affinity.sum(dim=1).clamp_min(torch.finfo(features.dtype).eps)
    normalized = affinity / torch.sqrt(degree[:, None] * degree[None, :])
    laplacian = torch.eye(
        sample_count, device=features.device, dtype=features.dtype
    ) - normalized
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    n = min(int(num_classes), sample_count - 2)
    return eigenvalues[: n + 1].sum() - eigenvalues[n + 1]

