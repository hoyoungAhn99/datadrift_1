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
    # A categorical probability may round to exactly one.  Clamp its log
    # below zero so both branches retain a finite forward and backward pass.
    epsilon = torch.finfo(log_probability.dtype).eps
    values = log_probability.clamp_max(-epsilon)
    return torch.where(
        values < math.log(0.5),
        torch.log1p(-values.exp()),
        torch.log(-torch.expm1(values)),
    )


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
    solver: str = "partial",
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
    n = min(int(num_classes), sample_count - 2)
    if solver == "partial":
        # The release computes only the smallest 2p eigenpairs with xitorch.
        # torch.lobpcg provides the same partial symmetric-eigensolver role
        # without adding a framework dependency.  Its orthogonal algorithm
        # requires N >= 3k, so tiny unit-test graphs use a deterministically
        # regularized dense fallback.
        pair_count = min(max(2 * n, n + 2), sample_count - 1)
        symmetric = 0.5 * (laplacian + laplacian.T)
        if pair_count > 0 and sample_count >= 3 * pair_count:
            # PyTorch's LOBPCG eigenvector backward is unstable for graph
            # Laplacians with clustered eigenvalues.  The loss depends only on
            # eigenvalues, so use detached Ritz vectors and differentiate their
            # Rayleigh quotients.  By the Hellmann-Feynman theorem this is the
            # exact first-order eigenvalue gradient for a converged eigenspace,
            # while avoiding the ill-conditioned eigenvector derivative.
            with torch.no_grad():
                _, eigenvectors = torch.lobpcg(
                    symmetric.detach(),
                    k=pair_count,
                    largest=False,
                    method="ortho",
                    niter=80,
                    tol=1e-6,
                )
            eigenvalues = (
                eigenvectors * (symmetric @ eigenvectors)
            ).sum(dim=0)
            eigenvalues = eigenvalues.sort().values
        else:
            epsilon = torch.finfo(symmetric.dtype).eps
            diagonal = torch.arange(
                sample_count,
                device=symmetric.device,
                dtype=symmetric.dtype,
            )
            regularized = symmetric + torch.diag(diagonal * epsilon)
            eigenvalues = torch.linalg.eigvalsh(regularized)
    elif solver == "dense":
        eigenvalues = torch.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    else:
        raise ValueError(f"unknown CaSpeR eigensolver: {solver}")
    return eigenvalues[: n + 1].sum() - eigenvalues[n + 1]
