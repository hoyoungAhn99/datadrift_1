from __future__ import annotations

import math
from typing import Any

import torch


def build_leaf_to_ancestor_mask(hierarchy, leaf_class_names: list[str]) -> torch.Tensor:
    n_leaves = len(leaf_class_names)
    n_nodes = len(hierarchy.id_node_list)
    mask = torch.zeros((n_leaves, n_nodes), dtype=torch.bool)
    for leaf_idx, leaf_name in enumerate(leaf_class_names):
        mask[leaf_idx, hierarchy.id_node_list.index(leaf_name)] = True
        for anc in hierarchy.node_ancestors[leaf_name]:
            mask[leaf_idx, anc] = True
    return mask


def fit_diagonal_gaussians(
    features: torch.Tensor,
    leaf_targets: torch.Tensor,
    hierarchy,
    leaf_class_names: list[str],
    eps: float = 1e-6,
):
    mask = build_leaf_to_ancestor_mask(hierarchy, leaf_class_names)
    n_nodes = mask.shape[1]
    feat_dim = features.shape[1]
    means = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    mean_directions = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    variances = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    counts = torch.zeros((n_nodes,), dtype=torch.long)

    for node_idx in range(n_nodes):
        member_leaf_mask = mask[:, node_idx]
        member_leaf_indices = torch.nonzero(member_leaf_mask, as_tuple=False).squeeze(1)
        sample_mask = torch.isin(leaf_targets, member_leaf_indices)
        node_features = features[sample_mask]
        counts[node_idx] = node_features.shape[0]
        if node_features.shape[0] == 0:
            variances[node_idx] = torch.full((feat_dim,), eps, dtype=features.dtype)
            continue
        means[node_idx] = node_features.mean(dim=0)
        mean_directions[node_idx] = torch.nn.functional.normalize(
            means[node_idx].unsqueeze(0),
            dim=-1,
            eps=eps,
        ).squeeze(0)
        if node_features.shape[0] == 1:
            variances[node_idx] = torch.full((feat_dim,), eps, dtype=features.dtype)
        else:
            variances[node_idx] = node_features.var(dim=0, unbiased=False) + eps

    return {
        "node_names": hierarchy.id_node_list,
        "node_to_index": {name: idx for idx, name in enumerate(hierarchy.id_node_list)},
        "feature_dim": feat_dim,
        "means": means,
        "mean_directions": mean_directions,
        "variances": variances,
        "counts": counts,
        "leaf_to_ancestor_mask": mask,
    }


def score_nodes(
    features: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    mean_directions: torch.Tensor | None = None,
    score_type: str = "gaussian_loglik",
    temperature: float = 1.0,
    kappa: float = 20.0,
) -> torch.Tensor:
    score_type = score_type.lower()
    diff = features[:, None, :] - means[None, :, :]
    if score_type == "gaussian_loglik":
        log_det = torch.log(variances).sum(dim=-1)
        maha = (diff.pow(2) / variances[None, :, :]).sum(dim=-1)
        const = means.shape[-1] * math.log(2.0 * math.pi)
        scores = -0.5 * (maha + log_det[None, :] + const)
    elif score_type == "raw_gaussian":
        log_det = torch.log(variances).sum(dim=-1)
        maha = (diff.pow(2) / variances[None, :, :]).sum(dim=-1)
        const = means.shape[-1] * math.log(2.0 * math.pi)
        scores = torch.exp(-0.5 * (maha + log_det[None, :] + const))
    elif score_type == "mahalanobis":
        scores = -1.0 * (diff.pow(2) / variances[None, :, :]).sum(dim=-1)
    elif score_type == "vmf":
        if mean_directions is None:
            mean_directions = torch.nn.functional.normalize(means, dim=-1, eps=1e-12)
        normalized_features = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
        normalized_means = torch.nn.functional.normalize(mean_directions, dim=-1, eps=1e-12)
        scores = float(kappa) * torch.matmul(normalized_features, normalized_means.transpose(0, 1))
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")

    return scores / max(temperature, 1e-8)
