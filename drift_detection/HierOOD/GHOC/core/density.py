from __future__ import annotations

import math
from typing import Any

import torch
from scipy.special import gammaln, ive


def build_leaf_to_ancestor_mask(hierarchy, leaf_class_names: list[str]) -> torch.Tensor:
    n_leaves = len(leaf_class_names)
    n_nodes = len(hierarchy.id_node_list)
    mask = torch.zeros((n_leaves, n_nodes), dtype=torch.bool)
    for leaf_idx, leaf_name in enumerate(leaf_class_names):
        mask[leaf_idx, hierarchy.id_node_list.index(leaf_name)] = True
        for anc in hierarchy.node_ancestors[leaf_name]:
            mask[leaf_idx, anc] = True
    return mask


def fit_node_distributions(
    features: torch.Tensor,
    leaf_targets: torch.Tensor,
    hierarchy,
    leaf_class_names: list[str],
    covariance_type: str = "diag",
    eps: float = 1e-6,
    covariance_shrinkage: float = 0.0,
):
    covariance_type = covariance_type.lower()
    covariance_shrinkage = float(covariance_shrinkage)
    if covariance_shrinkage < 0.0 or covariance_shrinkage > 1.0:
        raise ValueError("covariance_shrinkage must satisfy 0 <= value <= 1")
    mask = build_leaf_to_ancestor_mask(hierarchy, leaf_class_names)
    n_nodes = mask.shape[1]
    feat_dim = features.shape[1]
    means = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    mean_directions = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    variances = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    covariance_matrices = None
    shared_covariance = None
    counts = torch.zeros((n_nodes,), dtype=torch.long)
    node_features_by_idx: list[torch.Tensor] = []

    if covariance_type in {"full", "shared_full"}:
        covariance_matrices = torch.zeros((n_nodes, feat_dim, feat_dim), dtype=features.dtype)

    for node_idx in range(n_nodes):
        member_leaf_mask = mask[:, node_idx]
        member_leaf_indices = torch.nonzero(member_leaf_mask, as_tuple=False).squeeze(1)
        sample_mask = torch.isin(leaf_targets, member_leaf_indices)
        node_features = features[sample_mask]
        node_features_by_idx.append(node_features)
        counts[node_idx] = node_features.shape[0]
        if node_features.shape[0] == 0:
            variances[node_idx] = torch.full((feat_dim,), eps, dtype=features.dtype)
            if covariance_matrices is not None:
                covariance_matrices[node_idx] = torch.eye(feat_dim, dtype=features.dtype) * eps
            continue
        means[node_idx] = node_features.mean(dim=0)
        mean_directions[node_idx] = torch.nn.functional.normalize(
            means[node_idx].unsqueeze(0),
            dim=-1,
            eps=eps,
        ).squeeze(0)
        variances[node_idx] = _fit_diag_variance(node_features, eps)
        if covariance_matrices is not None:
            covariance_matrices[node_idx] = _fit_full_covariance(
                node_features,
                means[node_idx],
                eps,
                covariance_shrinkage,
            )

    if covariance_type == "shared_full":
        shared_covariance = _fit_depth_shared_full_covariance(
            node_features_by_idx,
            means,
            hierarchy,
            feat_dim,
            features.dtype,
            features.device,
            eps,
            covariance_shrinkage,
        )
        covariance_matrices = None
    if covariance_type == "depth_isotropic":
        variances = _fit_depth_shared_isotropic_variances(
            node_features_by_idx,
            means,
            hierarchy,
            feat_dim,
            features.dtype,
            features.device,
            eps,
        )
        covariance_type = "diag"

    return {
        "node_names": hierarchy.id_node_list,
        "node_to_index": {name: idx for idx, name in enumerate(hierarchy.id_node_list)},
        "feature_dim": feat_dim,
        "covariance_type": covariance_type,
        "covariance_shrinkage": covariance_shrinkage,
        "means": means,
        "mean_directions": mean_directions,
        "variances": variances,
        "covariance_matrices": covariance_matrices,
        "shared_covariance": shared_covariance,
        "counts": counts,
        "leaf_to_ancestor_mask": mask,
    }


def fit_diagonal_gaussians(
    features: torch.Tensor,
    leaf_targets: torch.Tensor,
    hierarchy,
    leaf_class_names: list[str],
    eps: float = 1e-6,
):
    return fit_node_distributions(
        features,
        leaf_targets,
        hierarchy,
        leaf_class_names,
        covariance_type="diag",
        eps=eps,
    )


def estimate_vmf_kappas(
    means: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    dimension = means.shape[1]
    resultant = means.norm(dim=1).clamp(min=eps, max=1.0 - 1e-3)
    return (
        resultant
        * (dimension - resultant.square())
        / (1.0 - resultant.square())
    )


def vmf_log_normalizer(kappa: torch.Tensor, dimension: int) -> torch.Tensor:
    flat_kappa = kappa.detach().double().cpu().reshape(-1)
    order = dimension / 2.0 - 1.0
    values = []
    uniform_log_c = float(
        gammaln(dimension / 2.0)
        - (dimension / 2.0) * math.log(2.0 * math.pi)
    )
    for value in flat_kappa.tolist():
        if value <= 1e-12:
            values.append(uniform_log_c)
            continue
        scaled_bessel = float(ive(order, value))
        if scaled_bessel <= 0.0 or not math.isfinite(scaled_bessel):
            log_bessel = (
                order * math.log(value / 2.0)
                - float(gammaln(order + 1.0))
                + math.log1p(value * value / (4.0 * (order + 1.0)))
            )
        else:
            log_bessel = math.log(scaled_bessel) + abs(value)
        values.append(
            order * math.log(value)
            - (dimension / 2.0) * math.log(2.0 * math.pi)
            - log_bessel
        )
    return torch.tensor(
        values,
        dtype=kappa.dtype,
        device=kappa.device,
    ).reshape(kappa.shape)


def vmf_logpdf(
    features: torch.Tensor,
    mean_directions: torch.Tensor,
    kappas: torch.Tensor,
) -> torch.Tensor:
    normalized_features = torch.nn.functional.normalize(features, dim=1, eps=1e-12)
    normalized_means = torch.nn.functional.normalize(
        mean_directions,
        dim=1,
        eps=1e-12,
    )
    log_c = vmf_log_normalizer(kappas, features.shape[1])
    return (
        normalized_features @ normalized_means.transpose(0, 1)
        * kappas.unsqueeze(0)
        + log_c.unsqueeze(0)
    )


def vmf_bump(
    features: torch.Tensor,
    mean_directions: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    normalized_features = torch.nn.functional.normalize(features, dim=1, eps=1e-12)
    normalized_means = torch.nn.functional.normalize(
        mean_directions,
        dim=1,
        eps=1e-12,
    )
    cosine = normalized_features @ normalized_means.transpose(0, 1)
    return torch.exp(float(kappa) * (cosine - 1.0)).clamp(max=1.0)


def vmf_bump_integrals(
    base_direction: torch.Tensor,
    base_kappa: torch.Tensor,
    child_directions: torch.Tensor,
    mask_kappa: float,
) -> torch.Tensor:
    base_direction = torch.nn.functional.normalize(
        base_direction,
        dim=0,
        eps=1e-12,
    )
    child_directions = torch.nn.functional.normalize(
        child_directions,
        dim=1,
        eps=1e-12,
    )
    combined = (
        base_kappa * base_direction.unsqueeze(0)
        + float(mask_kappa) * child_directions
    )
    combined_kappa = combined.norm(dim=1)
    base_log_c = vmf_log_normalizer(
        base_kappa.reshape(1),
        base_direction.numel(),
    )[0]
    combined_log_c = vmf_log_normalizer(
        combined_kappa,
        base_direction.numel(),
    )
    return torch.exp(
        -float(mask_kappa) + base_log_c - combined_log_c
    ).clamp(min=0.0, max=1.0)


def score_nodes(
    features: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    mean_directions: torch.Tensor | None = None,
    covariance_type: str = "diag",
    score_type: str = "gaussian_loglik",
    kappa: float = 20.0,
) -> torch.Tensor:
    covariance_type = covariance_type.lower()
    score_type = score_type.lower()
    diff = features[:, None, :] - means[None, :, :]

    if score_type == "vmf":
        if mean_directions is None:
            mean_directions = torch.nn.functional.normalize(means, dim=-1, eps=1e-12)
        normalized_features = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
        normalized_means = torch.nn.functional.normalize(mean_directions, dim=-1, eps=1e-12)
        scores = float(kappa) * torch.matmul(normalized_features, normalized_means.transpose(0, 1))
    else:
        maha, log_det = _gaussian_terms(
            diff,
            variances=variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
        )
        if score_type == "gaussian_loglik":
            const = means.shape[-1] * math.log(2.0 * math.pi)
            scores = -0.5 * (maha + log_det[None, :] + const)
        elif score_type == "raw_gaussian":
            const = means.shape[-1] * math.log(2.0 * math.pi)
            scores = torch.exp(-0.5 * (maha + log_det[None, :] + const))
        elif score_type == "mahalanobis":
            scores = -1.0 * maha
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")

    return scores


def gaussian_logpdf(
    features: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    covariance_type: str = "diag",
    node_indices: list[int] | torch.Tensor | None = None,
    covariance_scale: float = 1.0,
) -> torch.Tensor:
    covariance_scale = float(covariance_scale)
    if covariance_scale <= 0.0:
        raise ValueError("covariance_scale must be positive")
    selected = _select_gaussian_params(
        means,
        variances,
        covariance_matrices,
        shared_covariance,
        node_indices,
    )
    selected_means, selected_variances, selected_covariances, selected_shared = selected
    if selected_variances is not None:
        selected_variances = selected_variances * covariance_scale
    if selected_covariances is not None:
        selected_covariances = selected_covariances * covariance_scale
    if selected_shared is not None:
        selected_shared = selected_shared * covariance_scale
    diff = features[:, None, :] - selected_means[None, :, :]
    maha, log_det = _gaussian_terms(
        diff,
        variances=selected_variances,
        covariance_matrices=selected_covariances,
        shared_covariance=selected_shared,
        covariance_type=covariance_type.lower(),
    )
    const = selected_means.shape[-1] * math.log(2.0 * math.pi)
    return -0.5 * (maha + log_det[None, :] + const)


def gaussian_bump(
    features: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    covariance_type: str = "diag",
    node_indices: list[int] | torch.Tensor | None = None,
    covariance_scale: float = 1.0,
) -> torch.Tensor:
    covariance_scale = float(covariance_scale)
    if covariance_scale <= 0.0:
        raise ValueError("covariance_scale must be positive")
    selected = _select_gaussian_params(
        means,
        variances,
        covariance_matrices,
        shared_covariance,
        node_indices,
    )
    selected_means, selected_variances, selected_covariances, selected_shared = selected
    if selected_variances is not None:
        selected_variances = selected_variances * covariance_scale
    if selected_covariances is not None:
        selected_covariances = selected_covariances * covariance_scale
    if selected_shared is not None:
        selected_shared = selected_shared * covariance_scale
    diff = features[:, None, :] - selected_means[None, :, :]
    maha, _ = _gaussian_terms(
        diff,
        variances=selected_variances,
        covariance_matrices=selected_covariances,
        shared_covariance=selected_shared,
        covariance_type=covariance_type.lower(),
    )
    return torch.exp(-0.5 * maha).clamp(max=1.0)


def gaussian_bump_integrals(
    parent_idx: int,
    child_indices: list[int],
    means: torch.Tensor,
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    covariance_type: str = "diag",
    parent_covariance_scale: float = 1.0,
    child_covariance_scale: float = 1.0,
) -> torch.Tensor:
    if not child_indices:
        return torch.empty((0,), dtype=means.dtype, device=means.device)

    covariance_type = covariance_type.lower()
    parent_covariance_scale = float(parent_covariance_scale)
    if parent_covariance_scale <= 0.0:
        raise ValueError("parent_covariance_scale must be positive")
    child_covariance_scale = float(child_covariance_scale)
    if child_covariance_scale <= 0.0:
        raise ValueError("child_covariance_scale must be positive")
    parent_mean = means[parent_idx]
    child_means = means[child_indices]

    if covariance_type == "diag":
        if variances is None:
            raise ValueError("variances are required for diagonal bump integrals")
        parent_var = variances[parent_idx] * parent_covariance_scale
        child_var = variances[child_indices] * child_covariance_scale
        total_var = parent_var.unsqueeze(0) + child_var
        delta = parent_mean.unsqueeze(0) - child_means
        log_r = (
            0.5 * torch.log(child_var).sum(dim=-1)
            - 0.5 * torch.log(total_var).sum(dim=-1)
            - 0.5 * (delta.pow(2) / total_var).sum(dim=-1)
        )
        return torch.exp(log_r).clamp(min=0.0, max=1.0)

    parent_cov = (
        _node_covariance(parent_idx, covariance_matrices, shared_covariance, covariance_type)
        * parent_covariance_scale
    )
    child_cov = (
        _node_covariances(child_indices, covariance_matrices, shared_covariance, covariance_type)
        * child_covariance_scale
    )
    total_cov = child_cov + parent_cov.unsqueeze(0)
    delta = parent_mean.unsqueeze(0) - child_means

    sign_child, logdet_child = torch.linalg.slogdet(child_cov)
    sign_total, logdet_total = torch.linalg.slogdet(total_cov)
    if not torch.all(sign_child > 0) or not torch.all(sign_total > 0):
        raise ValueError("covariance matrices must be positive definite for bump integrals")
    inv_total = torch.linalg.inv(total_cov)
    maha = torch.einsum("nd,nde,ne->n", delta, inv_total, delta)
    log_r = 0.5 * logdet_child - 0.5 * logdet_total - 0.5 * maha
    return torch.exp(log_r).clamp(min=0.0, max=1.0)


def gaussian_covariance_matrices(
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    covariance_type: str = "diag",
    node_indices: list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    covariance_type = covariance_type.lower()
    if covariance_type == "diag":
        if variances is None:
            raise ValueError("variances are required for diagonal covariance")
        selected = variances if node_indices is None else variances[node_indices]
        return torch.diag_embed(selected)
    if node_indices is None:
        if covariance_type == "full":
            if covariance_matrices is None:
                raise ValueError("covariance_matrices are required for full covariance")
            return covariance_matrices
        if shared_covariance is None:
            raise ValueError("shared_covariance is required for shared_full covariance")
        return shared_covariance
    return _node_covariances(
        list(node_indices),
        covariance_matrices,
        shared_covariance,
        covariance_type,
    )


def gaussian_logpdf_from_covariance(
    features: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    diff = features - mean.unsqueeze(0)
    sign, logdet = torch.linalg.slogdet(covariance)
    if not bool(sign.item() > 0):
        raise ValueError("covariance must be positive definite")
    solution = torch.linalg.solve(covariance, diff.transpose(0, 1)).transpose(0, 1)
    maha = torch.sum(diff * solution, dim=1)
    const = mean.numel() * math.log(2.0 * math.pi)
    return -0.5 * (maha + logdet + const)


def gaussian_bump_integrals_from_covariance(
    base_mean: torch.Tensor,
    base_covariance: torch.Tensor,
    child_means: torch.Tensor,
    child_covariances: torch.Tensor,
) -> torch.Tensor:
    total_covariance = child_covariances + base_covariance.unsqueeze(0)
    delta = base_mean.unsqueeze(0) - child_means
    sign_child, logdet_child = torch.linalg.slogdet(child_covariances)
    sign_total, logdet_total = torch.linalg.slogdet(total_covariance)
    if not torch.all(sign_child > 0) or not torch.all(sign_total > 0):
        raise ValueError("covariance matrices must be positive definite for bump integrals")
    solution = torch.linalg.solve(total_covariance, delta.unsqueeze(-1)).squeeze(-1)
    maha = torch.sum(delta * solution, dim=1)
    log_r = 0.5 * logdet_child - 0.5 * logdet_total - 0.5 * maha
    return torch.exp(log_r).clamp(min=0.0, max=1.0)


def _select_gaussian_params(
    means: torch.Tensor,
    variances: torch.Tensor | None,
    covariance_matrices: torch.Tensor | None,
    shared_covariance: torch.Tensor | None,
    node_indices: list[int] | torch.Tensor | None,
):
    if node_indices is None:
        return means, variances, covariance_matrices, shared_covariance
    return (
        means[node_indices],
        variances[node_indices] if variances is not None else None,
        covariance_matrices[node_indices] if covariance_matrices is not None else None,
        shared_covariance[node_indices]
        if shared_covariance is not None and shared_covariance.dim() == 3
        else shared_covariance,
    )


def _node_covariance(
    node_idx: int,
    covariance_matrices: torch.Tensor | None,
    shared_covariance: torch.Tensor | None,
    covariance_type: str,
) -> torch.Tensor:
    if covariance_type == "full":
        if covariance_matrices is None:
            raise ValueError("covariance_matrices are required for full covariance")
        return covariance_matrices[node_idx]
    if covariance_type == "shared_full":
        if shared_covariance is None:
            raise ValueError("shared_covariance is required for shared_full covariance")
        return shared_covariance[node_idx] if shared_covariance.dim() == 3 else shared_covariance
    raise ValueError(f"Unsupported full covariance type: {covariance_type}")


def _node_covariances(
    node_indices: list[int],
    covariance_matrices: torch.Tensor | None,
    shared_covariance: torch.Tensor | None,
    covariance_type: str,
) -> torch.Tensor:
    if covariance_type == "full":
        if covariance_matrices is None:
            raise ValueError("covariance_matrices are required for full covariance")
        return covariance_matrices[node_indices]
    if covariance_type == "shared_full":
        if shared_covariance is None:
            raise ValueError("shared_covariance is required for shared_full covariance")
        if shared_covariance.dim() == 3:
            return shared_covariance[node_indices]
        return shared_covariance.unsqueeze(0).expand(len(node_indices), -1, -1)
    raise ValueError(f"Unsupported full covariance type: {covariance_type}")


def _fit_diag_variance(node_features: torch.Tensor, eps: float) -> torch.Tensor:
    feat_dim = node_features.shape[1]
    if node_features.shape[0] <= 1:
        return torch.full((feat_dim,), eps, dtype=node_features.dtype)
    return node_features.var(dim=0, unbiased=False) + eps


def _shrink_covariance(covariance: torch.Tensor, shrinkage: float) -> torch.Tensor:
    if shrinkage <= 0.0:
        return covariance
    diagonal = torch.diag_embed(torch.diagonal(covariance))
    return (1.0 - shrinkage) * covariance + shrinkage * diagonal


def _fit_full_covariance(
    node_features: torch.Tensor,
    mean: torch.Tensor,
    eps: float,
    covariance_shrinkage: float = 0.0,
) -> torch.Tensor:
    feat_dim = node_features.shape[1]
    eye = torch.eye(feat_dim, dtype=node_features.dtype, device=node_features.device)
    if node_features.shape[0] <= 1:
        return eye * eps
    centered = node_features - mean.unsqueeze(0)
    cov = centered.transpose(0, 1).matmul(centered) / node_features.shape[0]
    cov = _shrink_covariance(cov, covariance_shrinkage)
    return cov + eye * eps


def _fit_depth_shared_full_covariance(
    node_features_by_idx: list[torch.Tensor],
    means: torch.Tensor,
    hierarchy,
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    covariance_shrinkage: float = 0.0,
) -> torch.Tensor:
    eye = torch.eye(feat_dim, dtype=dtype, device=device)
    n_nodes = len(node_features_by_idx)
    covariances = torch.empty((n_nodes, feat_dim, feat_dim), dtype=dtype, device=device)
    depth_to_nodes: dict[int, list[int]] = {}

    for node_idx, node_name in enumerate(hierarchy.id_node_list):
        depth = len(hierarchy.node_ancestors[node_name])
        depth_to_nodes.setdefault(depth, []).append(node_idx)

    for node_indices in depth_to_nodes.values():
        total_count = 0
        scatter = torch.zeros((feat_dim, feat_dim), dtype=dtype, device=device)
        for node_idx in node_indices:
            node_features = node_features_by_idx[node_idx]
            if node_features.shape[0] == 0:
                continue
            centered = node_features - means[node_idx].unsqueeze(0)
            scatter += centered.transpose(0, 1).matmul(centered)
            total_count += node_features.shape[0]

        if total_count == 0:
            shared_cov = eye * eps
        else:
            shared_cov = scatter / total_count
            shared_cov = _shrink_covariance(shared_cov, covariance_shrinkage)
            shared_cov = shared_cov + eye * eps

        for node_idx in node_indices:
            covariances[node_idx] = shared_cov

    return covariances


def _fit_depth_shared_isotropic_variances(
    node_features_by_idx: list[torch.Tensor],
    means: torch.Tensor,
    hierarchy,
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    variances = torch.empty(
        (len(node_features_by_idx), feat_dim),
        dtype=dtype,
        device=device,
    )
    depth_to_nodes: dict[int, list[int]] = {}
    for node_idx, node_name in enumerate(hierarchy.id_node_list):
        depth = len(hierarchy.node_ancestors[node_name])
        depth_to_nodes.setdefault(depth, []).append(node_idx)

    for node_indices in depth_to_nodes.values():
        total_count = 0
        squared_residual_sum = torch.zeros((), dtype=dtype, device=device)
        for node_idx in node_indices:
            node_features = node_features_by_idx[node_idx]
            if node_features.shape[0] == 0:
                continue
            centered = node_features - means[node_idx].unsqueeze(0)
            squared_residual_sum += centered.square().sum()
            total_count += node_features.shape[0]
        scalar_variance = (
            squared_residual_sum / (total_count * feat_dim) + eps
            if total_count > 0
            else torch.tensor(eps, dtype=dtype, device=device)
        )
        for node_idx in node_indices:
            variances[node_idx].fill_(scalar_variance)
    return variances


def _gaussian_terms(
    diff: torch.Tensor,
    variances: torch.Tensor | None,
    covariance_matrices: torch.Tensor | None,
    shared_covariance: torch.Tensor | None,
    covariance_type: str,
):
    if covariance_type == "diag":
        if variances is None:
            raise ValueError("variances are required for diagonal covariance scoring")
        log_det = torch.log(variances).sum(dim=-1)
        maha = (diff.pow(2) / variances[None, :, :]).sum(dim=-1)
        return maha, log_det

    if covariance_type == "full":
        if covariance_matrices is None:
            raise ValueError("covariance_matrices are required for full covariance scoring")
        inv_cov = torch.linalg.inv(covariance_matrices)
        sign, logabsdet = torch.linalg.slogdet(covariance_matrices)
        if not torch.all(sign > 0):
            raise ValueError("full covariance matrices must be positive definite")
        maha = torch.einsum("bnd,nde,bne->bn", diff, inv_cov, diff)
        return maha, logabsdet

    if covariance_type == "shared_full":
        if shared_covariance is None:
            raise ValueError("shared_covariance is required for shared_full scoring")
        if shared_covariance.dim() == 3:
            inv_cov = torch.linalg.inv(shared_covariance)
            sign, logabsdet = torch.linalg.slogdet(shared_covariance)
            if not torch.all(sign > 0):
                raise ValueError("depth-shared full covariance matrices must be positive definite")
            maha = torch.einsum("bnd,nde,bne->bn", diff, inv_cov, diff)
            return maha, logabsdet
        inv_cov = torch.linalg.inv(shared_covariance)
        sign, logabsdet = torch.linalg.slogdet(shared_covariance)
        if not bool(sign.item() > 0):
            raise ValueError("shared full covariance must be positive definite")
        projected = torch.matmul(diff, inv_cov)
        maha = (projected * diff).sum(dim=-1)
        log_det = torch.full(
            (diff.shape[1],),
            logabsdet,
            dtype=diff.dtype,
            device=diff.device,
        )
        return maha, log_det

    raise ValueError(f"Unsupported covariance_type: {covariance_type}")
