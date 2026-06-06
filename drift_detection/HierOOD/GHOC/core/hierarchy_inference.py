from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from core.density import gaussian_bump, gaussian_bump_integrals, gaussian_logpdf, score_nodes
from libs.utils.hierarchy_utils import expected_hdist, get_hdist_matrix


def build_depth_maps(hierarchy):
    nodes_by_depth = defaultdict(list)
    for idx, name in enumerate(hierarchy.id_node_list):
        depth = len(hierarchy.node_ancestors[name])
        nodes_by_depth[depth].append(idx)
    return {depth: sorted(indices) for depth, indices in nodes_by_depth.items()}


def _normalize_depthwise_temperature(temperature, depths: list[int]) -> dict[int, float]:
    if isinstance(temperature, (list, tuple)):
        if len(temperature) != len(depths):
            raise ValueError(
                f"temperature vector length {len(temperature)} does not match "
                f"number of non-root depths {len(depths)}"
            )
        return {depth: max(float(temp), 1e-8) for depth, temp in zip(depths, temperature)}
    return {depth: max(float(temperature), 1e-8) for depth in depths}


def compute_depth_probs(node_scores: torch.Tensor, nodes_by_depth: dict[int, list[int]], temperature: float | list[float] = 1.0):
    depth_probs = {}
    non_root_depths = sorted(depth for depth in nodes_by_depth.keys() if depth != 0)
    temp_by_depth = _normalize_depthwise_temperature(temperature, non_root_depths)
    for depth, indices in nodes_by_depth.items():
        scaled_scores = node_scores[:, indices]
        if depth != 0:
            scaled_scores = scaled_scores / temp_by_depth[depth]
        depth_probs[depth] = torch.softmax(scaled_scores, dim=-1)
    return depth_probs


def _local_probabilities_for_node(
    parent_name: str,
    hierarchy,
    depth_probs: dict[int, torch.Tensor],
    nodes_by_depth: dict[int, list[int]],
):
    children = hierarchy.parent2children.get(parent_name, [])
    if not children:
        return None

    child_indices = [hierarchy.id_node_list.index(child) for child in children]
    child_depth = len(hierarchy.node_ancestors[children[0]])
    depth_indices = nodes_by_depth[child_depth]
    depth_index_map = {node_idx: pos for pos, node_idx in enumerate(depth_indices)}
    child_depth_probs = torch.stack(
        [depth_probs[child_depth][:, depth_index_map[idx]] for idx in child_indices],
        dim=1,
    )
    child_group_sum = child_depth_probs.sum(dim=1, keepdim=True)
    p_comp = 1.0 - child_group_sum.squeeze(1)
    p_comp = torch.clamp(p_comp, min=0.0)
    eps = 1e-12
    child_local_probs = child_depth_probs / child_group_sum.clamp_min(eps)
    entropy = -torch.sum(child_local_probs * torch.log(child_local_probs + eps), dim=1)
    ood_mass = p_comp + entropy
    ood_mass = torch.clamp(ood_mass, min=0.0)

    # Stay in probability space: use depth-wise child probabilities and
    # the derived OOD mass, then renormalize locally within siblings.
    local_mass = torch.cat([child_depth_probs, ood_mass.unsqueeze(1)], dim=1)
    local_normalizer = local_mass.sum(dim=1, keepdim=True).clamp_min(eps)
    local_probs = local_mass / local_normalizer
    return {
        "child_indices": child_indices,
        "child_depth_probs": child_depth_probs,
        "child_local_probs": child_local_probs,
        "child_probs": local_probs[:, :-1],
        "ood_prob": local_probs[:, -1],
        "ood_score": ood_mass,
        "p_comp": p_comp,
        "entropy": entropy,
    }


def _cgm_enabled(cgm_cfg: dict[str, Any] | None) -> bool:
    return bool(cgm_cfg and cgm_cfg.get("enabled", False))


def _is_unit_temperature(temperature) -> bool:
    if isinstance(temperature, (list, tuple)):
        return all(abs(float(temp) - 1.0) <= 1e-8 for temp in temperature)
    return abs(float(temperature) - 1.0) <= 1e-8


def _validate_cgm_cfg(cgm_cfg: dict[str, Any], score_type: str, temperature):
    mask_type = cgm_cfg.get("mask_type", "sum").lower()
    if mask_type != "sum":
        raise ValueError(f"Unsupported CGM mask_type: {mask_type}")
    lam = float(cgm_cfg.get("lambda", 0.9))
    if lam < 0.0 or lam >= 1.0:
        raise ValueError("CGM lambda must satisfy 0 <= lambda < 1")
    child_weight = cgm_cfg.get("child_weight", "uniform").lower()
    if child_weight not in {"uniform", "count"}:
        raise ValueError(f"Unsupported CGM child_weight: {child_weight}")
    if score_type.lower() != "gaussian_loglik":
        raise ValueError("CGM inference requires inference.score_type: gaussian_loglik")
    if not _is_unit_temperature(temperature):
        raise ValueError("CGM base inference requires temperature 1.0")


def _cgm_child_weights(
    child_indices: list[int],
    density_payload: dict[str, Any],
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)

    counts = density_payload.get("counts")
    if counts is None:
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)
    weights = counts[child_indices].to(device=device, dtype=dtype)
    total = weights.sum()
    if not bool(total.item() > 0):
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)
    return weights / total


def _cgm_local_probabilities_for_node(
    parent_name: str,
    features: torch.Tensor,
    hierarchy,
    node_logpdf: torch.Tensor,
    density_payload: dict[str, Any],
    cgm_cfg: dict[str, Any],
):
    children = hierarchy.parent2children.get(parent_name, [])
    if not children:
        return None

    eps = float(cgm_cfg.get("eps", 1e-12))
    lam = float(cgm_cfg.get("lambda", 0.9))
    child_weight_mode = cgm_cfg.get("child_weight", "uniform").lower()
    normalize_ood_pdf = bool(cgm_cfg.get("normalize_ood_pdf", True))

    parent_idx = hierarchy.id_node_list.index(parent_name)
    child_indices = [hierarchy.id_node_list.index(child) for child in children]
    covariance_type = density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag"))
    means = density_payload["means"]
    variances = density_payload.get("variances")
    covariance_matrices = density_payload.get("covariance_matrices")
    shared_covariance = density_payload.get("shared_covariance")

    child_logpdf = node_logpdf[:, child_indices]
    parent_logpdf = node_logpdf[:, parent_idx]
    child_bumps = gaussian_bump(
        features,
        means,
        variances,
        covariance_matrices=covariance_matrices,
        shared_covariance=shared_covariance,
        covariance_type=covariance_type,
        node_indices=child_indices,
    )
    child_weights = _cgm_child_weights(
        child_indices,
        density_payload,
        child_weight_mode,
        dtype=features.dtype,
        device=features.device,
    )
    weighted_bump = (child_bumps * child_weights.unsqueeze(0)).sum(dim=1)
    mask = 1.0 - lam * weighted_bump
    log_mask = torch.log(mask.clamp_min(eps))

    if normalize_ood_pdf:
        bump_integrals = gaussian_bump_integrals(
            parent_idx,
            child_indices,
            means,
            variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
        )
        normalizer = 1.0 - lam * torch.sum(child_weights * bump_integrals)
        log_normalizer = torch.log(normalizer.clamp_min(eps))
    else:
        bump_integrals = torch.empty((len(child_indices),), dtype=features.dtype, device=features.device)
        normalizer = torch.ones((), dtype=features.dtype, device=features.device)
        log_normalizer = torch.zeros((), dtype=features.dtype, device=features.device)

    ood_logpdf = parent_logpdf + log_mask - log_normalizer
    local_logits = torch.cat([child_logpdf, ood_logpdf.unsqueeze(1)], dim=1)
    local_probs = torch.softmax(local_logits, dim=1)

    return {
        "child_indices": child_indices,
        "child_probs": local_probs[:, :-1],
        "ood_prob": local_probs[:, -1],
        "child_logpdf": child_logpdf,
        "parent_logpdf": parent_logpdf,
        "child_bumps": child_bumps,
        "child_weights": child_weights,
        "weighted_bump": weighted_bump,
        "mask": mask,
        "log_mask": log_mask,
        "bump_integrals": bump_integrals,
        "normalizer": normalizer,
        "ood_logpdf": ood_logpdf,
    }


def hierarchical_node_probabilities(
    features: torch.Tensor | None,
    hierarchy,
    density_payload: dict[str, Any],
    score_type: str = "gaussian_loglik",
    temperature: float = 1.0,
    kappa: float = 20.0,
    include_debug: bool = True,
    node_scores: torch.Tensor | None = None,
    cgm_cfg: dict[str, Any] | None = None,
):
    use_cgm = _cgm_enabled(cgm_cfg)
    if use_cgm:
        if features is None:
            raise ValueError("features are required for CGM inference")
        _validate_cgm_cfg(cgm_cfg or {}, score_type, temperature)
        node_scores = gaussian_logpdf(
            features,
            density_payload["means"],
            density_payload.get("variances"),
            covariance_matrices=density_payload.get("covariance_matrices"),
            shared_covariance=density_payload.get("shared_covariance"),
            covariance_type=density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag")),
        )
    elif node_scores is None:
        if features is None:
            raise ValueError("features are required when node_scores are not provided")
        node_scores = score_nodes(
            features,
            density_payload["means"],
            density_payload.get("variances"),
            covariance_matrices=density_payload.get("covariance_matrices"),
            shared_covariance=density_payload.get("shared_covariance"),
            mean_directions=density_payload.get("mean_directions"),
            covariance_type=density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag")),
            score_type=score_type,
            kappa=kappa,
        )
    nodes_by_depth = build_depth_maps(hierarchy)
    depth_probs = None if use_cgm else compute_depth_probs(node_scores, nodes_by_depth, temperature=temperature)

    n_samples = node_scores.shape[0]
    n_nodes = len(hierarchy.id_node_list)
    final_probs = torch.zeros((n_samples, n_nodes), dtype=node_scores.dtype, device=node_scores.device)
    local_info = {}

    for parent in hierarchy.parent2children:
        if use_cgm:
            local = _cgm_local_probabilities_for_node(
                parent,
                features,
                hierarchy,
                node_scores,
                density_payload,
                cgm_cfg or {},
            )
        else:
            local = _local_probabilities_for_node(
                parent,
                hierarchy,
                depth_probs,
                nodes_by_depth,
            )
        if local is not None:
            local_info[parent] = local

    def recurse(node_name: str, incoming_prob: torch.Tensor):
        node_idx = hierarchy.id_node_list.index(node_name)
        local = local_info.get(node_name)
        if local is None:
            final_probs[:, node_idx] += incoming_prob
            return
        final_probs[:, node_idx] += incoming_prob * local["ood_prob"]
        for child_idx, child_node_idx in enumerate(local["child_indices"]):
            child_name = hierarchy.id_node_list[child_node_idx]
            child_prob = incoming_prob * local["child_probs"][:, child_idx]
            recurse(child_name, child_prob)

    root_name = "root"
    recurse(root_name, torch.ones((n_samples,), dtype=node_scores.dtype, device=node_scores.device))
    final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    debug = {
        "temperature": temperature,
        "cgm": cgm_cfg if use_cgm else {"enabled": False},
    }
    if include_debug:
        if use_cgm:
            debug.update(
                {
                    "node_logpdf": node_scores.cpu(),
                    "local_info": {
                        name: {
                            "child_indices": payload["child_indices"],
                            "child_probs": payload["child_probs"].cpu(),
                            "ood_prob": payload["ood_prob"].cpu(),
                            "child_logpdf": payload["child_logpdf"].cpu(),
                            "parent_logpdf": payload["parent_logpdf"].cpu(),
                            "child_bumps": payload["child_bumps"].cpu(),
                            "child_weights": payload["child_weights"].cpu(),
                            "weighted_bump": payload["weighted_bump"].cpu(),
                            "mask": payload["mask"].cpu(),
                            "log_mask": payload["log_mask"].cpu(),
                            "bump_integrals": payload["bump_integrals"].cpu(),
                            "normalizer": payload["normalizer"].detach().cpu(),
                            "ood_logpdf": payload["ood_logpdf"].cpu(),
                        }
                        for name, payload in local_info.items()
                    },
                }
            )
        else:
            debug.update(
                {
                    "node_scores": node_scores.cpu(),
                    "depth_probs": {depth: probs.cpu() for depth, probs in depth_probs.items()},
                    "local_info": {
                        name: {
                            "child_indices": payload["child_indices"],
                            "child_depth_probs": payload["child_depth_probs"].cpu(),
                            "child_local_probs": payload["child_local_probs"].cpu(),
                            "child_probs": payload["child_probs"].cpu(),
                            "ood_prob": payload["ood_prob"].cpu(),
                            "ood_score": payload["ood_score"].cpu(),
                            "p_comp": payload["p_comp"].cpu(),
                            "entropy": payload["entropy"].cpu(),
                        }
                        for name, payload in local_info.items()
                    },
                }
            )
    return final_probs, debug


def predict_from_probabilities(final_probs: torch.Tensor, hierarchy, mode: str = "argmax"):
    if mode == "argmax":
        return final_probs.argmax(dim=1)
    if mode == "min_hdist":
        hdist_mat = get_hdist_matrix(hierarchy, range(len(hierarchy.id_node_list)))
        neg_dists = -1.0 * expected_hdist(final_probs.cpu(), hdist_mat)
        return neg_dists.argmax(dim=1)
    raise ValueError(f"Unsupported prediction mode: {mode}")
