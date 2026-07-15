from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_ce_loss(
    image_features: torch.Tensor,
    child_features: torch.Tensor,
    target_indices: torch.Tensor,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    image_features = F.normalize(image_features.float(), dim=-1)
    child_features = F.normalize(child_features.float(), dim=-1)
    logits = image_features @ child_features.t() / float(tau)
    loss = F.cross_entropy(logits, target_indices.long().to(logits.device))
    acc = (logits.argmax(dim=1) == target_indices.to(logits.device)).float().mean()
    return loss, {"acc": float(acc.detach().cpu()), "loss": float(loss.detach().cpu())}


def ms_loss_level(
    sim_mat: torch.Tensor,
    labels: torch.Tensor,
    neg_weights: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)
    loss_list = []
    set_size_list = []

    for i in range(batch_size):
        label_i = labels[i]
        mask_pos = labels == label_i
        mask_neg = labels != label_i
        mask_pos[i] = False

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]
        current_neg_w = None if neg_weights is None else neg_weights[i][mask_neg]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        neg_keep = (neg_sim + mining_margin) > hardest_pos_sim
        pos_keep = (pos_sim - mining_margin) < hardest_neg_sim
        neg_sim = neg_sim[neg_keep]
        pos_sim = pos_sim[pos_keep]
        if current_neg_w is not None:
            current_neg_w = current_neg_w[neg_keep]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        pos_term = torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
        if current_neg_w is None:
            neg_exp = torch.exp(beta * (neg_sim - lam))
        else:
            neg_exp = current_neg_w * torch.exp(beta * (neg_sim - lam))
        neg_term = torch.log(1.0 + torch.sum(neg_exp)) / beta

        loss_list.append(pos_term + neg_term)
        set_size_list.append(torch.tensor(pos_sim.numel() + neg_sim.numel(), dtype=sim_mat.dtype, device=device))

    return torch.stack(loss_list), torch.stack(set_size_list)


def multi_similarity_loss(
    features: torch.Tensor,
    path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    sim_mat = torch.clamp(features @ features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    _, labels = torch.unique(path_labels, dim=0, return_inverse=True)
    losses, _ = ms_loss_level(
        sim_mat,
        labels,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
    )
    return losses.mean()


def get_slice_distance_weights(
    slice_labels: torch.Tensor,
    scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    matches = (slice_labels.unsqueeze(1) == slice_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = (float(slice_labels.size(1)) - shared_depth).pow(dist_pow)
    return torch.pow(float(scale), tree_dist - 1.0)


def hierarchical_ms_min_loss(
    features: torch.Tensor,
    path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    use_distance_weights: bool = False,
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    features = F.normalize(features.float(), dim=-1)
    sim_mat = torch.clamp(features @ features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    num_levels = int(path_labels.size(1))
    device = features.device

    level_losses = []
    for level in range(num_levels):
        current_path = path_labels[:, : level + 1]
        _, labels_level = torch.unique(current_path, dim=0, return_inverse=True)
        neg_weights = None
        if use_distance_weights:
            neg_weights = get_slice_distance_weights(
                current_path,
                scale=dist_scale,
                dist_pow=dist_pow,
            )
        losses_l, _ = ms_loss_level(
            sim_mat,
            labels_level,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        level_losses.append(losses_l)

    total_loss = torch.tensor(0.0, dtype=features.dtype, device=device)
    cur_min = None
    for rev_level in range(num_levels - 1, -1, -1):
        base_loss_l = level_losses[rev_level]
        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)
        cur_min = torch.min(cur_loss) if minimum_mode == "batch" else cur_loss
        k_l = torch.exp(torch.tensor(1.0 / float(rev_level + 1), dtype=features.dtype, device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / float(num_levels)
    return total_loss


def _cross_modal_ms_loss_level(
    sim_mat: torch.Tensor,
    image_labels: torch.Tensor,
    prompt_labels: torch.Tensor,
    neg_weights: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_mat.device
    num_images = sim_mat.size(0)
    loss_list = []
    set_size_list = []

    for i in range(num_images):
        mask_pos = prompt_labels == image_labels[i]
        mask_neg = ~mask_pos

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]
        current_neg_w = None if neg_weights is None else neg_weights[i][mask_neg]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        neg_keep = (neg_sim + mining_margin) > hardest_pos_sim
        pos_keep = (pos_sim - mining_margin) < hardest_neg_sim
        neg_sim = neg_sim[neg_keep]
        pos_sim = pos_sim[pos_keep]
        if current_neg_w is not None:
            current_neg_w = current_neg_w[neg_keep]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        pos_term = torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
        if current_neg_w is None:
            neg_exp = torch.exp(beta * (neg_sim - lam))
        else:
            neg_exp = current_neg_w * torch.exp(beta * (neg_sim - lam))
        neg_term = torch.log(1.0 + torch.sum(neg_exp)) / beta

        loss_list.append(pos_term + neg_term)
        set_size_list.append(torch.tensor(pos_sim.numel() + neg_sim.numel(), dtype=sim_mat.dtype, device=device))

    return torch.stack(loss_list), torch.stack(set_size_list)


def _cross_modal_distance_weights(
    image_slice_labels: torch.Tensor,
    prompt_slice_labels: torch.Tensor,
    scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    matches = (image_slice_labels.unsqueeze(1) == prompt_slice_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = (float(image_slice_labels.size(1)) - shared_depth).pow(dist_pow)
    return torch.pow(float(scale), tree_dist - 1.0)


def cross_modal_multi_similarity_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    prompt_features = F.normalize(prompt_features.float(), dim=-1)
    sim_mat = torch.clamp(image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    _, all_labels = torch.unique(
        torch.cat([image_path_labels, prompt_path_labels], dim=0),
        dim=0,
        return_inverse=True,
    )
    image_labels = all_labels[: image_path_labels.size(0)]
    prompt_labels = all_labels[image_path_labels.size(0):]
    losses, _ = _cross_modal_ms_loss_level(
        sim_mat,
        image_labels,
        prompt_labels,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
    )
    return losses.mean()


def cross_modal_hierarchical_ms_min_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    use_distance_weights: bool = False,
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    image_features = F.normalize(image_features.float(), dim=-1)
    prompt_features = F.normalize(prompt_features.float(), dim=-1)
    sim_mat = torch.clamp(image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    num_levels = int(image_path_labels.size(1))
    device = image_features.device

    level_losses = []
    for level in range(num_levels):
        image_path = image_path_labels[:, : level + 1]
        prompt_path = prompt_path_labels[:, : level + 1]
        _, all_labels = torch.unique(
            torch.cat([image_path, prompt_path], dim=0),
            dim=0,
            return_inverse=True,
        )
        image_labels = all_labels[: image_path.size(0)]
        prompt_labels = all_labels[image_path.size(0):]
        neg_weights = None
        if use_distance_weights:
            neg_weights = _cross_modal_distance_weights(
                image_path,
                prompt_path,
                scale=dist_scale,
                dist_pow=dist_pow,
            )

        losses_l, _ = _cross_modal_ms_loss_level(
            sim_mat,
            image_labels,
            prompt_labels,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        level_losses.append(losses_l)

    total_loss = torch.tensor(0.0, dtype=image_features.dtype, device=device)
    cur_min = None
    for rev_level in range(num_levels - 1, -1, -1):
        base_loss_l = level_losses[rev_level]
        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)
        cur_min = torch.min(cur_loss) if minimum_mode == "batch" else cur_loss
        k_l = torch.exp(torch.tensor(1.0 / float(rev_level + 1), dtype=image_features.dtype, device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / float(num_levels)
    return total_loss


def prompt_metric_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    loss_name: str,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    image_path_labels = image_path_labels.to(image_features.device)
    prompt_path_labels = prompt_path_labels.to(image_features.device)
    loss_name = loss_name.lower()

    if loss_name == "ms":
        loss = cross_modal_multi_similarity_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
    elif loss_name in {"hims_min", "himsmin"}:
        loss = cross_modal_hierarchical_ms_min_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
            minimum_mode=minimum_mode,
            use_distance_weights=False,
        )
    elif loss_name in {"weihims", "hims_min_wei"}:
        loss = cross_modal_hierarchical_ms_min_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
            minimum_mode=minimum_mode,
            use_distance_weights=True,
            dist_scale=dist_scale,
            dist_pow=dist_pow,
        )
    else:
        raise ValueError(f"Unsupported prompt metric loss: {loss_name}")

    return loss, {
        "pair_mode": "cross_modal_image_prompt_only",
        "metric_loss": float(loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def depthwise_prompt_metric_loss(
    image_features: torch.Tensor,
    prompt_features_by_depth: dict[int, torch.Tensor],
    image_node_labels_by_depth: torch.Tensor,
    prompt_node_labels_by_depth: dict[int, torch.Tensor],
    prompt_path_labels_by_depth: dict[int, torch.Tensor],
    loss_name: str,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    loss_name = loss_name.lower()
    if loss_name not in {"ms", "hims_min", "himsmin", "weihims", "hims_min_wei"}:
        raise ValueError(f"Unsupported depthwise prompt metric loss: {loss_name}")

    image_features = F.normalize(image_features.float(), dim=-1)
    image_node_labels_by_depth = image_node_labels_by_depth.to(image_features.device)
    use_distance_weights = loss_name in {"weihims", "hims_min_wei"}
    num_images = int(image_features.shape[0])
    device = image_features.device

    level_losses = {}
    valid_counts = {}
    for depth in sorted(prompt_features_by_depth):
        if depth <= 0:
            continue
        prompt_features = prompt_features_by_depth[depth]
        if prompt_features.numel() == 0:
            continue

        image_labels = image_node_labels_by_depth[:, depth]
        valid = image_labels >= 0
        if not bool(valid.any()):
            continue

        prompt_labels = prompt_node_labels_by_depth[depth].to(device)
        prompt_paths = prompt_path_labels_by_depth[depth].to(device)
        valid_image_features = image_features[valid]
        valid_image_labels = image_labels[valid]
        prompt_features = F.normalize(prompt_features.float(), dim=-1)
        sim_mat = torch.clamp(valid_image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)

        neg_weights = None
        if use_distance_weights:
            image_paths = image_node_labels_by_depth[valid, : depth + 1]
            neg_weights = _cross_modal_distance_weights(
                image_paths,
                prompt_paths,
                scale=dist_scale,
                dist_pow=dist_pow,
            )

        losses_l, _ = _cross_modal_ms_loss_level(
            sim_mat,
            valid_image_labels,
            prompt_labels,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        full_losses = torch.full((num_images,), float("inf"), dtype=losses_l.dtype, device=device)
        full_losses[valid] = losses_l
        level_losses[depth] = full_losses
        valid_counts[depth] = int(valid.sum().detach().cpu())

    if not level_losses:
        zero = image_features.sum() * 0.0
        return zero, {
            "pair_mode": "depthwise_global_image_prompt",
            "metric_loss": 0.0,
            "loss": 0.0,
            "valid_counts": valid_counts,
        }

    if loss_name == "ms":
        loss_terms = []
        for full_losses in level_losses.values():
            finite = torch.isfinite(full_losses)
            if bool(finite.any()):
                loss_terms.append(full_losses[finite].mean())
        loss = torch.stack(loss_terms).mean() if loss_terms else image_features.sum() * 0.0
    else:
        cur_min = torch.full((num_images,), float("inf"), dtype=image_features.dtype, device=device)
        loss_terms = []
        max_depth = max(level_losses)
        for depth in range(max_depth, 0, -1):
            if depth not in level_losses:
                continue
            base_loss = level_losses[depth]
            finite = torch.isfinite(base_loss)
            if not bool(finite.any()):
                continue
            if minimum_mode == "batch":
                depth_min = torch.min(base_loss[finite])
                cur_min[finite] = torch.minimum(base_loss[finite], depth_min.expand_as(base_loss[finite]))
            else:
                cur_min[finite] = torch.minimum(base_loss[finite], cur_min[finite])
            k_l = torch.exp(torch.tensor(1.0 / float(depth + 1), dtype=image_features.dtype, device=device))
            loss_terms.append(k_l * cur_min[finite].mean())
        loss = torch.stack(loss_terms).mean() if loss_terms else image_features.sum() * 0.0

    return loss, {
        "pair_mode": "depthwise_global_image_prompt",
        "metric_loss": float(loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "valid_counts": valid_counts,
    }


def unknown_ce_loss(
    image_features: torch.Tensor,
    candidate_features: torch.Tensor,
    target_indices: torch.Tensor,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    return positive_ce_loss(image_features, candidate_features, target_indices, tau=tau)


def unknown_regularization(
    unknown_feature: torch.Tensor,
    parent_feature: torch.Tensor,
    child_features: torch.Tensor,
    lambda_anchor: float = 0.1,
    lambda_child_sep: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    unknown_feature = F.normalize(unknown_feature.float(), dim=-1)
    parent_feature = F.normalize(parent_feature.float(), dim=-1)
    child_features = F.normalize(child_features.float(), dim=-1)

    anchor_loss = -torch.sum(unknown_feature * parent_feature)
    child_sep_loss = torch.mean(child_features @ unknown_feature)
    total = float(lambda_anchor) * anchor_loss + float(lambda_child_sep) * child_sep_loss
    return total, {
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "child_sep_loss": float(child_sep_loss.detach().cpu()),
        "regularizer": float(total.detach().cpu()),
    }
