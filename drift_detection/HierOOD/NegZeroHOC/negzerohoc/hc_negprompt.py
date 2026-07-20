from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _sibling_shell_targets(
    positive_features: torch.Tensor,
    parent_feature: torch.Tensor,
) -> torch.Tensor:
    """Return the normalized midpoint similarity to the nearest sibling."""
    positives = F.normalize(positive_features.float(), dim=-1)
    if positives.shape[0] == 1:
        parent = F.normalize(parent_feature.float(), dim=-1)
        nearest = positives @ parent
    else:
        pairwise = positives @ positives.t()
        diagonal = torch.eye(
            positives.shape[0],
            dtype=torch.bool,
            device=positives.device,
        )
        pairwise = pairwise.masked_fill(diagonal, -1.0)
        nearest = pairwise.max(dim=1).values
    return torch.sqrt(((1.0 + nearest).clamp(min=0.0, max=2.0)) / 2.0)


def _tangent_diversity(
    negatives: torch.Tensor,
    positives: torch.Tensor,
) -> torch.Tensor:
    num_prompts = int(negatives.shape[1])
    if num_prompts <= 1:
        return negatives.new_zeros(())

    positive_directions = positives.unsqueeze(1)
    projections = (negatives * positive_directions).sum(dim=-1, keepdim=True)
    residuals = negatives - projections * positive_directions
    residuals = F.normalize(residuals, dim=-1)
    similarities = residuals @ residuals.transpose(1, 2)
    off_diagonal = ~torch.eye(
        num_prompts,
        dtype=torch.bool,
        device=negatives.device,
    )
    return similarities[:, off_diagonal].square().mean()


def hierarchy_constrained_negprompt_loss(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    target_indices: torch.Tensor,
    parent_feature: torch.Tensor,
    route_levels: list[tuple[torch.Tensor, int]],
    *,
    tau: float,
    hierarchy_tau: float,
    safety_margin: float,
    lambda_hnis: float,
    lambda_safe: float,
    lambda_shell: float,
    lambda_diversity: float,
    lambda_route: float,
    lambda_balance: float,
) -> tuple[torch.Tensor, dict]:
    """Train hierarchy-local negative prompts without pseudo-OOD targets.

    Every image remains assigned to its true positive child. Image-conditioned
    terms either equalize negative responses or keep their total mass below the
    true positive. Hierarchy placement is learned entirely from frozen text
    features.
    """
    if float(tau) <= 0.0 or float(hierarchy_tau) <= 0.0:
        raise ValueError("tau and hierarchy_tau must be positive")
    if positive_features.ndim != 2:
        raise ValueError("positive_features must have shape [children, dim]")
    if negative_features.ndim != 3:
        raise ValueError("negative_features must have shape [children, prompts, dim]")
    if negative_features.shape[0] != positive_features.shape[0]:
        raise ValueError("Positive and negative child counts must match")
    if not route_levels:
        raise ValueError("route_levels must contain at least one ancestor decision")

    images = F.normalize(image_features.float(), dim=-1)
    positives = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_features.float(), dim=-1)
    negative_flat = negatives.flatten(0, 1)
    targets = target_indices.long().to(images.device)

    negative_logits = images @ negative_flat.t() / float(tau)
    negative_log_probs = F.log_softmax(negative_logits, dim=1)
    hnis_loss = -negative_log_probs.mean()
    hnis_excess = hnis_loss - math.log(negative_logits.shape[1])

    positive_logits = images @ positives.t() / float(tau)
    true_positive_logits = positive_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mass = torch.logsumexp(negative_logits, dim=1)
    safe_loss = F.softplus(
        negative_mass - true_positive_logits + float(safety_margin)
    ).mean()

    shell_targets = _sibling_shell_targets(positives, parent_feature)
    positive_per_negative = positives.unsqueeze(1).expand_as(negatives)
    negative_positive_cosines = (negatives * positive_per_negative).sum(dim=-1)
    shell_loss = (
        negative_positive_cosines - shell_targets.unsqueeze(1)
    ).square().mean()

    diversity_loss = _tangent_diversity(negatives, positives)
    center = F.normalize(negative_flat.mean(dim=0), dim=-1)

    route_losses = []
    route_correct = []
    for level_features, target_index in route_levels:
        level = F.normalize(level_features.float(), dim=-1)
        logits = (center @ level.t()) / float(hierarchy_tau)
        target = torch.tensor([int(target_index)], dtype=torch.long, device=logits.device)
        route_losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        route_correct.append(float(int(logits.argmax().item()) == int(target_index)))
    route_loss = torch.stack(route_losses).mean()

    local_logits = (center @ positives.t()) / float(hierarchy_tau)
    local_log_probs = F.log_softmax(local_logits, dim=0)
    balance_loss = -local_log_probs.mean()
    balance_excess = balance_loss - math.log(int(positives.shape[0]))

    total = (
        float(lambda_hnis) * hnis_loss
        + float(lambda_safe) * safe_loss
        + float(lambda_shell) * shell_loss
        + float(lambda_diversity) * diversity_loss
        + float(lambda_route) * route_loss
        + float(lambda_balance) * balance_loss
    )
    unknown_would_win = negative_mass > positive_logits.max(dim=1).values
    return total, {
        "loss": float(total.detach().cpu()),
        "hnis_loss": float(hnis_loss.detach().cpu()),
        "hnis_excess": float(hnis_excess.detach().cpu()),
        "safe_loss": float(safe_loss.detach().cpu()),
        "shell_loss": float(shell_loss.detach().cpu()),
        "shell_target": float(shell_targets.mean().detach().cpu()),
        "negative_positive_cosine": float(
            negative_positive_cosines.mean().detach().cpu()
        ),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "route_loss": float(route_loss.detach().cpu()),
        "route_acc": sum(route_correct) / len(route_correct),
        "balance_loss": float(balance_loss.detach().cpu()),
        "balance_excess": float(balance_excess.detach().cpu()),
        "id_unknown_mass_win_rate": float(
            unknown_would_win.float().mean().detach().cpu()
        ),
    }
