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
