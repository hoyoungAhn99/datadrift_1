from __future__ import annotations

import torch
from torch import Tensor, nn

from sacil.anchors.affinity import anchor_affinity
from sacil.anchors.hierarchical_anchor_bank import HierarchicalAnchorBank


class AnchorGeometryLoss(nn.Module):
    """Group-normalized anchor-affinity preservation."""

    def __init__(
        self,
        anchor_bank: HierarchicalAnchorBank,
        leaf_weights: Tensor,
        internal_weights: Tensor,
        use_internal_anchors: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if leaf_weights.numel() != anchor_bank.leaf_anchors.shape[0]:
            raise ValueError("leaf weight count mismatch")
        internal_ids, internal_anchors = anchor_bank.internal_without_root()
        if internal_weights.numel() != len(internal_ids):
            raise ValueError("internal weight count mismatch")
        self.register_buffer(
            "leaf_anchors", anchor_bank.leaf_anchors.clone()
        )
        self.register_buffer("leaf_weights", leaf_weights.float().clone())
        self.register_buffer("internal_anchors", internal_anchors.clone())
        self.register_buffer(
            "internal_weights", internal_weights.float().clone()
        )
        self.use_internal_anchors = bool(use_internal_anchors)
        self.epsilon = float(epsilon)

    def _group_loss(
        self,
        current_features: Tensor,
        reference_features: Tensor,
        anchors: Tensor,
        weights: Tensor,
    ) -> Tensor | None:
        if anchors.shape[0] == 0:
            return None
        current = anchor_affinity(current_features, anchors)
        with torch.no_grad():
            reference = anchor_affinity(reference_features, anchors)
        squared_error = (current - reference) ** 2
        denominator = (
            current_features.shape[0] * weights.sum().clamp_min(self.epsilon)
        )
        return (squared_error * weights.unsqueeze(0)).sum() / denominator

    def forward(
        self, current_features: Tensor, reference_features: Tensor
    ) -> Tensor:
        if current_features.shape != reference_features.shape:
            raise ValueError("current and reference feature shapes differ")
        if current_features.shape[0] == 0:
            return current_features.sum() * 0.0
        losses = []
        leaf_loss = self._group_loss(
            current_features,
            reference_features,
            self.leaf_anchors,
            self.leaf_weights,
        )
        if leaf_loss is not None:
            losses.append(leaf_loss)
        if self.use_internal_anchors:
            internal_loss = self._group_loss(
                current_features,
                reference_features,
                self.internal_anchors,
                self.internal_weights,
            )
            if internal_loss is not None:
                losses.append(internal_loss)
        if not losses:
            return current_features.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def per_anchor_drift(
        self, current_features: Tensor, reference_features: Tensor
    ) -> dict[str, Tensor]:
        result = {
            "leaf": (
                anchor_affinity(current_features, self.leaf_anchors)
                - anchor_affinity(reference_features, self.leaf_anchors)
            )
            .pow(2)
            .mean(dim=0)
        }
        if self.internal_anchors.shape[0] > 0:
            result["internal"] = (
                anchor_affinity(current_features, self.internal_anchors)
                - anchor_affinity(reference_features, self.internal_anchors)
            ).pow(2).mean(dim=0)
        else:
            result["internal"] = torch.empty(
                0, device=current_features.device
            )
        return result

