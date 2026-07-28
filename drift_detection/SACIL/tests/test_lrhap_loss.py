from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank
from sacil.hierarchy import GriffinPeronaGreedy
from sacil.methods import AnchorGeometryLoss, global_preservation_weights


def _loss() -> AnchorGeometryLoss:
    prototypes = F.normalize(torch.randn(4, 8), dim=1)
    affinity = torch.tensor(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ]
    )
    tree = GriffinPeronaGreedy().build((0, 1, 2, 3), affinity)
    bank = HierarchicalAnchorBank.from_tree(
        PrototypeBank((0, 1, 2, 3), prototypes), tree
    )
    weights = global_preservation_weights(bank)
    return AnchorGeometryLoss(
        bank, weights.leaf_weights, weights.internal_weights
    )


def test_geometry_loss_is_zero_for_identical_features() -> None:
    loss = _loss()
    current = torch.randn(5, 8, requires_grad=True)
    reference = current.detach().clone()
    value = loss(current, reference)
    assert torch.allclose(value, torch.tensor(0.0), atol=1e-8)
    value.backward()
    assert current.grad is not None


def test_geometry_loss_increases_after_feature_perturbation() -> None:
    loss = _loss()
    reference = torch.randn(5, 8)
    identical = loss(reference.clone(), reference)
    perturbed = loss(reference + torch.randn_like(reference), reference)
    assert perturbed > identical


def test_empty_old_batch_is_safe() -> None:
    loss = _loss()
    empty = torch.empty(0, 8, requires_grad=True)
    value = loss(empty, empty.detach())
    assert value.item() == 0.0

