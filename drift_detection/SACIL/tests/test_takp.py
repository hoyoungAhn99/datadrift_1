from __future__ import annotations

import torch

from sacil.engine.trainer import inverse_class_sample_weights
from sacil.methods import takp_mixed_classification_loss
from sacil.models import TaKPIncrementalNet


def test_takp_public_bbn_mixing_and_forward_shapes() -> None:
    model = TaKPIncrementalNet(num_classes=10, mix_scale=2.0).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model.forward_detailed(images, alpha=0.5)
        logits, features = model(images, return_features=True)

    assert output.conventional_features.shape == (2, 512)
    assert output.rebalancing_features.shape == (2, 512)
    assert features.shape == (2, 1024)
    assert logits.shape == (2, 10)
    assert torch.allclose(
        features,
        torch.cat(
            (
                output.conventional_features,
                output.rebalancing_features,
            ),
            dim=1,
        ),
    )
    assert torch.allclose(logits, output.logits)


def test_takp_classifier_expansion_preserves_old_rows() -> None:
    model = TaKPIncrementalNet(num_classes=10)
    old_weight = model.classifier.weight.detach().clone()
    old_bias = model.classifier.bias.detach().clone()

    model.expand_classes(20)

    assert model.num_classes == 20
    assert torch.equal(model.classifier.weight[:10], old_weight)
    assert torch.equal(model.classifier.bias[:10], old_bias)


def test_inverse_sampler_matches_paper_class_probabilities() -> None:
    targets = [0, 0, 1, 1, 1, 1]
    weights = inverse_class_sample_weights(targets)
    class_zero_mass = weights[:2].sum()
    class_one_mass = weights[2:].sum()

    assert torch.allclose(
        class_zero_mass / class_one_mass,
        torch.tensor(2.0, dtype=torch.double),
    )


def test_takp_mixed_loss_updates_both_label_terms() -> None:
    logits = torch.randn(4, 6, requires_grad=True)
    conventional_targets = torch.tensor([0, 1, 2, 3])
    rebalancing_targets = torch.tensor([3, 2, 1, 0])

    loss = takp_mixed_classification_loss(
        logits,
        conventional_targets,
        rebalancing_targets,
        alpha=0.75,
    )
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
