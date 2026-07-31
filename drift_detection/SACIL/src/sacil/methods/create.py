from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pointwise_block(
    in_channels: int,
    out_channels: int,
    *,
    activation: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    ]
    if activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ClassAutoencoder(nn.Module):
    """One class-specific 1x1 convolutional autoencoder from CREATE."""

    def __init__(
        self,
        in_features: int,
        hidden_layers: Sequence[int],
        latent_features: int,
    ) -> None:
        super().__init__()
        if in_features <= 0 or latent_features <= 0:
            raise ValueError("CREATE dimensions must be positive")
        encoder: list[nn.Module] = []
        decoder: list[nn.Module] = []
        current = int(in_features)
        for index, hidden in enumerate(hidden_layers):
            if int(hidden) <= 0:
                raise ValueError("CREATE hidden dimensions must be positive")
            encoder.append(
                _pointwise_block(current, int(hidden), activation=True)
            )
            decoder.append(
                _pointwise_block(
                    int(hidden), current, activation=index != 0
                )
            )
            current = int(hidden)
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(reversed(decoder))
        self.to_latent = _pointwise_block(
            current, int(latent_features), activation=True
        )
        self.from_latent = _pointwise_block(
            int(latent_features), current, activation=True
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        encoded = inputs
        for layer in self.encoder:
            encoded = layer(encoded)
        latent = self.to_latent(encoded)
        reconstruction = self.from_latent(latent)
        for layer in self.decoder:
            reconstruction = layer(reconstruction)
        return reconstruction, latent


class ClasswiseAutoencoderClassifier(nn.Module):
    """CREATE's native reconstruction-error classifier."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        hidden_layers: Sequence[int] = (128,),
        latent_features: int = 32,
        reconstruction_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("CREATE requires at least one class")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.latent_features = int(latent_features)
        self.reconstruction_scale = float(reconstruction_scale)
        self.class_autoencoders = nn.ModuleList(
            ClassAutoencoder(
                in_features, hidden_layers, latent_features
            )
            for _ in range(num_classes)
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim == 2:
            features_4d = features[:, :, None, None]
        elif features.ndim == 4:
            features_4d = features
        else:
            raise ValueError("CREATE features must be 2D or 4D")
        errors = []
        latents = []
        reconstructions = []
        for autoencoder in self.class_autoencoders:
            reconstruction, latent = autoencoder(features_4d)
            error = -self.reconstruction_scale * torch.linalg.vector_norm(
                reconstruction - features_4d, ord=1, dim=1, keepdim=True
            )
            errors.append(error.clamp(-500.0, 500.0))
            latents.append(latent.flatten(start_dim=1))
            reconstructions.append(reconstruction.flatten(start_dim=1))
        error_logits = torch.cat(errors, dim=1)
        pooled_errors = F.adaptive_avg_pool2d(
            error_logits, (1, 1)
        ).flatten(start_dim=1)
        probabilities = F.adaptive_avg_pool2d(
            F.softmax(error_logits, dim=1) + 1e-10, (1, 1)
        ).flatten(start_dim=1)
        return {
            "logits": probabilities,
            "error_logits": pooled_errors,
            "latents": torch.stack(latents, dim=1),
            "reconstructions": torch.stack(reconstructions, dim=1),
            "features": features,
        }


def create_classification_loss(
    probabilities: Tensor, targets: Tensor
) -> Tensor:
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("invalid CREATE classification inputs")
    return F.nll_loss(probabilities.clamp_min(1e-10).log(), targets)


def reconstruction_confidence_weights(
    error_logits: Tensor,
    *,
    alpha: float = -10.0,
) -> Tensor:
    if error_logits.ndim != 2 or error_logits.shape[1] < 2:
        raise ValueError("CREATE confidence needs at least two classes")
    sorted_values = error_logits.sort(dim=1, descending=True).values
    score = (sorted_values[:, 1] - sorted_values[:, 0]).abs()
    score = score / (
        (sorted_values[:, 0] - sorted_values[:, -1]).abs() + 1e-8
    )
    return 1.0 + torch.exp(-float(alpha) * score.detach())


def create_contrastive_loss(
    latents: Tensor,
    targets: Tensor,
    *,
    sample_weights: Tensor | None = None,
    temperature: float = 0.1,
    base_temperature: float = 0.07,
) -> Tensor:
    """CREATE's class-wise horizontal supervised contrastive objective."""

    if latents.ndim != 3:
        raise ValueError("CREATE latents must be [batch, classes, features]")
    batch_size, num_classes, _ = latents.shape
    if targets.shape != (batch_size,):
        raise ValueError("invalid CREATE contrastive targets")
    if sample_weights is None:
        sample_weights = latents.new_ones(batch_size)
    if sample_weights.shape != (batch_size,):
        raise ValueError("sample weights must be a batch vector")
    if temperature <= 0 or base_temperature <= 0:
        raise ValueError("contrastive temperatures must be positive")

    identity = torch.eye(
        batch_size, dtype=torch.bool, device=latents.device
    )
    total = latents.new_zeros(())
    for class_id in range(num_classes):
        features = latents[:, class_id]
        logits = (features @ features.T) / float(temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits = logits.clamp_min(-95.0)
        valid = ~identity
        positive = (
            (targets[:, None] == targets[None, :])
            & (targets[:, None] == class_id)
            & valid
        )
        log_denominator = torch.logsumexp(
            logits.masked_fill(~valid, -torch.inf), dim=1
        )
        log_probability = logits - log_denominator[:, None]
        positive_count = positive.sum(dim=1).clamp_min(1)
        mean_positive = (
            log_probability.masked_fill(~positive, 0.0).sum(dim=1)
            / positive_count
        )
        total = total - (
            (temperature / base_temperature)
            * mean_positive
            * sample_weights
        ).mean()
    return total / num_classes

