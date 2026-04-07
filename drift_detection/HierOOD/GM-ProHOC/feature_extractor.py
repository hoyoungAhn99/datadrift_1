from typing import Optional, Union

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPVisionModelWithProjection


class ResNet50(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        weights: Optional[Union[str, object]] = None,
        freeze_backbone: bool = False,
        normalize: bool = False,
    ):
        super().__init__()

        if isinstance(weights, str):
            weights = models.ResNet50_Weights[weights]

        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.proj = nn.Identity() if feat_dim == in_features else nn.Linear(in_features, feat_dim)
        self.normalize = normalize

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.proj(features)
        if self.normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features


class CLIPImageEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_backbone: bool = False,
        normalize: bool = False,
    ):
        super().__init__()

        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        in_features = self.backbone.visual_projection.out_features

        self.proj = nn.Identity() if feat_dim == in_features else nn.Linear(in_features, feat_dim)
        self.normalize = normalize

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=x).image_embeds
        features = self.proj(features)
        if self.normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features


class FeatExtractor(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self(x)
        # Loss implementation would go here depending on the training objective
        return features

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
