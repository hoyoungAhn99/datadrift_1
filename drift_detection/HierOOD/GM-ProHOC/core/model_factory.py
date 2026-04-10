from __future__ import annotations

from typing import Any

from feature_extractor import CLIPImageEncoder, ResNet50


def build_backbone(config: dict[str, Any]):
    backbone_cfg = config["backbone"]
    model_type = backbone_cfg["type"].lower()
    feat_dim = backbone_cfg["feat_dim"]
    normalize = backbone_cfg.get("normalize", False)
    freeze_backbone = backbone_cfg.get("freeze_backbone", False)

    if model_type == "resnet50":
        return ResNet50(
            feat_dim=feat_dim,
            weights=backbone_cfg.get("weights"),
            freeze_backbone=freeze_backbone,
            normalize=normalize,
        )
    if model_type == "clip":
        return CLIPImageEncoder(
            feat_dim=feat_dim,
            model_name=backbone_cfg.get("model_name", "openai/clip-vit-base-patch32"),
            freeze_backbone=freeze_backbone,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported backbone type: {model_type}")


def backbone_summary(config: dict[str, Any]) -> dict[str, Any]:
    backbone_cfg = dict(config["backbone"])
    return {
        "type": backbone_cfg.get("type"),
        "feat_dim": backbone_cfg.get("feat_dim"),
        "normalize": backbone_cfg.get("normalize"),
        "freeze_backbone": backbone_cfg.get("freeze_backbone"),
        "weights": backbone_cfg.get("weights"),
        "model_name": backbone_cfg.get("model_name"),
    }
