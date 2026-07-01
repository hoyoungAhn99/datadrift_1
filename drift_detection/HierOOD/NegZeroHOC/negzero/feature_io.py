from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_feature_artifact(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_feature_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_feature_artifact(
    artifact: dict[str, Any],
    *,
    expected_model: str | None = None,
    expected_dim: int | None = None,
    require_model_match: bool = True,
) -> None:
    required = ["features", "targets", "node_targets", "class_names", "clip_model"]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"Feature artifact is missing keys: {missing}")

    features = artifact["features"]
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("artifact['features'] must be a rank-2 tensor")
    n = features.shape[0]
    if artifact["targets"].shape[0] != n:
        raise ValueError("targets length does not match features length")
    if artifact["node_targets"].shape[0] != n:
        raise ValueError("node_targets length does not match features length")
    if expected_model and artifact.get("clip_model") != expected_model:
        message = (
            f"Feature CLIP model {artifact.get('clip_model')!r} does not match "
            f"config model {expected_model!r}"
        )
        if require_model_match:
            raise ValueError(message)
        print(f"WARNING: {message}")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension {features.shape[1]} does not match text dimension {expected_dim}"
        )


def feature_paths(config: dict) -> dict[str, Path]:
    feature_cfg = config["features"]
    root = Path(feature_cfg["dir"])
    return {
        "train": root / feature_cfg.get("train", "features_train.pt"),
        "id": root / feature_cfg.get("id_val", "features_val.pt"),
        "ood": root / feature_cfg.get("ood_val", "features_ood.pt"),
        "config": root / feature_cfg.get("config", "feature_config.yaml"),
    }

