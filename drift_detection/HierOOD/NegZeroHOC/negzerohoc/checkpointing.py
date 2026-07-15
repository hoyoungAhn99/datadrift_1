from __future__ import annotations

from pathlib import Path

import torch

from .feature_io import ensure_dir
from .prompt_text import TEXT_TEMPLATES_VERSION


def save_idea3_checkpoint(
    path: str | Path,
    *,
    stage: str,
    dataset: str,
    clip_model: str,
    hierarchy: str,
    id_split: str,
    prompt_config: dict,
    positive_state_dict: dict | None = None,
    unknown_state_dict: dict | None = None,
    image_adapter_config: dict | None = None,
    image_adapter_state_dict: dict | None = None,
    positive_checkpoint: str | None = None,
    metrics: dict | None = None,
    args: dict | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "stage": stage,
        "dataset": dataset,
        "clip_model": clip_model,
        "hierarchy": hierarchy,
        "id_split": id_split,
        "prompt_config": prompt_config,
        "positive_state_dict": positive_state_dict,
        "unknown_state_dict": unknown_state_dict,
        "image_adapter_config": image_adapter_config,
        "image_adapter_state_dict": image_adapter_state_dict,
        "positive_checkpoint": positive_checkpoint,
        "text_templates_version": TEXT_TEMPLATES_VERSION,
        "metrics": metrics or {},
        "args": args or {},
    }
    torch.save(payload, path)
    return path


def load_idea3_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
