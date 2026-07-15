from __future__ import annotations

import torch


def configured_device(runtime_cfg: dict | None, default: str = "cuda") -> str:
    runtime_cfg = dict(runtime_cfg or {})
    device = str(runtime_cfg.get("device", default))
    gpu_id = runtime_cfg.get("gpu_id")
    if gpu_id is not None and device == "cuda":
        device = f"cuda:{int(gpu_id)}"
    return device


def available_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device
