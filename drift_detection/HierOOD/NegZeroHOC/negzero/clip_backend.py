from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CLIPBackendConfig:
    backend: str
    model_name: str
    device: str = "cuda"


class CLIPBackend:
    def __init__(self, config: CLIPBackendConfig):
        backend = config.backend.lower()
        if backend != "transformers":
            raise ValueError(f"Unsupported CLIP backend: {config.backend}")
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("transformers is required for CLIP backend 'transformers'.") from exc

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA requested but unavailable; using CPU.")
        else:
            device = config.device

        self.config = CLIPBackendConfig(
            backend=backend,
            model_name=config.model_name,
            device=device,
        )
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(config.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model.eval()

    @classmethod
    def from_config(cls, model_cfg: dict):
        return cls(
            CLIPBackendConfig(
                backend=model_cfg.get("backend", "transformers"),
                model_name=model_cfg["name"],
                device=model_cfg.get("device", "cuda"),
            )
        )

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        features = self.model.get_image_features(pixel_values=pixel_values)
        if normalize:
            features = F.normalize(features.float(), dim=-1)
        return features

    @torch.no_grad()
    def encode_text(self, prompts: list[str], normalize: bool = True, batch_size: int = 256) -> torch.Tensor:
        chunks = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(**inputs)
            if normalize:
                features = F.normalize(features.float(), dim=-1)
            chunks.append(features.detach())
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=self.device)

