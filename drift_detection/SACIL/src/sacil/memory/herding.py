from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def herding_select(
    features: Tensor,
    dataset_indices: Sequence[int],
    exemplars_per_class: int,
) -> list[int]:
    """Select exemplars whose running mean approximates the class mean."""
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if features.shape[0] != len(dataset_indices):
        raise ValueError("feature and index counts do not match")
    count = min(int(exemplars_per_class), features.shape[0])
    if count <= 0:
        return []

    normalized = F.normalize(features.detach().float(), dim=1)
    class_mean = F.normalize(normalized.mean(dim=0, keepdim=True), dim=1)[0]
    selected_mask = torch.zeros(
        normalized.shape[0], dtype=torch.bool, device=normalized.device
    )
    running_sum = torch.zeros_like(class_mean)
    selected_positions: list[int] = []

    for selected_count in range(count):
        candidate_means = (running_sum.unsqueeze(0) + normalized) / (
            selected_count + 1
        )
        distances = torch.sum(
            (candidate_means - class_mean.unsqueeze(0)) ** 2, dim=1
        )
        distances[selected_mask] = torch.inf
        position = int(torch.argmin(distances).item())
        selected_positions.append(position)
        selected_mask[position] = True
        running_sum = running_sum + normalized[position]

    return [int(dataset_indices[position]) for position in selected_positions]

