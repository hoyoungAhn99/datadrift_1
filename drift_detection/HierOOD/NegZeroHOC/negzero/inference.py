from __future__ import annotations

from typing import Any

import torch


def predict_one(
    feature: torch.Tensor,
    hierarchy,
    text_cache: dict,
    temperature: float = 1.0,
    return_trace: bool = False,
) -> tuple[int, list[dict[str, Any]] | None]:
    node = "root"
    trace = []
    while node in hierarchy.parent2children:
        local = text_cache[node]
        text_embeddings = local.text_embeddings.to(feature.device)
        logits = float(temperature) * torch.mv(text_embeddings, feature)
        choice = int(logits.argmax().item())
        is_unknown = local.unknown_position >= 0 and choice == local.unknown_position
        if return_trace:
            trace.append(
                {
                    "parent": node,
                    "choice": choice,
                    "is_unknown": is_unknown,
                    "logits": logits.detach().cpu(),
                }
            )
        if is_unknown:
            return hierarchy.id_node_list.index(node), trace if return_trace else None
        node = local.child_names[choice]
    return hierarchy.id_node_list.index(node), trace if return_trace else None


def predict_features(
    features: torch.Tensor,
    hierarchy,
    text_cache: dict,
    temperature: float = 1.0,
    return_trace: bool = False,
) -> tuple[torch.Tensor, list[list[dict[str, Any]]] | None]:
    preds = []
    traces = [] if return_trace else None
    for feature in features:
        pred, trace = predict_one(
            feature,
            hierarchy,
            text_cache,
            temperature=temperature,
            return_trace=return_trace,
        )
        preds.append(pred)
        if return_trace:
            traces.append(trace or [])
    return torch.tensor(preds, dtype=torch.long), traces

