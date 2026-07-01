from __future__ import annotations

from collections import Counter
from typing import Any

import torch


def node_depth(hierarchy, node_idx: int) -> int:
    node_name = hierarchy.id_node_list[int(node_idx)]
    return len(hierarchy.node_ancestors[node_name])


def summarize_predictions(preds: torch.Tensor, hierarchy, traces: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    n = int(preds.numel())
    stop_depths = [node_depth(hierarchy, int(pred)) for pred in preds.tolist()]
    internal_stop_count = sum(
        1 for pred in preds.tolist() if hierarchy.id_node_list[int(pred)] in hierarchy.parent2children
    )
    summary = {
        "num_samples": n,
        "unknown_selection_rate": internal_stop_count / max(n, 1),
        "stop_depth_histogram": dict(sorted(Counter(stop_depths).items())),
        "mean_stop_depth": sum(stop_depths) / max(n, 1),
    }
    if traces is not None:
        path_lengths = [len(trace) for trace in traces]
        summary["mean_trace_length"] = sum(path_lengths) / max(len(path_lengths), 1)
    return summary

