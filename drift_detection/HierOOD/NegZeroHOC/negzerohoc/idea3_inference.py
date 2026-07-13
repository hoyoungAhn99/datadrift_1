from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .semantic_index import LocalSemanticCandidates


@dataclass
class Idea3SemanticIndex:
    locals: dict[str, LocalSemanticCandidates]


@torch.no_grad()
def build_idea3_semantic_index(
    hierarchy,
    positive_learner,
    unknown_learner=None,
    mode: str = "positive_child_only",
    allow_root_unknown: bool = False,
) -> dict[str, LocalSemanticCandidates]:
    if mode not in {"positive_child_only", "positive_pathscore_diagnostic", "parent_unknown"}:
        raise ValueError(f"Unsupported Idea 3 inference mode: {mode}")

    index = {}
    was_training = positive_learner.training
    positive_learner.eval()
    if unknown_learner is not None:
        unknown_was_training = unknown_learner.training
        unknown_learner.eval()
    else:
        unknown_was_training = False

    for parent, children in hierarchy.parent2children.items():
        child_features = positive_learner.encode_children(parent, list(children)).detach().cpu()
        unknown_feature = None
        candidate_names = list(children)
        prompts = {
            child: [positive_learner.edge_text(parent, child)]
            for child in children
        }

        if mode == "parent_unknown" and unknown_learner is not None and (parent != "root" or allow_root_unknown):
            unknown_feature = unknown_learner.encode_unknown(parent).detach().cpu()
            unknown_name = f"__unknown__:{parent}"
            candidate_names.append(unknown_name)
            prompts[unknown_name] = [unknown_learner.unknown_text(parent)]

        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(children),
            child_features=child_features,
            unknown_feature=unknown_feature,
            candidate_names=candidate_names,
            prompts=prompts,
        )

    if was_training:
        positive_learner.train()
    if unknown_learner is not None and unknown_was_training:
        unknown_learner.train()
    return index


def predict_one_idea3(
    image_feature: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    return_trace: bool = False,
):
    node = "root"
    trace = []

    while node in hierarchy.parent2children:
        local = semantic_index[node]
        feats = local.child_features.to(image_feature.device)
        names = list(local.children)

        if mode == "parent_unknown" and local.unknown_feature is not None:
            feats = torch.cat([feats, local.unknown_feature.to(image_feature.device).unsqueeze(0)], dim=0)
            names.append(f"__unknown__:{node}")

        logits = float(tau) * torch.mv(feats, image_feature)
        winner_idx = int(torch.argmax(logits).item())
        winner = names[winner_idx]

        if return_trace:
            trace.append({
                "parent": node,
                "candidates": names,
                "winner": winner,
                "logits": logits.detach().cpu().tolist(),
            })

        if winner.startswith("__unknown__:"):
            return node, trace
        node = winner

    return node, trace


def predict_features_idea3(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    return_trace: bool = False,
) -> dict:
    features = F.normalize(features.float(), dim=-1)
    preds = []
    traces = [] if return_trace else None
    stop_nodes = []

    for feat in features:
        node, trace = predict_one_idea3(
            feat,
            hierarchy,
            semantic_index,
            mode=mode,
            tau=tau,
            return_trace=return_trace,
        )
        preds.append(hierarchy.id_node_list.index(node))
        stop_nodes.append(node)
        if return_trace:
            traces.append(trace)

    return {
        "preds": torch.tensor(preds, dtype=torch.long),
        "traces": traces,
        "diagnostics": summarize_stop_nodes(hierarchy, stop_nodes),
    }


def summarize_stop_nodes(hierarchy, stop_nodes: list[str]) -> dict:
    depth_counts = Counter(len(hierarchy.node_ancestors.get(node, [])) for node in stop_nodes)
    node_counts = Counter(stop_nodes)
    return {
        "stop_depth_counts": dict(sorted(depth_counts.items())),
        "stop_node_counts": dict(node_counts.most_common()),
    }
