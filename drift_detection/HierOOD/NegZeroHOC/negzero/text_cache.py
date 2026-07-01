from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from negzero.prompts import format_child_prompt, format_unknown_prompts


@dataclass
class LocalTextCandidates:
    parent_name: str
    child_names: list[str]
    child_indices: list[int]
    text_embeddings: torch.Tensor
    unknown_position: int
    prompts: list[str]


def _mean_normalized(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def build_text_cache(hierarchy, backend, prompt_cfg: dict, include_unknown: bool = True):
    child_template = prompt_cfg["child_template"]
    unknown_templates = prompt_cfg.get("unknown_templates", [])
    if include_unknown and not unknown_templates:
        raise ValueError("prompts.unknown_templates must be non-empty when include_unknown is true")

    cache = {}
    for parent, children in hierarchy.parent2children.items():
        if not children:
            continue
        child_prompts = [
            format_child_prompt(child, parent, hierarchy, child_template)
            for child in children
        ]
        child_embeddings = backend.encode_text(child_prompts)
        prompts = list(child_prompts)
        embeddings = [child_embeddings]
        unknown_position = -1

        if include_unknown:
            unknown_prompts = format_unknown_prompts(parent, hierarchy, unknown_templates)
            unknown_embeddings = backend.encode_text(unknown_prompts)
            unknown_embedding = _mean_normalized(unknown_embeddings).unsqueeze(0)
            unknown_position = len(children)
            prompts.extend(unknown_prompts)
            embeddings.append(unknown_embedding)

        text_embeddings = torch.cat(embeddings, dim=0)
        cache[parent] = LocalTextCandidates(
            parent_name=parent,
            child_names=list(children),
            child_indices=[hierarchy.id_node_list.index(child) for child in children],
            text_embeddings=text_embeddings,
            unknown_position=unknown_position,
            prompts=prompts,
        )
    return cache

