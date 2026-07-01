from __future__ import annotations

import re


_PREFIX_RE = re.compile(r"^[a-z]-")


def clean_node_text(name: str, hierarchy=None) -> str:
    raw = name
    if hierarchy is not None:
        raw = hierarchy.node_description.get(name, name)
    if raw == "root":
        return "object"
    raw = _PREFIX_RE.sub("", raw)
    raw = raw.replace("_", " ")
    raw = raw.replace("/", " ")
    raw = " ".join(raw.split())
    return raw


def format_child_prompt(child: str, parent: str, hierarchy, template: str) -> str:
    return template.format(
        child=clean_node_text(child, hierarchy),
        parent=clean_node_text(parent, hierarchy),
        child_name=child,
        parent_name=parent,
    )


def format_unknown_prompts(parent: str, hierarchy, templates: list[str]) -> list[str]:
    return [
        template.format(
            parent=clean_node_text(parent, hierarchy),
            parent_name=parent,
        )
        for template in templates
    ]

