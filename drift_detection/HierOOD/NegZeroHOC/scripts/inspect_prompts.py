from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config import namespace_from_config
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.prompts import (
    build_positive_prompts,
    build_unknown_prompts,
    canonicalize_node_name,
    infer_node_role,
    node_path_names,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    config_arg = parser.parse_args()
    return namespace_from_config(
        config_arg.config,
        defaults={
            "dataset": "fgvc-aircraft",
            "hierarchy": "hierarchies/fgvc-aircraft.json",
            "id_split": "data/fgvc-aircraft-id-labels.csv",
            "nodes": ["m-Boeing", "f-Boeing_737", "v-737-300"],
            "parents": ["root", "m-Boeing", "f-Boeing_737"],
        },
    )


def main():
    args = parse_args()
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)

    print("== Positive prompts ==")
    for node in args.nodes:
        if node not in hierarchy.id_node_list:
            print(f"[skip] {node}: not in retained ID hierarchy")
            continue
        path = node_path_names(hierarchy, node, include_self=True)
        role = infer_node_role(args.dataset, node)
        canonical = canonicalize_node_name(args.dataset, node, path, role)
        parent = None
        if node in hierarchy.child2parent:
            parent = hierarchy.child2parent[node]
        print(f"\nnode: {node}")
        print(f"role: {role}")
        print(f"path: {' > '.join(path)}")
        print(f"canonical: {canonical}")
        for prompt in build_positive_prompts(args.dataset, node, parent, path, role):
            print(f"- {prompt}")

    print("\n== Unknown prompts ==")
    for parent in args.parents:
        if parent not in hierarchy.id_node_list:
            print(f"[skip] {parent}: not in retained ID hierarchy")
            continue
        if parent not in hierarchy.parent2children:
            print(f"[skip] {parent}: not an internal parent")
            continue
        path = node_path_names(hierarchy, parent, include_self=True)
        canonical = canonicalize_node_name(args.dataset, parent, path, infer_node_role(args.dataset, parent))
        print(f"\nparent: {parent}")
        print(f"path: {' > '.join(path)}")
        print(f"canonical: {canonical}")
        print(f"children: {', '.join(hierarchy.parent2children[parent])}")
        for prompt in build_unknown_prompts(args.dataset, parent, path):
            print(f"- {prompt}")


if __name__ == "__main__":
    main()
