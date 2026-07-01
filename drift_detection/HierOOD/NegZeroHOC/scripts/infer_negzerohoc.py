from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ProHOC"))

from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.config import namespace_from_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, load_feature_file
from negzerohoc.inference import predict_features
from negzerohoc.semantic_index import build_semantic_index


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
            "features_dir": None,
            "clip_model": "openai/clip-vit-base-patch32",
            "mode": None,
            "outdir": "outputs",
            "device": "cuda",
            "tau": 1.0,
            "local_files_only": True,
            "save_trace": False,
        },
        required=("features_dir", "mode"),
        choices={"mode": {"child_only", "manual_unknown"}},
    )


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    dists_mats = make_distance_mats(hierarchy)

    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    semantic_index = build_semantic_index(args.dataset, hierarchy, backend, args.mode)

    features_dir = Path(args.features_dir)
    val_payload = load_feature_file(features_dir / "val-features.pt")
    ood_payload = load_feature_file(features_dir / "ood-features.pt")

    results = {
        "args": vars(args),
        "mode": args.mode,
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "semantic_prompts": {
            parent: local.prompts for parent, local in semantic_index.items()
        },
    }

    for split_name, payload in [("val", val_payload), ("ood", ood_payload)]:
        pred_out = predict_features(
            payload["features"].to(device),
            hierarchy,
            semantic_index,
            mode=args.mode,
            tau=args.tau,
            return_trace=args.save_trace,
        )
        preds = pred_out["preds"].cpu()
        node_labels, metrics = evaluate_split(hierarchy, payload, preds, dists_mats=dists_mats)
        results[split_name] = {
            "preds": preds,
            "targets": node_labels.cpu(),
            "metrics": metrics,
            "diagnostics": pred_out["diagnostics"],
        }
        if args.save_trace:
            results[split_name]["trace"] = pred_out["traces"]

    results["mixed"] = mixed_summary(results["val"]["metrics"], results["ood"]["metrics"])

    model_key = f"clip_{safe_model_name(args.clip_model)}"
    result_dir = ensure_dir(Path(args.outdir) / "results")
    save_path = result_dir / f"negzerohoc-{args.dataset}-{model_key}-{args.mode}.result"
    torch.save(results, save_path)

    print(f"saved: {save_path}")
    print(f"ID BAcc: {results['val']['metrics']['balanced_acc']:.6f}")
    print(f"OOD BAcc: {results['ood']['metrics']['balanced_acc']:.6f}")
    print(f"Mixed BAcc: {results['mixed']['mixed_balanced_acc']:.6f}")
    print(f"ID Balanced H-Dist: {float(results['val']['metrics']['balanced_hdist']):.6f}")
    print(f"OOD Balanced H-Dist: {float(results['ood']['metrics']['balanced_hdist']):.6f}")
    print(f"Mixed Balanced H-Dist: {results['mixed']['mixed_balanced_hdist']:.6f}")


if __name__ == "__main__":
    main()
