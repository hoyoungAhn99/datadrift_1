from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.config import load_config, save_config
from negzero.clip_backend import CLIPBackend
from negzero.data import build_data_bundle
from negzero.feature_io import feature_paths, save_feature_artifact


def collect_features(backend: CLIPBackend, loader, desc: str) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    for inputs, batch_targets in tqdm(loader, desc=desc):
        batch_features = backend.encode_images(inputs)
        features.append(batch_features.detach().cpu())
        targets.append(batch_targets.long().cpu())
    return F.normalize(torch.cat(features, dim=0), dim=-1), torch.cat(targets, dim=0)


def split_payload(
    split: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    dataset,
    hierarchy,
    config: dict,
) -> dict:
    return {
        "split": split,
        "features": features,
        "targets": targets,
        "node_targets": hierarchy.gen_ds2node_map(dataset.classes)[targets],
        "class_names": list(dataset.classes),
        "samples": [path for path, _ in dataset.samples],
        "clip_model": config["model"]["name"],
        "clip_backend": config["model"].get("backend", "transformers"),
        "feature_norm": "l2",
        "preprocessing": {
            "transform_preset": config["dataset"].get("transform_preset", "clip"),
            "resize": config["dataset"].get("resize"),
            "cropsize": config["dataset"].get("cropsize"),
            "mean": config["dataset"].get("mean"),
            "std": config["dataset"].get("std"),
        },
        "dataset": {
            "name": config["dataset"]["name"],
            "datadir": config["dataset"]["datadir"],
            "id_split": config["dataset"]["id_split"],
        },
        "hierarchy": {
            "path": config["dataset"]["hierarchy"],
            "node_names": hierarchy.id_node_list,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = feature_paths(config)
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    save_config(config, paths["config"])

    data = build_data_bundle(config)
    backend = CLIPBackend.from_config(config["model"])

    split_specs = [
        ("train", data.train_loader, data.train_dataset, paths["train"]),
        ("val", data.val_loader, data.val_dataset, paths["id"]),
        ("ood", data.ood_loader, data.ood_dataset, paths["ood"]),
    ]
    for split, loader, dataset, output_path in split_specs:
        features, targets = collect_features(backend, loader, desc=f"Extract {split}")
        save_feature_artifact(
            split_payload(split, features, targets, dataset, data.hierarchy, config),
            output_path,
        )
        print(f"Saved {split} features: {Path(output_path)}")


if __name__ == "__main__":
    main()

