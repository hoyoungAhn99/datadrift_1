from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config, save_config
from core.feature_io import save_artifact
from core.hierarchy_labels import build_leaf_path_matrix, targets_to_path_labels
from core.metric_losses import build_metric_loss
from core.model_factory import backbone_summary, build_backbone, maybe_wrap_dataparallel, unwrap_model
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import gen_datasets, get_id_classes


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"].lower()
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            nesterov=opt_cfg.get("nesterov", False),
        )
    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")


def build_scheduler(optimizer, config):
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg.get('name')}")


def evaluate_loss(model, loader, loss_fn, leaf_path_matrix, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = model(inputs)
            path_labels = targets_to_path_labels(targets, leaf_path_matrix, device=device)
            loss = loss_fn(features, path_labels)
            total_loss += float(loss.item())
            steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / "resolved_config.yaml")

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    train_ds, val_ds, _ = gen_datasets(dataset_cfg["datadir"], id_classes, hierarchy.ood_train_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=config["dataloader"].get("shuffle_train", True),
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"].get("pin_memory", False),
        drop_last=config["dataloader"].get("drop_last", False),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["dataloader"].get("eval_batch_size", config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"].get("pin_memory", False),
        drop_last=False,
    )

    use_pruned = config.get("hierarchy", {}).get("use_pruned_for_loss_labels", True)
    leaf_path_matrix, path_meta = build_leaf_path_matrix(
        hierarchy,
        train_ds.classes,
        use_pruned=use_pruned,
        hierarchy_path=dataset_cfg["hierarchy"],
    )

    model = build_backbone(config).to(device)
    model = maybe_wrap_dataparallel(model, config)
    loss_cfg = dict(config["loss"])
    loss_name = loss_cfg.pop("name")
    loss_fn = build_metric_loss(loss_name, **loss_cfg)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for inputs, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = model(inputs)
            path_labels = targets_to_path_labels(targets, leaf_path_matrix, device=device)
            loss = loss_fn(features, path_labels)
            optimizer.zero_grad()
            loss.backward()
            if config["training"].get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, leaf_path_matrix, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unwrap_model(model).state_dict(), experiment_dir / "checkpoint_backbone.pt")

    save_artifact(
        {
            "history": history,
            "best_val_loss": best_val_loss,
            "path_metadata": path_meta,
            "backbone": backbone_summary(config),
            "runtime": config.get("runtime", {}),
        },
        experiment_dir / "train_metrics.pt",
    )
    save_artifact(
        {
            "config": config,
            "best_val_loss": best_val_loss,
            "path_metadata": path_meta,
        },
        experiment_dir / "train_config.pt",
    )


if __name__ == "__main__":
    main()
