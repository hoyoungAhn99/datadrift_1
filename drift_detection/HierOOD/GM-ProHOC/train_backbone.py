from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config, save_config
from core.density import fit_diagonal_gaussians, score_nodes
from core.feature_io import save_artifact
from core.hierarchy_labels import build_leaf_path_matrix, targets_to_path_labels
from core.metric_losses import build_metric_loss
from core.model_factory import backbone_summary, build_backbone, maybe_wrap_dataparallel, unwrap_model
from core.hierarchy_inference import build_depth_maps
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


def evaluate_loss(model, loader, loss_fn, leaf_path_matrix, device, desc="Val loss"):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = model(inputs)
            path_labels = targets_to_path_labels(targets, leaf_path_matrix, device=device)
            loss = loss_fn(features, path_labels)
            total_loss += float(loss.item())
            steps += 1
    return total_loss / max(steps, 1)


def collect_features_and_targets(model, loader, device, desc):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            feats = model(inputs)
            features.append(feats.detach().cpu())
            targets.append(batch_targets.long().cpu())
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def compute_depthwise_accuracy(
    model,
    train_loader,
    val_loader,
    hierarchy,
    train_classes,
    leaf_path_matrix,
    density_eps,
    score_type,
    temperature,
    device,
):
    train_features, train_targets = collect_features_and_targets(
        model, train_loader, device, desc="Depth eval train features"
    )
    val_features, val_targets = collect_features_and_targets(
        model, val_loader, device, desc="Depth eval val features"
    )

    density = fit_diagonal_gaussians(
        train_features.float(),
        train_targets.long(),
        hierarchy,
        train_classes,
        eps=density_eps,
    )
    node_scores = score_nodes(
        val_features.float(),
        density["means"].float(),
        density["variances"].float(),
        score_type=score_type,
        temperature=temperature,
    )
    nodes_by_depth = build_depth_maps(hierarchy)
    val_class_names = [train_classes[int(idx)] for idx in val_targets.long().tolist()]

    depth_metrics = {}
    top1_values = []
    top5_values = []

    for depth in sorted(nodes_by_depth.keys()):
        if depth == 0:
            continue
        depth_indices = nodes_by_depth[depth]
        logits = node_scores[:, depth_indices]
        depth_index_map = {node_idx: pos for pos, node_idx in enumerate(depth_indices)}
        valid_rows = []
        true_local_list = []

        for row_idx, class_name in enumerate(val_class_names):
            ancestors = hierarchy.node_ancestors[class_name].copy()
            actual_depth = len(ancestors)
            if actual_depth < depth:
                continue

            if depth == actual_depth:
                node_idx = hierarchy.id_node_list.index(class_name)
            else:
                node_idx = ancestors[depth]

            if node_idx not in depth_index_map:
                continue

            valid_rows.append(row_idx)
            true_local_list.append(depth_index_map[node_idx])

        if not valid_rows:
            continue

        valid_logits = logits[valid_rows]
        true_local = torch.tensor(true_local_list, dtype=torch.long)
        top1 = valid_logits.argmax(dim=1)
        top1_acc = (top1.cpu() == true_local).float().mean().item()
        k = min(5, valid_logits.shape[1])
        topk = torch.topk(valid_logits, k=k, dim=1).indices.cpu()
        top5_acc = (topk == true_local.unsqueeze(1)).any(dim=1).float().mean().item()

        depth_metrics[depth] = {
            "acc_top1": top1_acc,
            "acc_top5": top5_acc,
            "num_classes": len(depth_indices),
            "num_samples": len(valid_rows),
        }
        top1_values.append(top1_acc)
        top5_values.append(top5_acc)

    summary = {
        "depth_metrics": depth_metrics,
        "acc_top1_mean": sum(top1_values) / max(len(top1_values), 1),
        "acc_top5_mean": sum(top5_values) / max(len(top5_values), 1),
    }
    return summary


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
    tensorboard_dir = experiment_dir / "tensorboard"
    use_tensorboard = config.get("logging", {}).get("use_tensorboard", False)
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if use_tensorboard else None

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
    eval_train_loader = DataLoader(
        train_ds,
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
    validation_cfg = config.get("validation", {})
    enable_depth_eval = validation_cfg.get("enable_depth_eval", False)
    depth_eval_every = max(int(validation_cfg.get("depth_eval_every", 1)), 1)

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
        val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
            leaf_path_matrix,
            device,
            desc=f"Val loss {epoch+1}/{epochs}",
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        depth_eval_summary = None
        if enable_depth_eval and ((epoch + 1) % depth_eval_every == 0):
            depth_eval_summary = compute_depthwise_accuracy(
                model,
                eval_train_loader,
                val_loader,
                hierarchy,
                train_ds.classes,
                leaf_path_matrix,
                config["density"]["eps"],
                config["inference"].get("score_type", "gaussian_loglik"),
                config["inference"].get("temperature", 1.0),
                device,
            )
            history.setdefault("depth_eval", []).append({"epoch": epoch, **depth_eval_summary})

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.6g}"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("test/loss", val_loss, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)
            if depth_eval_summary is not None:
                writer.add_scalar("test/acc_top1_mean", depth_eval_summary["acc_top1_mean"], epoch)
                writer.add_scalar("test/acc_top5_mean", depth_eval_summary["acc_top5_mean"], epoch)
                for depth, metrics in depth_eval_summary["depth_metrics"].items():
                    writer.add_scalar(f"test/acc_top1_depth_{depth}", metrics["acc_top1"], epoch)
                    writer.add_scalar(f"test/acc_top5_depth_{depth}", metrics["acc_top5"], epoch)
            writer.flush()

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
            "tensorboard_dir": str(tensorboard_dir) if use_tensorboard else None,
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
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {tensorboard_dir}")


if __name__ == "__main__":
    main()
