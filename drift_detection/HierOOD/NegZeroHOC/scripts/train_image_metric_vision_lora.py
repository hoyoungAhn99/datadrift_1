from __future__ import annotations

import argparse
from argparse import Namespace
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import save_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import save_json
from negzerohoc.image_metric import (
    PKBatchSampler,
    batch_hard_hierarchical_triplet_loss,
    class_tree_distance_matrix,
    cosine_proxy_loss,
    supervised_contrastive_loss,
)
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.runtime import available_device, configured_device
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_enabled,
    set_vision_lora_train_mode,
    vision_lora_parameters,
    vision_lora_state_dict,
)
from scripts.train_idea3_joint_vision_lora import (
    autocast_context,
    build_datasets,
    load_clip_and_tokenizer,
    make_grad_scaler,
    make_loader,
)


CHECKPOINT_STAGE = "image_metric_vision_lora"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    train_cfg = cfg.get("image_metric_training", {})
    loss_cfg = train_cfg.get("loss", {})
    validation_cfg = train_cfg.get("validation", {})
    experiment_name = str(experiment_cfg.get("name", "image-metric-vision-lora"))
    output_root = Path(experiment_cfg.get("output_root", "outputs"))

    def artifact(configured, kind: str, filename: str) -> str:
        return str(resolve_experiment_artifact(
            configured,
            output_root=output_root,
            experiment_name=experiment_name,
            kind=kind,
            default_filename=filename,
        ))

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=str(dataset_cfg.get("datadir", "")),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch16"),
        tokenizer_model=clip_cfg.get("tokenizer_model", clip_cfg.get("model", "openai/clip-vit-base-patch16")),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        augmentation=cfg.get("augmentation", {}),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        vision_lora=cfg.get("vision_lora", {}),
        epochs=max(1, int(train_cfg.get("epochs", 30))),
        classes_per_batch=max(2, int(train_cfg.get("classes_per_batch", 4))),
        examples_per_class=max(2, int(train_cfg.get("examples_per_class", 4))),
        eval_batch_size=max(1, int(validation_cfg.get("batch_size", 64))),
        lora_lr=float(train_cfg.get("lora_lr", 1e-4)),
        proxy_lr=float(train_cfg.get("proxy_lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        supcon_temperature=float(loss_cfg.get("supcon_temperature", 0.1)),
        proxy_temperature=float(loss_cfg.get("proxy_temperature", 0.07)),
        proxy_margin=float(loss_cfg.get("proxy_margin", 0.05)),
        triplet_base_margin=float(loss_cfg.get("triplet_base_margin", 0.1)),
        triplet_hierarchy_margin=float(loss_cfg.get("triplet_hierarchy_margin", 0.1)),
        lambda_supcon=float(loss_cfg.get("lambda_supcon", 1.0)),
        lambda_triplet=float(loss_cfg.get("lambda_triplet", 0.5)),
        lambda_proxy=float(loss_cfg.get("lambda_proxy", 1.0)),
        lambda_retention=float(loss_cfg.get("lambda_retention", 0.5)),
        validation_every_n_epochs=max(1, int(validation_cfg.get("every_n_epochs", 1))),
        checkpoint=artifact(train_cfg.get("checkpoint"), "checkpoints", f"{experiment_name}.pt"),
        last_checkpoint=artifact(validation_cfg.get("last_checkpoint"), "checkpoints", f"{experiment_name}-last.pt"),
        diagnostics_path=artifact(train_cfg.get("diagnostics_path"), "diagnostics", f"{experiment_name}-diagnostics.json"),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def balanced_accuracy(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    values = []
    for target in sorted(set(targets.detach().cpu().tolist())):
        mask = targets == int(target)
        values.append(float((predictions[mask] == targets[mask]).float().mean()))
    return sum(values) / max(1, len(values))


@torch.no_grad()
def encode_base_features(args, clip_model, images: torch.Tensor, device: str) -> torch.Tensor:
    set_vision_lora_enabled(clip_model, False)
    try:
        with autocast_context(args, device):
            return clip_model.get_image_features(pixel_values=images).float()
    finally:
        set_vision_lora_enabled(clip_model, True)


@torch.no_grad()
def initialize_proxies(args, clip_model, loader, num_classes: int, feature_dim: int, device: str) -> torch.Tensor:
    sums = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    iterator = tqdm(loader, desc="initialize image proxies", leave=False) if tqdm else loader
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.long().to(device)
        features = encode_base_features(args, clip_model, images, device)
        sums.index_add_(0, targets, features)
        counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float32))
    if bool((counts == 0).any()):
        raise RuntimeError("Proxy initialization found an empty class")
    return torch.nn.functional.normalize(sums / counts.unsqueeze(1), dim=-1)


@torch.no_grad()
def evaluate_proxy(args, clip_model, proxies: torch.Tensor, loader, device: str) -> dict:
    clip_model.eval()
    set_vision_lora_enabled(clip_model, True)
    set_vision_lora_train_mode(clip_model, False)
    target_chunks = []
    prediction_chunks = []
    cosine_chunks = []
    normalized_proxies = torch.nn.functional.normalize(proxies.float(), dim=-1)
    iterator = tqdm(loader, desc="metric ID validation", leave=False) if tqdm else loader
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            features = clip_model.get_image_features(pixel_values=images)
        logits = torch.nn.functional.normalize(features.float(), dim=-1) @ normalized_proxies.t()
        predictions = logits.argmax(dim=1).cpu()
        target_chunks.append(targets.long().cpu())
        prediction_chunks.append(predictions)
        cosine_chunks.append(logits.max(dim=1).values.cpu())
    targets = torch.cat(target_chunks)
    predictions = torch.cat(prediction_chunks)
    max_cosines = torch.cat(cosine_chunks)
    return {
        "balanced_acc": balanced_accuracy(targets, predictions),
        "accuracy": float((targets == predictions).float().mean()),
        "mean_max_proxy_cosine": float(max_cosines.mean()),
    }


def save_checkpoint(args, path: str, lora_cfg, clip_model, proxies, classes, metrics):
    checkpoint_path = save_idea3_checkpoint(
        path,
        stage=CHECKPOINT_STAGE,
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config={},
        vision_lora_config=lora_cfg.to_dict(),
        vision_lora_state_dict=vision_lora_state_dict(clip_model),
        metrics=metrics,
        args=vars(args),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["metric_proxies"] = proxies.detach().cpu().clone()
    checkpoint["metric_proxy_classes"] = list(classes)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main():
    args = parse_args()
    if not args.datadir:
        raise ValueError("Missing dataset.datadir")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    train_dataset, val_dataset, _ = build_datasets(args, hierarchy)
    batch_sampler = PKBatchSampler(
        train_dataset.targets,
        classes_per_batch=args.classes_per_batch,
        examples_per_class=args.examples_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    proxy_init_loader = make_loader(
        train_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    clip_model, _ = load_clip_and_tokenizer(args, device)
    if args.gradient_checkpointing and hasattr(clip_model, "gradient_checkpointing_enable"):
        clip_model.gradient_checkpointing_enable()
    lora_cfg = VisionLoRAConfig.from_dict(args.vision_lora)
    replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    lora_params = vision_lora_parameters(clip_model)
    feature_dim = int(clip_model.config.projection_dim)
    initial_proxies = initialize_proxies(
        args, clip_model, proxy_init_loader, len(train_dataset.classes), feature_dim, device
    )
    proxies = nn.Parameter(initial_proxies)

    class_node_indices = hierarchy.gen_ds2node_map(train_dataset.classes)
    class_nodes = [hierarchy.id_node_list[int(index)] for index in class_node_indices.tolist()]
    class_distances = class_tree_distance_matrix(hierarchy, class_nodes).to(device)
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr, "group_name": "vision_lora"},
        {"params": [proxies], "lr": args.proxy_lr, "group_name": "metric_proxy"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(device.startswith("cuda") and args.precision in {"fp16", "float16"})
    trainable_params = lora_params + [proxies]

    print(
        "image-only metric Vision LoRA training: "
        f"modules={len(replaced_modules)}, classes={len(train_dataset.classes)}, "
        f"P={args.classes_per_batch}, K={args.examples_per_class}, "
        f"lora_params={sum(parameter.numel() for parameter in lora_params)}"
    )

    history = []
    best_bacc = float("-inf")
    best_epoch = None
    best_lora_state = None
    best_proxy_state = None
    for epoch in range(1, args.epochs + 1):
        batch_sampler.set_epoch(epoch)
        clip_model.eval()
        set_vision_lora_enabled(clip_model, True)
        set_vision_lora_train_mode(clip_model, True)
        totals: dict[str, float] = {}
        steps = 0
        iterator = tqdm(train_loader, desc=f"metric epoch {epoch}/{args.epochs}", leave=False) if tqdm else train_loader
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            base_features = encode_base_features(args, clip_model, images, device)
            with autocast_context(args, device):
                tuned_features = clip_model.get_image_features(pixel_values=images)
            supcon_loss, supcon_stats = supervised_contrastive_loss(
                tuned_features, targets, temperature=args.supcon_temperature
            )
            triplet_loss, triplet_stats = batch_hard_hierarchical_triplet_loss(
                tuned_features,
                targets,
                class_distances,
                base_margin=args.triplet_base_margin,
                hierarchy_margin=args.triplet_hierarchy_margin,
            )
            proxy_loss, proxy_stats = cosine_proxy_loss(
                tuned_features,
                proxies,
                targets,
                temperature=args.proxy_temperature,
                margin=args.proxy_margin,
            )
            retention_loss = (
                1.0 - torch.nn.functional.cosine_similarity(
                    tuned_features.float(), base_features.detach().float(), dim=-1
                )
            ).mean()
            loss = (
                args.lambda_supcon * supcon_loss
                + args.lambda_triplet * triplet_loss
                + args.lambda_proxy * proxy_loss
                + args.lambda_retention * retention_loss
            )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                optimizer.step()
            stats = {
                **supcon_stats,
                **triplet_stats,
                **proxy_stats,
                "loss": float(loss.detach().cpu()),
                "retention_loss": float(retention_loss.detach().cpu()),
            }
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
        scheduler.step()
        epoch_stats = {key: value / max(1, steps) for key, value in totals.items()}
        epoch_stats.update({"epoch": epoch, "steps": steps})
        for group in optimizer.param_groups:
            epoch_stats[f"{group['group_name']}_lr"] = group["lr"]

        if epoch % args.validation_every_n_epochs == 0 or epoch == args.epochs:
            validation = evaluate_proxy(args, clip_model, proxies, val_loader, device)
            epoch_stats.update({f"val_{key}": value for key, value in validation.items()})
            if validation["balanced_acc"] > best_bacc:
                best_bacc = validation["balanced_acc"]
                best_epoch = epoch
                best_lora_state = clone_state(vision_lora_state_dict(clip_model))
                best_proxy_state = proxies.detach().cpu().clone()
                save_checkpoint(
                    args,
                    args.checkpoint,
                    lora_cfg,
                    clip_model,
                    proxies,
                    train_dataset.classes,
                    {"train_history": history + [epoch_stats], "best_validation": {"epoch": epoch, **validation}},
                )
        history.append(epoch_stats)
        print(
            f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, "
            f"supcon={epoch_stats['supcon_loss']:.6f}, "
            f"triplet={epoch_stats['triplet_loss']:.6f}, "
            f"retention={epoch_stats['retention_loss']:.6f}, "
            f"val_bacc={epoch_stats.get('val_balanced_acc', float('nan')):.6f}"
        )

    save_checkpoint(
        args,
        args.last_checkpoint,
        lora_cfg,
        clip_model,
        proxies,
        train_dataset.classes,
        {"train_history": history, "best_validation": {"epoch": best_epoch, "balanced_acc": best_bacc}},
    )
    if best_lora_state is None or best_proxy_state is None:
        raise RuntimeError("Metric training did not produce a validation checkpoint")
    load_vision_lora_state_dict(clip_model, best_lora_state)
    with torch.no_grad():
        proxies.copy_(best_proxy_state.to(device))
    final_validation = evaluate_proxy(args, clip_model, proxies, val_loader, device)
    metrics = {
        "train_history": history,
        "best_validation": {"epoch": best_epoch, **final_validation},
        "used_text_during_training": False,
        "used_ood_for_training_or_selection": False,
    }
    save_checkpoint(
        args,
        args.checkpoint,
        lora_cfg,
        clip_model,
        proxies,
        train_dataset.classes,
        metrics,
    )
    save_json(args.diagnostics_path, metrics)
    print(f"best epoch: {best_epoch}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved diagnostics: {args.diagnostics_path}")
    print(f"ID proxy BAcc: {final_validation['balanced_acc']:.6f}")


if __name__ == "__main__":
    main()
