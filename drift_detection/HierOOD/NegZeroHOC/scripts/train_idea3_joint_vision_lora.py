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
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.runtime import available_device, configured_device
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_train_mode,
    vision_lora_parameters,
    vision_lora_state_dict,
)
from scripts.train_idea3_positive_prompts import backward_sparse_path_bottleneck_streaming


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    prompt_cfg = cfg.get("prompt", {})
    dataloader_cfg = cfg.get("dataloader", {})
    augmentation_cfg = cfg.get("augmentation", {})
    lora_cfg = cfg.get("vision_lora", {})
    train_cfg = cfg.get("joint_training", {})
    loss_cfg = train_cfg.get("loss", {})
    validation_cfg = train_cfg.get("validation", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea3-joint-vision-lora")
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    checkpoint = Path(
        train_cfg.get("checkpoint")
        or output_root / "checkpoints" / f"{experiment_name}-positive.pt"
    )
    last_checkpoint = Path(
        validation_cfg.get("last_checkpoint")
        or checkpoint.with_name(f"{checkpoint.stem}-last{checkpoint.suffix}")
    )

    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=str(datadir),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch16"),
        tokenizer_model=clip_cfg.get("tokenizer_model", clip_cfg.get("model", "openai/clip-vit-base-patch16")),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        prompt=prompt_cfg,
        vision_lora=lora_cfg,
        augmentation=augmentation_cfg,
        train_batch_size=int(train_cfg.get("batch_size", dataloader_cfg.get("batch_size", 16))),
        eval_batch_size=int(validation_cfg.get("batch_size", inference_cfg.get("batch_size", 64))),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        epochs=int(train_cfg.get("epochs", 50)),
        prompt_lr=float(train_cfg.get("prompt_lr", 1e-3)),
        vision_lora_lr=float(train_cfg.get("vision_lora_lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        loss_bottleneck_weight=float(loss_cfg.get("bottleneck_weight", 0.5)),
        loss_bottleneck_temperature=float(loss_cfg.get("bottleneck_temperature", 0.5)),
        loss_route_margin=float(loss_cfg.get("route_margin", 0.05)),
        loss_margin_weight=float(loss_cfg.get("margin_weight", 0.25)),
        validation_enabled=bool(validation_cfg.get("enabled", True)),
        validation_every_n_epochs=max(1, int(validation_cfg.get("every_n_epochs", 1))),
        validation_start_epoch=max(1, int(validation_cfg.get("start_epoch", 1))),
        validation_mode=validation_cfg.get("mode", "positive_global_path"),
        inference_mode=inference_cfg.get("mode", "positive_global_path"),
        inference_tau=float(inference_cfg.get("tau", 1.0 / float(train_cfg.get("tau", 0.07)))),
        checkpoint=str(checkpoint),
        last_checkpoint=str(last_checkpoint),
        result_path=str(
            train_cfg.get("result_path")
            or output_root / "results" / f"{experiment_name}-positive_global_path.result"
        ),
        diagnostics_path=str(
            train_cfg.get("diagnostics_path")
            or output_root / "diagnostics" / f"{experiment_name}-positive-diagnostics.json"
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def build_transforms(args):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_size = int(args.augmentation.get("image_size", 224))
    resize_size = int(args.augmentation.get("eval_resize", image_size))
    scale = tuple(float(value) for value in args.augmentation.get("random_resized_crop_scale", [0.7, 1.0]))
    flip_probability = float(args.augmentation.get("horizontal_flip_probability", 0.5))
    jitter_cfg = args.augmentation.get("color_jitter", {})
    jitter_probability = float(jitter_cfg.get("probability", 0.0))

    train_steps = [
        transforms.RandomResizedCrop(
            image_size,
            scale=scale,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=flip_probability),
    ]
    if jitter_probability > 0.0:
        jitter = transforms.ColorJitter(
            brightness=float(jitter_cfg.get("brightness", 0.0)),
            contrast=float(jitter_cfg.get("contrast", 0.0)),
            saturation=float(jitter_cfg.get("saturation", 0.0)),
            hue=float(jitter_cfg.get("hue", 0.0)),
        )
        train_steps.append(transforms.RandomApply([jitter], p=jitter_probability))
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return transforms.Compose(train_steps), eval_transform


def build_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    train_transform, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    train_dataset = SubsetImageFolder(datadir / "train", id_classes, transform=train_transform)
    val_dataset = SubsetImageFolder(datadir / "val", id_classes, transform=eval_transform)
    ood_dataset = SubsetImageFolder(
        datadir / "val",
        hierarchy.ood_train_classes,
        transform=eval_transform,
    )
    empty_files = sorted({
        path
        for dataset in (train_dataset, val_dataset, ood_dataset)
        for path, _ in dataset.samples
        if Path(path).stat().st_size == 0
    })
    if empty_files:
        preview = ", ".join(empty_files[:3])
        raise RuntimeError(
            f"Raw-image joint training found {len(empty_files)} zero-byte image files "
            f"under {datadir}. Populate the real dataset before training. Examples: {preview}"
        )
    print(f"# ID Train: {len(train_dataset)}")
    print(f"# ID Val: {len(val_dataset)}")
    print(f"# OOD: {len(ood_dataset)}")
    return train_dataset, val_dataset, ood_dataset


def build_eval_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    _, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    val_dataset = SubsetImageFolder(datadir / "val", id_classes, transform=eval_transform)
    ood_dataset = SubsetImageFolder(
        datadir / "val",
        hierarchy.ood_train_classes,
        transform=eval_transform,
    )
    empty_files = sorted({
        path
        for dataset in (val_dataset, ood_dataset)
        for path, _ in dataset.samples
        if Path(path).stat().st_size == 0
    })
    if empty_files:
        preview = ", ".join(empty_files[:3])
        raise RuntimeError(
            f"Raw-image inference found {len(empty_files)} zero-byte image files "
            f"under {datadir}. Populate the real dataset before inference. Examples: {preview}"
        )
    print(f"# ID Val: {len(val_dataset)}")
    print(f"# OOD: {len(ood_dataset)}")
    return val_dataset, ood_dataset


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def load_clip_and_tokenizer(args, device):
    from transformers import CLIPModel, CLIPTokenizerFast

    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        local_files_only=args.local_files_only,
    ).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(
        args.tokenizer_model,
        local_files_only=args.local_files_only,
    )
    return clip_model, tokenizer


def prompt_only_state_dict(learner: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in learner.state_dict().items()
        if not key.startswith("text_encoder.")
    }


def load_prompt_only_state_dict(learner: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = learner.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected positive prompt checkpoint keys: {unexpected}")


def prompt_parameters(learner: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in learner.named_parameters()
        if not name.startswith("text_encoder.") and parameter.requires_grad
    ]


def target_leaf_nodes(hierarchy, classes: list[str], targets: torch.Tensor) -> list[str]:
    ds_to_node = hierarchy.gen_ds2node_map(classes)
    node_indices = ds_to_node[targets.detach().cpu().long()]
    return [hierarchy.id_node_list[int(index)] for index in node_indices.tolist()]


def set_joint_train_mode(clip_model, learner):
    learner.train()
    clip_model.eval()
    set_vision_lora_train_mode(clip_model, True)


def autocast_context(args, device):
    enabled = device.startswith("cuda") and args.precision in {"fp16", "float16"}
    return torch.cuda.amp.autocast(enabled=enabled)


@torch.no_grad()
def evaluate_split_raw(
    args,
    hierarchy,
    clip_model,
    learner,
    dataset,
    loader,
    device,
    split_name: str,
    inference_mode: str | None = None,
):
    inference_mode = inference_mode or args.inference_mode
    clip_model.eval()
    learner.eval()
    set_vision_lora_train_mode(clip_model, False)
    semantic_index = build_idea3_semantic_index(
        hierarchy,
        learner,
        mode=inference_mode,
    )

    predictions = []
    targets = []
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            image_features = clip_model.get_image_features(pixel_values=images)
        output = predict_features_idea3(
            image_features,
            hierarchy,
            semantic_index,
            mode=inference_mode,
            tau=args.inference_tau,
        )
        predictions.append(output["preds"].cpu())
        targets.append(batch_targets.cpu())

    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    payload = {
        "classes": list(dataset.classes),
        "targets": target_tensor,
    }
    node_targets, metrics = evaluate_split(
        hierarchy,
        payload,
        prediction_tensor,
        dists_mats=make_distance_mats(hierarchy),
    )
    return {
        "preds": prediction_tensor,
        "targets": node_targets.cpu(),
        "metrics": metrics,
        "split": split_name,
    }


def save_joint_checkpoint(
    args,
    path,
    hierarchy,
    prompt_cfg,
    lora_cfg,
    learner,
    clip_model,
    metrics,
):
    return save_idea3_checkpoint(
        path,
        stage="positive_joint_vision_lora",
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=prompt_only_state_dict(learner),
        vision_lora_config=lora_cfg.to_dict(),
        vision_lora_state_dict=vision_lora_state_dict(clip_model),
        metrics=metrics,
        args=vars(args),
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    train_dataset, val_dataset, ood_dataset = build_datasets(args, hierarchy)
    train_loader = make_loader(
        train_dataset,
        args.train_batch_size,
        args.num_workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    ood_loader = make_loader(
        ood_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    clip_model, tokenizer = load_clip_and_tokenizer(args, device)
    if args.gradient_checkpointing and hasattr(clip_model, "gradient_checkpointing_enable"):
        clip_model.gradient_checkpointing_enable()

    lora_cfg = VisionLoRAConfig.from_dict(args.vision_lora)
    replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    text_encoder = SoftPromptTextEncoder(
        clip_model,
        tokenizer,
        max_length=int(args.prompt.get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(args.prompt)
    learner = PositivePromptLearner(
        args.dataset,
        hierarchy,
        text_encoder,
        prompt_cfg,
    ).to(device)

    prompt_params = prompt_parameters(learner)
    lora_params = vision_lora_parameters(clip_model)
    optimizer = torch.optim.AdamW(
        [
            {"params": prompt_params, "lr": args.prompt_lr},
            {"params": lora_params, "lr": args.vision_lora_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    use_scaler = device.startswith("cuda") and args.precision in {"fp16", "float16"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    identity_adapter = nn.Identity()

    print(
        "joint Vision LoRA training: "
        f"replaced_modules={len(replaced_modules)}, "
        f"prompt_params={sum(parameter.numel() for parameter in prompt_params)}, "
        f"vision_lora_params={sum(parameter.numel() for parameter in lora_params)}, "
        f"batch_size={args.train_batch_size}, precision={args.precision}"
    )

    history = []
    best_bacc = float("-inf")
    best_epoch = None
    best_prompt_state = None
    best_lora_state = None

    for epoch in range(1, args.epochs + 1):
        set_joint_train_mode(clip_model, learner)
        epoch_loss = 0.0
        epoch_path_acc = 0.0
        epoch_local_acc = 0.0
        steps = 0
        iterator = tqdm(
            train_loader,
            desc=f"joint epoch {epoch}/{args.epochs}",
            leave=False,
        ) if tqdm else train_loader

        for images, batch_targets in iterator:
            images = images.to(device, non_blocking=True)
            leaf_nodes = target_leaf_nodes(hierarchy, train_dataset.classes, batch_targets)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(args, device):
                image_features = clip_model.get_image_features(pixel_values=images)
            stats = backward_sparse_path_bottleneck_streaming(
                args,
                hierarchy,
                learner,
                identity_adapter,
                image_features,
                leaf_nodes,
                grad_scaler=scaler,
            )

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(prompt_params + lora_params, args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(prompt_params + lora_params, args.gradient_clip_norm)
                optimizer.step()

            epoch_loss += stats["loss"]
            epoch_path_acc += stats["path_acc"]
            epoch_local_acc += stats["local_acc"]
            steps += 1

        scheduler.step()
        epoch_stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "path_acc": epoch_path_acc / max(1, steps),
            "local_acc": epoch_local_acc / max(1, steps),
            "prompt_lr": optimizer.param_groups[0]["lr"],
            "vision_lora_lr": optimizer.param_groups[1]["lr"],
            "steps": steps,
        }

        validation_due = (
            args.validation_enabled
            and epoch >= args.validation_start_epoch
            and (epoch - args.validation_start_epoch) % args.validation_every_n_epochs == 0
        )
        is_best = False
        if validation_due:
            val_result = evaluate_split_raw(
                args,
                hierarchy,
                clip_model,
                learner,
                val_dataset,
                val_loader,
                device,
                "val",
                inference_mode=args.validation_mode,
            )
            val_bacc = float(val_result["metrics"]["balanced_acc"])
            val_bmhd = float(val_result["metrics"]["balanced_hdist"])
            epoch_stats["val_balanced_acc"] = val_bacc
            epoch_stats["val_balanced_hdist"] = val_bmhd
            if val_bacc > best_bacc:
                best_bacc = val_bacc
                best_epoch = epoch
                best_prompt_state = prompt_only_state_dict(learner)
                best_lora_state = vision_lora_state_dict(clip_model)
                is_best = True

        history.append(epoch_stats)
        message = (
            f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, "
            f"path_acc={epoch_stats['path_acc']:.6f}, "
            f"local_acc={epoch_stats['local_acc']:.6f}"
        )
        if validation_due:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"val_bmhd={epoch_stats['val_balanced_hdist']:.6f}"
            )
            if is_best:
                message += " [best]"
        print(message)

        if is_best:
            save_joint_checkpoint(
                args,
                args.checkpoint,
                hierarchy,
                prompt_cfg,
                lora_cfg,
                learner,
                clip_model,
                {
                    "train_history": history,
                    "best_validation": {
                        "epoch": best_epoch,
                        "val_balanced_acc": best_bacc,
                    },
                },
            )

    last_metrics = {
        "train_history": history,
        "best_validation": {
            "epoch": best_epoch,
            "val_balanced_acc": best_bacc if math.isfinite(best_bacc) else None,
        },
    }
    save_joint_checkpoint(
        args,
        args.last_checkpoint,
        hierarchy,
        prompt_cfg,
        lora_cfg,
        learner,
        clip_model,
        last_metrics,
    )

    if best_prompt_state is None or best_lora_state is None:
        best_epoch = args.epochs
        best_prompt_state = prompt_only_state_dict(learner)
        best_lora_state = vision_lora_state_dict(clip_model)
    load_prompt_only_state_dict(learner, best_prompt_state)
    load_vision_lora_state_dict(clip_model, best_lora_state)

    val_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        val_dataset,
        val_loader,
        device,
        "val",
    )
    ood_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        ood_dataset,
        ood_loader,
        device,
        "ood",
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])
    result = {
        "args": vars(args),
        "mode": args.inference_mode,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": "positive_joint_vision_lora",
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)

    final_metrics = {
        "train_history": history,
        "best_validation": {
            "epoch": best_epoch,
            "val_balanced_acc": float(val_result["metrics"]["balanced_acc"]),
            "val_balanced_hdist": float(val_result["metrics"]["balanced_hdist"]),
        },
        "final": {
            "val_balanced_acc": float(val_result["metrics"]["balanced_acc"]),
            "ood_balanced_acc": float(ood_result["metrics"]["balanced_acc"]),
            "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
            "val_balanced_hdist": float(val_result["metrics"]["balanced_hdist"]),
            "ood_balanced_hdist": float(ood_result["metrics"]["balanced_hdist"]),
            "mixed_balanced_hdist": float(mixed["mixed_balanced_hdist"]),
        },
    }
    save_joint_checkpoint(
        args,
        args.checkpoint,
        hierarchy,
        prompt_cfg,
        lora_cfg,
        learner,
        clip_model,
        final_metrics,
    )
    save_json(args.diagnostics_path, final_metrics)

    print(f"best epoch: {best_epoch}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved last checkpoint: {args.last_checkpoint}")
    print(f"saved result: {args.result_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"ID BMHD: {float(val_result['metrics']['balanced_hdist']):.6f}")


if __name__ == "__main__":
    main()
