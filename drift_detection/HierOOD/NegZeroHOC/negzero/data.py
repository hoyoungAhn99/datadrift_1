from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import gen_datasets, get_id_classes


@dataclass
class NegZeroDataBundle:
    id_classes: list[str]
    hierarchy: Hierarchy
    train_dataset: object
    val_dataset: object
    ood_dataset: object
    train_loader: DataLoader
    val_loader: DataLoader
    ood_loader: DataLoader


def dataset_transform_kwargs(config: dict) -> dict:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    return {
        "preset": dataset_cfg.get("transform_preset", "clip"),
        "mean": dataset_cfg.get("mean"),
        "std": dataset_cfg.get("std"),
        "resize": dataset_cfg.get("resize"),
        "cropsize": dataset_cfg.get("cropsize"),
        "model_name": model_cfg["name"],
    }


def filter_unreadable_images(dataset, split_name: str, fallback_dir: str | None = None):
    fallback_root = Path(fallback_dir) if fallback_dir else None
    kept = []
    removed = []
    replaced = 0
    for path, target in dataset.samples:
        candidate_path = Path(path)
        if fallback_root is not None and (
            not candidate_path.exists() or candidate_path.stat().st_size <= 0
        ):
            fallback_path = fallback_root / candidate_path.name
            if fallback_path.exists():
                candidate_path = fallback_path
                replaced += 1
        try:
            if candidate_path.stat().st_size <= 0:
                raise OSError("empty image file")
            with Image.open(candidate_path) as image:
                image.verify()
            kept.append((str(candidate_path), target))
        except Exception as exc:
            removed.append((path, type(exc).__name__, str(exc)))

    if replaced:
        print(f"Using fallback image paths for {replaced} {split_name} images.")
    if removed:
        print(f"Skipping {len(removed)} unreadable {split_name} images.")
        for path, err_type, message in removed[:10]:
            print(f"  {err_type}: {path} ({message})")
    if replaced or removed:
        dataset.samples = kept
        dataset.imgs = kept
        dataset.targets = [target for _, target in kept]
    return dataset


def _build_loader(dataset, dataloader_cfg: dict, *, train: bool) -> DataLoader:
    batch_size = dataloader_cfg["batch_size"] if train else dataloader_cfg.get(
        "eval_batch_size",
        dataloader_cfg["batch_size"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(dataloader_cfg.get("shuffle_train", True)) if train else False,
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        pin_memory=bool(dataloader_cfg.get("pin_memory", False)),
        drop_last=bool(dataloader_cfg.get("drop_last", False)) if train else False,
    )


def build_data_bundle(config: dict) -> NegZeroDataBundle:
    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    train_ds, val_ds, ood_ds = gen_datasets(
        dataset_cfg["datadir"],
        id_classes,
        hierarchy.ood_train_classes,
        **dataset_transform_kwargs(config),
    )
    fallback_dir = dataset_cfg.get("image_fallback_dir")
    train_ds = filter_unreadable_images(train_ds, "train", fallback_dir=fallback_dir)
    val_ds = filter_unreadable_images(val_ds, "val", fallback_dir=fallback_dir)
    ood_ds = filter_unreadable_images(ood_ds, "ood", fallback_dir=fallback_dir)

    dataloader_cfg = config.get("dataloader", {})
    return NegZeroDataBundle(
        id_classes=id_classes,
        hierarchy=hierarchy,
        train_dataset=train_ds,
        val_dataset=val_ds,
        ood_dataset=ood_ds,
        train_loader=_build_loader(train_ds, dataloader_cfg, train=True),
        val_loader=_build_loader(val_ds, dataloader_cfg, train=False),
        ood_loader=_build_loader(ood_ds, dataloader_cfg, train=False),
    )

