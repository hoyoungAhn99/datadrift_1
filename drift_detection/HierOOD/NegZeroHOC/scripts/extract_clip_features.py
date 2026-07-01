from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ProHOC"))

from libs.utils.dataset_util import SubsetImageFolder, get_id_classes

from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import ensure_dir, save_feature_file, save_json


class ImageFolderWithPaths(SubsetImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def collate_pil(batch):
    images, targets, paths = zip(*batch)
    return list(images), torch.tensor(targets, dtype=torch.long), list(paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fgvc-aircraft")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--hierarchy", default="hierarchies/fgvc-aircraft.json")
    parser.add_argument("--id_split", default="data/fgvc-aircraft-id-labels.csv")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def encode_split(backend, split_name, dataset, loader, clip_model):
    feats = []
    targets = []
    paths = []
    for images, batch_targets, batch_paths in loader:
        batch_feats = backend.encode_images(images).cpu()
        feats.append(batch_feats)
        targets.append(batch_targets.cpu())
        paths.extend(batch_paths)

    return {
        "features": torch.cat(feats, dim=0),
        "targets": torch.cat(targets, dim=0),
        "classes": list(dataset.classes),
        "paths": paths,
        "split": split_name,
        "clip_model": clip_model,
    }


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    id_classes = get_id_classes(args.id_split)
    ood_classes = hierarchy.ood_train_classes

    datadir = Path(args.datadir)
    train_ds = ImageFolderWithPaths(datadir / "train", id_classes, transform=None)
    val_ds = ImageFolderWithPaths(datadir / "val", id_classes, transform=None)
    ood_ds = ImageFolderWithPaths(datadir / "val", ood_classes, transform=None)

    backend = ClipBackend(args.clip_model, device=args.device, local_files_only=args.local_files_only)

    model_key = f"clip_{safe_model_name(args.clip_model)}"
    feature_dir = ensure_dir(Path(args.outdir) / "features" / args.dataset / model_key)
    split_items = [("val", val_ds), ("ood", ood_ds)]
    if not args.skip_train:
        split_items.insert(0, ("train", train_ds))

    for split_name, dataset in split_items:
        print(f"Encoding {split_name}: {len(dataset)} images")
        payload = encode_split(
            backend,
            split_name,
            dataset,
            make_loader(dataset, args),
            args.clip_model,
        )
        save_path = feature_dir / f"{split_name}-features.pt"
        save_feature_file(save_path, payload)
        print(f"saved: {save_path}")

    save_json(
        feature_dir / "meta.json",
        {
            "dataset": args.dataset,
            "clip_model": args.clip_model,
            "model_key": model_key,
            "hierarchy": str(args.hierarchy),
            "id_split": str(args.id_split),
            "datadir": str(args.datadir),
            "num_id_classes": len(id_classes),
            "num_ood_classes": len(ood_classes),
        },
    )


if __name__ == "__main__":
    main()
