from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SplitInfo:
    id_classes: tuple[int, ...]
    ood_classes: tuple[int, ...]


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, input_kind: str):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.input_kind = input_kind

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _split_id_train_val(
    x: np.ndarray,
    y: np.ndarray,
    id_classes: tuple[int, ...],
    val_ratio: float,
    seed: int,
):
    id_mask = np.isin(y, id_classes)
    x_id = x[id_mask]
    y_id = y[id_mask]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y_id))
    val_size = int(round(len(indices) * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x_id[train_idx], y_id[train_idx], x_id[val_idx], y_id[val_idx]


def _split_test(x: np.ndarray, y: np.ndarray, id_classes: tuple[int, ...]):
    id_mask = np.isin(y, id_classes)
    return x[id_mask], y[id_mask], x[~id_mask], y[~id_mask]


def _load_fashionmnist(root: Path):
    base = root / "FashionMNIST"
    train = pd.read_csv(base / "fashion-mnist_train.csv")
    test = pd.read_csv(base / "fashion-mnist_test.csv")
    x_train = train.drop(columns=["label"]).to_numpy(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_train = train["label"].to_numpy(np.int64)
    x_test = test.drop(columns=["label"]).to_numpy(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_test = test["label"].to_numpy(np.int64)
    split = SplitInfo(id_classes=tuple(range(7)), ood_classes=(7, 8, 9))
    return x_train, y_train, x_test, y_test, split, "image"


def _load_usps(root: Path):
    with h5py.File(root / "usps.h5", "r") as f:
        x_train = f["train"]["data"][:].astype(np.float32).reshape(-1, 1, 16, 16)
        y_train = f["train"]["target"][:].astype(np.int64)
        x_test = f["test"]["data"][:].astype(np.float32).reshape(-1, 1, 16, 16)
        y_test = f["test"]["target"][:].astype(np.int64)
    split = SplitInfo(id_classes=tuple(range(7)), ood_classes=(7, 8, 9))
    return x_train, y_train, x_test, y_test, split, "image"


def _load_reuters10k(root: Path):
    train = np.load(root / "reuters-10k" / "train.npy", allow_pickle=True).item()
    test = np.load(root / "reuters-10k" / "test.npy", allow_pickle=True).item()
    x_train = train["data"].astype(np.float32)
    y_train = train["label"].astype(np.int64)
    x_test = test["data"].astype(np.float32)
    y_test = test["label"].astype(np.int64)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    classes, counts = np.unique(y_train, return_counts=True)
    id_classes = tuple(int(c) for c in classes[np.argsort(counts)[::-1][:2]])
    ood_classes = tuple(int(c) for c in classes if int(c) not in id_classes)
    split = SplitInfo(id_classes=id_classes, ood_classes=ood_classes)
    return x_train, y_train, x_test, y_test, split, "tabular"


def build_datasets(config: dict):
    dataset_name = config["dataset"]["name"].lower()
    root = Path(config["dataset"].get("root", "dataset"))
    val_ratio = float(config["dataset"].get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))

    if dataset_name == "fashionmnist":
        x_train, y_train, x_test, y_test, split, input_kind = _load_fashionmnist(root)
    elif dataset_name == "usps":
        x_train, y_train, x_test, y_test, split, input_kind = _load_usps(root)
    elif dataset_name in {"reuters10k", "reuters-10k"}:
        x_train, y_train, x_test, y_test, split, input_kind = _load_reuters10k(root)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")

    tr_x, tr_y, val_x, val_y = _split_id_train_val(
        x_train, y_train, split.id_classes, val_ratio, seed
    )
    id_x, id_y, ood_x, ood_y = _split_test(x_test, y_test, split.id_classes)

    return {
        "train": ArrayDataset(tr_x, tr_y, input_kind),
        "val": ArrayDataset(val_x, val_y, input_kind),
        "id_test": ArrayDataset(id_x, id_y, input_kind),
        "ood_test": ArrayDataset(ood_x, ood_y, input_kind),
        "split_info": split,
        "input_shape": tuple(tr_x.shape[1:]),
        "input_dim": int(np.prod(tr_x.shape[1:])),
        "input_kind": input_kind,
    }


def build_dataloaders(config: dict):
    datasets = build_datasets(config)
    batch_size = int(config["training"].get("batch_size", 128))
    num_workers = int(config["training"].get("num_workers", 0))
    loaders = {
        key: DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=num_workers,
        )
        for key in ("train", "val", "id_test", "ood_test")
    }
    return datasets, loaders
