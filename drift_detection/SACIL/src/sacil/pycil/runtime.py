from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PYCIL_ROOT = Path("ref_codes/00_frameworks/PyCIL")
_REQUIRED_PYCIL_FILES = (
    Path("trainer.py"),
    Path("models/base.py"),
    Path("utils/data_manager.py"),
    Path("utils/inc_net.py"),
)


def _resolved(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def validate_pycil_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    missing = [
        str(relative)
        for relative in _REQUIRED_PYCIL_FILES
        if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{root} is not a PyCIL checkout; missing: {', '.join(missing)}"
        )
    return root


def load_experiment_config(
    path: str | Path, *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _resolved(path, root)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("PyCIL experiment configuration must be a JSON object")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    for key in ("data_root", "class_order_path", "artifact_dir"):
        if key in config:
            config[key] = str(_resolved(config[key], root))
    return config


def _load_class_order(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("class_order") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("class-order file must contain a non-empty list")
    order = [int(value) for value in values]
    if len(set(order)) != len(order):
        raise ValueError("class order contains duplicate class IDs")
    if sorted(order) != list(range(len(order))):
        raise ValueError("class order must be a permutation of [0, N)")
    return order


def _pycil_dataset_class(data_module, dataset_name: str):
    name = dataset_name.lower()
    mapping = {
        "cifar10": data_module.iCIFAR10,
        "cifar10_aa": data_module.iCIFAR10_AA,
        "cifar100": data_module.iCIFAR100,
        "cifar100_aa": data_module.iCIFAR100_AA,
        "imagenet100": data_module.iImageNet100,
        "imagenet1000": data_module.iImageNet1000,
    }
    if name not in mapping:
        raise ValueError(f"explicit PyCIL dataset setup is unsupported: {name}")
    return mapping[name]


def _configure_class_order(config: dict[str, Any], data_module) -> None:
    order_path = config.get("class_order_path")
    if order_path is None:
        return
    order = _load_class_order(order_path)
    dataset_class = _pycil_dataset_class(data_module, config["dataset"])
    dataset_class.class_order = order
    # PyCIL uses ``idata.class_order`` only when shuffle is disabled.
    config["shuffle"] = False
    config["_explicit_class_order"] = order


def _configure_data_root(config: dict[str, Any], data_module) -> None:
    root_value = config.get("data_root")
    if root_value is None:
        return
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PyCIL data_root does not exist: {root}")

    dataset_name = str(config["dataset"]).lower()
    download = bool(config.get("download", False))
    if dataset_name in {"cifar10", "cifar10_aa"}:
        from torchvision import datasets

        def download_data(self):
            train = datasets.CIFAR10(
                str(root), train=True, download=download
            )
            test = datasets.CIFAR10(
                str(root), train=False, download=download
            )
            self.train_data = train.data
            self.train_targets = np.asarray(train.targets)
            self.test_data = test.data
            self.test_targets = np.asarray(test.targets)

    elif dataset_name in {"cifar100", "cifar100_aa"}:
        from torchvision import datasets

        def download_data(self):
            train = datasets.CIFAR100(
                str(root), train=True, download=download
            )
            test = datasets.CIFAR100(
                str(root), train=False, download=download
            )
            self.train_data = train.data
            self.train_targets = np.asarray(train.targets)
            self.test_data = test.data
            self.test_targets = np.asarray(test.targets)

    elif dataset_name in {"imagenet100", "imagenet1000"}:
        from torchvision import datasets
        from utils.toolkit import split_images_labels

        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet root must contain train/ and val/: {root}"
            )

        def download_data(self):
            train = datasets.ImageFolder(str(train_dir))
            test = datasets.ImageFolder(str(val_dir))
            self.train_data, self.train_targets = split_images_labels(
                train.imgs
            )
            self.test_data, self.test_targets = split_images_labels(test.imgs)

    else:
        raise ValueError(f"data_root patch is unsupported for {dataset_name}")

    dataset_class = _pycil_dataset_class(data_module, dataset_name)
    dataset_class.download_data = download_data


def _install_sacil_factory() -> None:
    factory = importlib.import_module("utils.factory")
    if getattr(factory, "_sacil_factory_installed", False):
        return
    original = factory.get_model

    def get_model(model_name, args):
        if str(model_name).lower() in {"sacil", "pycil_sacil"}:
            learner_module = importlib.import_module(
                "sacil.pycil.learner"
            )
            return learner_module.SACIL(args)
        return original(model_name, args)

    factory.get_model = get_model
    factory._sacil_factory_installed = True
    factory._sacil_original_get_model = original


def activate_pycil(
    config: dict[str, Any], *, pycil_root: str | Path
) -> dict[str, Any]:
    """Activate an official PyCIL checkout and register the SACIL learner."""

    root = validate_pycil_root(pycil_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # PyCIL uses top-level module names (``utils``, ``models``). Refuse to
    # silently bind to an unrelated package with the same generic name.
    for module_name in ("utils", "models"):
        loaded = sys.modules.get(module_name)
        module_file = getattr(loaded, "__file__", None)
        if module_file is not None:
            origin = Path(module_file).resolve()
            if root not in origin.parents:
                raise RuntimeError(
                    f"top-level module {module_name!r} was already loaded "
                    f"from outside PyCIL: {origin}"
                )

    data_module = importlib.import_module("utils.data")
    resolved = copy.deepcopy(config)
    _configure_data_root(resolved, data_module)
    _configure_class_order(resolved, data_module)
    _install_sacil_factory()
    resolved["_pycil_root"] = str(root)
    return resolved


def run_pycil_experiment(
    config: dict[str, Any], *, pycil_root: str | Path
) -> None:
    resolved = activate_pycil(config, pycil_root=pycil_root)
    trainer = importlib.import_module("trainer")
    trainer.train(resolved)
