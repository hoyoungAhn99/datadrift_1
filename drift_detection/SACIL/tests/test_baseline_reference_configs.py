from __future__ import annotations

from pathlib import Path

import pytest

from sacil.config import deep_update, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "table1" / "cifar100"


def _resolved(name: str) -> dict:
    path = CONFIG_ROOT / name
    child = load_config(path)
    parent = child.pop("extends", None)
    if parent is not None:
        child = deep_update(load_config(path.parent / parent), child)
    return child


@pytest.mark.parametrize(
    ("name", "epochs", "milestones"),
    [
        ("finetune_nme_b50_inc5_resnet32.yaml", 80, [40, 70]),
        ("replay_nme_b50_inc5_resnet32.yaml", 70, [30, 50]),
    ],
)
def test_control_baselines_use_their_reference_incremental_recipe(
    name: str, epochs: int, milestones: list[int]
):
    config = _resolved(name)
    incremental = config["training"]["incremental"]
    assert incremental == {
        "epochs": epochs,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 2e-4,
        "nesterov": False,
        "scheduler": "multistep",
        "milestones": milestones,
        "lr_decay": 0.1,
    }


def test_icarl_uses_original_cifar_optimization_recipe():
    config = _resolved("icarl_nme_b50_inc5_resnet32.yaml")
    assert config["training"]["batch_size"] == 128
    for phase in ("base", "incremental"):
        settings = config["training"][phase]
        assert settings["epochs"] == 70
        assert settings["lr"] == 2.0
        assert settings["momentum"] == 0.9
        assert settings["weight_decay"] == 1e-5
        assert settings["milestones"] == [49, 63]
        assert settings["lr_decay"] == 0.2


@pytest.mark.parametrize(
    "name",
    [path.name for path in CONFIG_ROOT.glob("*.yaml") if not path.name.startswith("_")],
)
def test_table1_methods_share_protocol_memory_and_augmentation(name: str):
    config = _resolved(name)
    assert config["data"]["protocol"].endswith(
        "cifar100_b50_t10_afc_order1.json"
    )
    assert config["data"]["color_jitter"] is True
    assert config["memory"]["mode"] == "per_class"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["memory"]["selection"] == "icarl_herding"
