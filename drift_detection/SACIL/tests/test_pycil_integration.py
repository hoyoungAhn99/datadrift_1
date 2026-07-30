from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from sacil.pycil.runtime import (
    activate_pycil,
    load_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYCIL_ROOT = PROJECT_ROOT / "ref_codes" / "00_frameworks" / "PyCIL"


class _TensorTripletDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            index,
            torch.from_numpy(self.data[index]),
            int(self.targets[index]),
        )


class _TinyDataManager:
    def __init__(self) -> None:
        generator = np.random.default_rng(7)
        self.data = generator.normal(
            size=(16, 3, 32, 32)
        ).astype(np.float32)
        self.targets = np.repeat(np.arange(4), 4)
        self.nb_tasks = 2

    @staticmethod
    def get_task_size(task: int) -> int:
        if task not in (0, 1):
            raise IndexError(task)
        return 2

    def get_dataset(
        self,
        indices,
        source,
        mode,
        appendent=None,
        ret_data=False,
        **_,
    ):
        del source, mode
        pieces_data = []
        pieces_targets = []
        for class_id in indices:
            mask = self.targets == int(class_id)
            pieces_data.append(self.data[mask])
            pieces_targets.append(self.targets[mask])
        if appendent is not None and len(appendent) != 0:
            pieces_data.append(np.asarray(appendent[0]))
            pieces_targets.append(np.asarray(appendent[1]))
        if not pieces_data:
            raise ValueError("tiny dataset request is empty")
        data = np.concatenate(pieces_data)
        targets = np.concatenate(pieces_targets)
        dataset = _TensorTripletDataset(data, targets)
        if ret_data:
            return data, targets, dataset
        return dataset


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_runtime_registers_sacil_and_explicit_order(tmp_path):
    config = load_experiment_config(
        "configs/pycil/cifar100/sacil_b50_inc5.json",
        project_root=PROJECT_ROOT,
    )
    config["artifact_dir"] = str(tmp_path)
    resolved = activate_pycil(config, pycil_root=PYCIL_ROOT)

    from utils import factory

    args = dict(resolved)
    args["device"] = [torch.device("cpu")]
    learner = factory.get_model("sacil", args)
    assert type(learner).__name__ == "SACIL"
    assert learner.feature_dim == 64
    assert resolved["shuffle"] is False
    assert resolved["_explicit_class_order"][:5] == [87, 0, 52, 58, 44]


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_sacil_two_task_smoke(tmp_path):
    activate_pycil(
        {
            "dataset": "cifar100",
            "shuffle": False,
        },
        pycil_root=PYCIL_ROOT,
    )
    from utils import factory

    args = {
        "memory_size": 4,
        "memory_per_class": 1,
        "fixed_memory": True,
        "device": [torch.device("cpu")],
        "convnet_type": "resnet32",
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "init_epochs": 1,
        "epochs": 1,
        "init_milestones": [],
        "milestones": [],
        "eval_interval": 0,
        "disable_tqdm": True,
        "save_checkpoints": False,
        "artifact_dir": str(tmp_path),
        "lambda_kd": 1.0,
        "lambda_geo": 1.0,
    }
    learner = factory.get_model("sacil", args)
    manager = _TinyDataManager()

    learner.incremental_train(manager)
    learner.after_task()
    learner.incremental_train(manager)

    assert learner._total_classes == 4
    assert learner.exemplar_size == 4
    assert learner._geometry_loss is not None
    assert learner._conflict_weights is not None
    assert learner._artifacts.tree.num_leaves == 4
    assert (tmp_path / "tree_task_00.json").is_file()
    assert (tmp_path / "tree_task_01.json").is_file()
