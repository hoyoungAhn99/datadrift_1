from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMPT_SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(CMPT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CMPT_SOURCE_ROOT))

from sacil.engine.table1_trainer import UnifiedTable1Trainer  # noqa: E402


def test_resume_config_ignores_machine_local_paths_and_device() -> None:
    windows = {
        "_config_path": r"C:\\workspace\\configs\\train_cscct.yaml",
        "device": "cuda:1",
        "runtime": {"cpu_threads": 6},
        "training": {"batch_size": 128},
    }
    linux = {
        "_config_path": "/home/user/workspace/configs/train_cscct.yaml",
        "device": "cuda:0",
        "runtime": {"cpu_threads": 12},
        "training": {"batch_size": 128},
    }

    normalize = UnifiedTable1Trainer._resume_compatible_config
    assert normalize(windows) == normalize(linux)


def test_resume_config_still_rejects_trajectory_changes() -> None:
    source = {
        "_config_path": r"C:\\workspace\\configs\\train_afc.yaml",
        "training": {"batch_size": 64},
    }
    target = {
        "_config_path": "/home/user/workspace/configs/train_afc.yaml",
        "training": {"batch_size": 128},
    }

    normalize = UnifiedTable1Trainer._resume_compatible_config
    assert normalize(source) != normalize(target)
