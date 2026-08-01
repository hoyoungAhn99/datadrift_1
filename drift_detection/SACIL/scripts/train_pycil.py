from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sacil.pycil import (  # noqa: E402
    DEFAULT_PYCIL_ROOT,
    activate_pycil,
    load_experiment_config,
    run_pycil_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SACIL or a stock baseline through official PyCIL."
    )
    parser.add_argument("--config", required=True, help="PyCIL JSON config")
    parser.add_argument(
        "--pycil-root",
        default=str(DEFAULT_PYCIL_ROOT),
        help="Path to the official PyCIL checkout",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Base path for config-relative paths",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Override the config with one PyCIL CUDA device index",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the config with one random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and learner route without training",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    project_root = Path(cli.project_root).expanduser().resolve()
    config = load_experiment_config(
        cli.config, project_root=project_root
    )
    if cli.device is not None:
        config["device"] = [cli.device]
    if cli.seed is not None:
        config["seed"] = [cli.seed]
    pycil_root = Path(cli.pycil_root).expanduser()
    if not pycil_root.is_absolute():
        pycil_root = project_root / pycil_root
    if cli.dry_run:
        import torch

        resolved = activate_pycil(config, pycil_root=pycil_root)
        from utils import factory

        resolved["device"] = [torch.device("cpu")]
        learner = factory.get_model(resolved["model_name"], resolved)
        print(f"implementation_source={resolved.get('implementation_source')}")
        print(
            "learner="
            f"{learner.__class__.__module__}.{learner.__class__.__name__}"
        )
        print(
            f"protocol=init{resolved['init_cls']}/inc{resolved['increment']}"
        )
        return
    run_pycil_experiment(config, pycil_root=pycil_root)


if __name__ == "__main__":
    main()
