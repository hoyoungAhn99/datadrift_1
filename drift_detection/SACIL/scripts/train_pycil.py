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
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    project_root = Path(cli.project_root).expanduser().resolve()
    config = load_experiment_config(
        cli.config, project_root=project_root
    )
    pycil_root = Path(cli.pycil_root).expanduser()
    if not pycil_root.is_absolute():
        pycil_root = project_root / pycil_root
    run_pycil_experiment(config, pycil_root=pycil_root)


if __name__ == "__main__":
    main()
