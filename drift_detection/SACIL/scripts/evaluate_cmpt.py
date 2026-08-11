from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.cmpt import (  # noqa: E402
    CMPTCheckpointEvaluator,
    CMPTExperimentSettings,
)
from sacil.config import load_config_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate current-memory NME and checkpoint-frozen CMPT-NCM "
            "without retraining the CIL learner"
        )
    )
    parser.add_argument("config", type=Path, help="CMPT YAML config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="evaluate only the first N sessions for a smoke test",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="audit checkpoint compatibility without loading data/models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing result file",
    )
    return parser.parse_args()


def _smoke_output(path: Path, sessions: int) -> Path:
    return path.with_name(
        f"{path.stem}_smoke_s0_s{int(sessions) - 1}{path.suffix}"
    )


def main() -> int:
    args = parse_args()
    config = load_config_tree(args.config)
    if args.device is not None:
        config["device"] = args.device
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    evaluator = CMPTCheckpointEvaluator(
        settings,
        PROJECT_ROOT,
        source_root=SOURCE_ROOT,
        max_sessions=args.max_sessions,
        progress=print,
    )
    if args.validate_only:
        print(
            json.dumps(
                evaluator.validation_payload(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output = None if args.output is None else args.output.resolve()
    if output is None and args.max_sessions is not None:
        output = _smoke_output(settings.output_file, args.max_sessions)
    payload = evaluator.run(output_file=output, force=args.force)
    summary = payload["summary"]
    print(
        f"complete: {settings.learner} | "
        f"NME AIA={summary['baseline_aia_percent']:.3f} | "
        f"CMPT AIA={summary['cmpt_aia_percent']:.3f} | "
        f"delta={summary['aia_delta_percent_points']:+.3f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
