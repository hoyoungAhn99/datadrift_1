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
            "Evaluate the native classifier, current-memory NME, and "
            "checkpoint-frozen CMPT-NCM without retraining the CIL learner"
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
    native_name = payload["native_classifier"]["classifier"]
    print(
        f"complete: {settings.learner} | "
        f"Native[{native_name}] AIA={summary['native_aia_percent']:.3f} | "
        f"NME AIA={summary['baseline_aia_percent']:.3f} | "
        f"CMPT AIA={summary['cmpt_aia_percent']:.3f} | "
        f"delta={summary['aia_delta_percent_points']:+.3f} pp"
    )
    interpolation = summary.get("prototype_interpolation")
    if interpolation:
        values = " | ".join(
            f"alpha={entry['alpha']:.2f}: "
            f"AIA={entry['aia_percent']:.3f}, "
            f"final={entry['final_percent']:.3f}"
            for entry in interpolation.values()
        )
        print(f"prototype interpolation sweep | {values}")
    adaptive = summary.get("adaptive_alpha")
    if adaptive:
        values = " | ".join(
            f"{mode}: AIA={entry['aia_percent']:.3f}, "
            f"delta={entry['aia_delta_vs_nme_percent_points']:+.3f}, "
            f"mean-alpha={entry['mean_incremental_alpha']:.3f}"
            for mode, entry in adaptive.items()
        )
        print(f"adaptive alpha | {values}")
    accuracy_oracle = summary.get("accuracy_oracle")
    if accuracy_oracle:
        global_oracle = accuracy_oracle["learner_global"]
        session_oracle = accuracy_oracle["session"]
        print(
            "accuracy oracle (test-label upper bound) | "
            f"global alpha={global_oracle['alpha']:.2f}, "
            f"AIA={global_oracle['aia_percent']:.3f} | "
            f"session AIA={session_oracle['aia_percent']:.3f}, "
            "alphas="
            f"{session_oracle['alphas']}"
        )
    geometric_oracle = summary.get("class_geometric_oracle")
    if geometric_oracle:
        print(
            "class-geometric oracle (full-old-data upper bound) | "
            f"AIA={geometric_oracle['aia_percent']:.3f}, "
            f"delta-vs-CMPT="
            f"{geometric_oracle['aia_delta_vs_pure_cmpt_percent_points']:+.3f}"
        )
    full_mean_oracle = summary.get("full_mean_oracle")
    if full_mean_oracle:
        old_only = full_mean_oracle["old_only"]
        all_seen = full_mean_oracle["all_seen"]
        print(
            "full-training-mean NME oracle | "
            f"old-only AIA={old_only['aia_percent']:.3f}, "
            "delta-vs-NME="
            f"{old_only['aia_delta_vs_nme_percent_points']:+.3f} | "
            f"all-seen AIA={all_seen['aia_percent']:.3f}, "
            "delta-vs-NME="
            f"{all_seen['aia_delta_vs_nme_percent_points']:+.3f}"
        )
    components = summary.get("component_ablation")
    if components:
        class_translation = components["class_translation"]
        combined = components["combined_cmpt"]
        print(
            "CMPT component ablation | "
            f"Class-T AIA={class_translation['aia_percent']:.3f}, "
            f"final={class_translation['final_percent']:.3f} | "
            f"Combined AIA={combined['aia_percent']:.3f}, "
            f"final={combined['final_percent']:.3f}, "
            "delta-vs-Global="
            f"{combined['aia_delta_vs_global_percent_points']:+.3f}, "
            "delta-vs-Class="
            f"{combined['aia_delta_vs_class_percent_points']:+.3f}"
        )
    neighbor = summary.get("neighbor_affine")
    if neighbor:
        print(
            "local-neighbor affine | "
            f"AIA={neighbor['aia_percent']:.3f}, "
            f"final={neighbor['final_percent']:.3f}, "
            "delta-vs-NME="
            f"{neighbor['aia_delta_vs_nme_percent_points']:+.3f}, "
            "delta-vs-Global="
            f"{neighbor['aia_delta_vs_global_percent_points']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
