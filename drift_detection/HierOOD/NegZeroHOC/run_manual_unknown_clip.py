from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.config import load_config, save_config
from core.eval import evaluate_predictions
from negzero.clip_backend import CLIPBackend
from negzero.diagnostics import summarize_predictions
from negzero.feature_io import feature_paths, load_feature_artifact, validate_feature_artifact
from negzero.inference import predict_features
from negzero.text_cache import build_text_cache
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def _metrics(preds: torch.Tensor, targets: torch.Tensor, hierarchy, extra: dict) -> dict:
    result = evaluate_predictions(preds, targets, hierarchy)
    result.update(extra)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("features", {}).get("mode", "cached") != "cached":
        raise ValueError("run_manual_unknown_clip.py currently expects features.mode: cached")

    id_classes = get_id_classes(config["dataset"]["id_split"])
    hierarchy = Hierarchy(id_classes, config["dataset"]["hierarchy"])
    backend = CLIPBackend.from_config(config["model"])
    include_unknown = bool(config.get("inference", {}).get("include_unknown", True))
    text_cache = build_text_cache(
        hierarchy,
        backend,
        config["prompts"],
        include_unknown=include_unknown,
    )
    first_local = next(iter(text_cache.values()))
    text_dim = int(first_local.text_embeddings.shape[1])

    paths = feature_paths(config)
    id_artifact = load_feature_artifact(paths["id"])
    ood_artifact = load_feature_artifact(paths["ood"])
    require_model_match = bool(config["features"].get("require_model_match", True))
    validate_feature_artifact(
        id_artifact,
        expected_model=config["model"]["name"],
        expected_dim=text_dim,
        require_model_match=require_model_match,
    )
    validate_feature_artifact(
        ood_artifact,
        expected_model=config["model"]["name"],
        expected_dim=text_dim,
        require_model_match=require_model_match,
    )

    temperature = float(config.get("inference", {}).get("temperature", 1.0))
    trace_enabled = bool(config.get("inference", {}).get("trace", False))
    id_preds, id_traces = predict_features(
        id_artifact["features"],
        hierarchy,
        text_cache,
        temperature=temperature,
        return_trace=trace_enabled,
    )
    ood_preds, ood_traces = predict_features(
        ood_artifact["features"],
        hierarchy,
        text_cache,
        temperature=temperature,
        return_trace=trace_enabled,
    )

    id_targets = id_artifact["node_targets"].long()
    ood_targets = ood_artifact["node_targets"].long()
    mixed_preds = torch.cat([id_preds, ood_preds], dim=0)
    mixed_targets = torch.cat([id_targets, ood_targets], dim=0)

    diagnostics = {
        "id": summarize_predictions(id_preds, hierarchy, id_traces),
        "ood": summarize_predictions(ood_preds, hierarchy, ood_traces),
        "include_unknown": include_unknown,
    }
    common_extra = {
        "prediction_mode": config.get("inference", {}).get("prediction_mode", "argmax"),
        "score_type": "clip_cosine_manual_unknown" if include_unknown else "clip_cosine_child_only",
        "temperature": temperature,
        "collapsed_ood": False,
        "negzero": diagnostics,
    }
    result = {
        "experiment_name": config["experiment"]["name"],
        "dataset": config["dataset"]["name"],
        "config": config,
        "results": {
            "id": _metrics(id_preds, id_targets, hierarchy, common_extra),
            "ood": _metrics(ood_preds, ood_targets, hierarchy, common_extra),
            "mixed": _metrics(mixed_preds, mixed_targets, hierarchy, common_extra),
        },
        "predictions": {
            "id": id_preds,
            "ood": ood_preds,
        },
        "targets": {
            "id": id_targets,
            "ood": ood_targets,
        },
        "diagnostics": diagnostics,
    }

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    result_path = output_dir / f"{config['experiment']['name']}.result"
    torch.save(result, result_path)
    print(f"Saved result: {result_path}")
    print(
        "Summary: "
        f"ID BAcc={result['results']['id']['balanced_acc']:.6f}, "
        f"OOD BAcc={result['results']['ood']['balanced_acc']:.6f}, "
        f"Mixed BAcc={result['results']['mixed']['balanced_acc']:.6f}"
    )


if __name__ == "__main__":
    main()

