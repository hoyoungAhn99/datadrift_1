from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from compare_results import load_result_rows
from core.config import load_config
from core.feature_io import load_artifact
from feature_generation.utils.io import resolve_feature_tensor
from hierarchical_density_inference import evaluate_split
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


SCALAR_METRICS = ["acc", "balanced_acc", "avg_hdist", "balanced_hdist"]
HIGHER_IS_BETTER = {"acc", "balanced_acc"}
LOWER_IS_BETTER = {"avg_hdist", "balanced_hdist"}


def parse_list_arg(raw: str | None, default_values: list[float]) -> list[float]:
    if raw is None:
        return default_values
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def default_log_grid(start: float, end: float, num: int) -> list[float]:
    if num <= 1:
        return [float(start)]
    values = torch.logspace(
        start=torch.log10(torch.tensor(float(start))),
        end=torch.log10(torch.tensor(float(end))),
        steps=num,
    )
    return [float(v.item()) for v in values]


def metric_sort(metric: str):
    if metric in HIGHER_IS_BETTER:
        return False
    if metric in LOWER_IS_BETTER:
        return True
    raise ValueError(f"Unsupported metric: {metric}")


def build_inference_cfg(base_cfg: dict[str, Any], temperature: float, alpha: float, beta: float) -> dict[str, Any]:
    inference_cfg = dict(base_cfg)
    inference_cfg["temperature"] = float(temperature)
    inference_cfg["alpha"] = float(alpha)
    inference_cfg["beta"] = float(beta)
    return inference_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prohoc", default=None, help="Optional ProHOC .result file or directory for comparison rows.")
    parser.add_argument("--target_split", default="ood", choices=["val", "ood"], help="Split used to rank top-5 hyperparameters.")
    parser.add_argument("--taus", default=None, help="Comma-separated temperatures. Default: logspace in [1, 10].")
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha values. Default: logspace in [0.1, 10].")
    parser.add_argument("--betas", default=None, help="Comma-separated beta values. Default: logspace in [0.1, 50].")
    parser.add_argument("--num_taus", type=int, default=5)
    parser.add_argument("--num_alphas", type=int, default=5)
    parser.add_argument("--num_betas", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "hyperparam_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    taus = parse_list_arg(args.taus, default_log_grid(1.0, 10.0, args.num_taus))
    alphas = parse_list_arg(args.alphas, default_log_grid(0.1, 10.0, args.num_alphas))
    betas = parse_list_arg(args.betas, default_log_grid(0.1, 50.0, args.num_betas))

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    density_payload = load_artifact(experiment_dir / "node_density.pt")
    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, val_feature_meta = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, ood_feature_meta = resolve_feature_tensor(config, experiment_dir, "ood")

    rows: list[dict[str, Any]] = []

    combinations = list(itertools.product(taus, alphas, betas))
    progress = tqdm(combinations, desc="Hyperparameter tuning", total=len(combinations))

    for temperature, alpha, beta in progress:
        progress.set_postfix(
            tau=f"{temperature:.4g}",
            alpha=f"{alpha:.4g}",
            beta=f"{beta:.4g}",
        )
        inference_cfg = build_inference_cfg(config["inference"], temperature, alpha, beta)
        val_metrics = evaluate_split(
            val_artifact,
            val_features,
            hierarchy,
            density_payload,
            inference_cfg,
            val_feature_meta,
        )
        ood_metrics = evaluate_split(
            ood_artifact,
            ood_features,
            hierarchy,
            density_payload,
            inference_cfg,
            ood_feature_meta,
        )

        for split_name, metrics in [("val", val_metrics), ("ood", ood_metrics)]:
            row = {
                "experiment_name": config["experiment"]["name"],
                "dataset": dataset_cfg.get("name", Path(dataset_cfg["id_split"]).stem),
                "split": split_name,
                "score_type": inference_cfg.get("score_type"),
                "covariance_type": density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type")),
                "temperature": float(temperature),
                "alpha": float(alpha),
                "beta": float(beta),
                "kappa": float(inference_cfg.get("kappa", 20.0)),
            }
            for metric in SCALAR_METRICS:
                value = metrics.get(metric)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                row[metric] = value
            rows.append(row)

    tuning_df = pd.DataFrame(rows)
    tuning_csv = output_dir / "tuning_results.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    top_rows: list[pd.DataFrame] = []
    split_df = tuning_df[tuning_df["split"] == args.target_split].copy()
    for metric in SCALAR_METRICS:
        ascending = metric_sort(metric)
        ranked = split_df.sort_values(
            by=[metric, "temperature", "alpha", "beta"],
            ascending=[ascending, True, True, True],
        ).head(args.topk).copy()
        ranked.insert(0, "rank_metric", metric)
        top_rows.append(ranked)

    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    top_csv = output_dir / f"top_{args.topk}_{args.target_split}.csv"
    top_df.to_csv(top_csv, index=False)

    summary_rows = []
    for metric in SCALAR_METRICS:
        metric_top = top_df[top_df["rank_metric"] == metric]
        for _, row in metric_top.iterrows():
            summary_rows.append(
                {
                    "metric": metric,
                    "split": args.target_split,
                    "temperature": row["temperature"],
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                    "score_type": row["score_type"],
                    "covariance_type": row["covariance_type"],
                    metric: row[metric],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"best_hparams_summary_{args.target_split}.csv"
    summary_df.to_csv(summary_csv, index=False)

    if args.prohoc:
        prohoc_rows = load_result_rows([Path(p) for p in _discover_paths(args.prohoc)], source_hint="prohoc")
        gm_compare_rows = []
        for _, row in split_df.iterrows():
            gm_compare_rows.append(
                {
                    "source": "gm_prohoc",
                    "result_file": str(experiment_dir / "hinference_density.result"),
                    "dataset": row["dataset"],
                    "method": (
                        f"{config['experiment']['name']}"
                        f"_tau{row['temperature']:.6g}"
                        f"_a{row['alpha']:.6g}"
                        f"_b{row['beta']:.6g}"
                    ),
                    "split": row["split"],
                    "acc": row["acc"],
                    "balanced_acc": row["balanced_acc"],
                    "avg_hdist": row["avg_hdist"],
                    "balanced_hdist": row["balanced_hdist"],
                }
            )
        compare_df = pd.DataFrame(prohoc_rows + gm_compare_rows)
        compare_df.to_csv(output_dir / "comparison_with_prohoc.csv", index=False)

    print(f"Saved tuning results to: {tuning_csv}")
    print(f"Saved top-{args.topk} hyperparameters to: {top_csv}")
    print(f"Saved summary to: {summary_csv}")


def _discover_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.result"))


if __name__ == "__main__":
    main()
