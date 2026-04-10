from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_config, save_config
from core.density import fit_diagonal_gaussians
from core.feature_io import load_artifact, save_artifact
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config.yaml")

    train_artifact = load_artifact(experiment_dir / "features_train.pt")
    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    density_cfg = config["density"]
    density = fit_diagonal_gaussians(
        train_artifact["features"].float(),
        train_artifact["targets"].long(),
        hierarchy,
        train_artifact["class_names"],
        eps=density_cfg["eps"],
    )
    density["config"] = density_cfg
    save_artifact(density, experiment_dir / "node_density.pt")


if __name__ == "__main__":
    main()
