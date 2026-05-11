from __future__ import annotations

import argparse
from pathlib import Path

import torch

from MS import MS_loss
from data import build_dataloaders
from models import build_model
from utils.gpu import get_device, maybe_data_parallel, unwrap_model
from utils.io import load_config, save_json
from utils.seed import set_seed


def _run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            features = model(x)
            loss = MS_loss(features, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def train_model(config: dict, run_dir: str | Path):
    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(config)

    datasets, loaders = build_dataloaders(config)
    model = build_model(
        config,
        datasets["input_shape"],
        datasets["input_dim"],
        datasets["input_kind"],
    ).to(device)
    model = maybe_data_parallel(model, config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    max_epochs = int(config["training"].get("epochs", 50))
    patience = int(config["training"].get("patience", 10))
    best_val = float("inf")
    bad_epochs = 0
    history = []
    checkpoint_path = run_dir / "checkpoint.pt"

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, loaders["train"], optimizer, device)
        val_loss = _run_epoch(model, loaders["val"], None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_model_state": unwrap_model(model).state_dict(),
                    "config": config,
                    "input_shape": datasets["input_shape"],
                    "input_dim": datasets["input_dim"],
                    "input_kind": datasets["input_kind"],
                    "data_parallel": isinstance(model, torch.nn.DataParallel),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    split = datasets["split_info"]
    save_json(
        run_dir / "metadata.json",
        {
            "id_classes": list(split.id_classes),
            "ood_classes": list(split.ood_classes),
            "best_val_loss": best_val,
            "epochs_ran": len(history),
            "device": str(device),
            "data_parallel": isinstance(model, torch.nn.DataParallel),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    )
    save_json(run_dir / "train_log.json", {"history": history})
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    train_model(load_config(args.config), args.run_dir)


if __name__ == "__main__":
    main()
