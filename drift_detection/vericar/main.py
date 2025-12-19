import torch
import yaml
import os
import glob
import csv
from pathlib import Path
from train import train_config
from test import test_config
from step1_cos import ood_detection_main


def train_test_ood(config):
    log_dir = train_config(config)
    print(f"Training finished. Log directory: {log_dir}")

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = log_dir

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    
    target_ckpts = {}
    for f in ckpt_files:
        fname = os.path.basename(f)
        if "best-loss" in fname:
            target_ckpts["best_loss"] = f
        elif "best-map" in fname:
            target_ckpts["best_map"] = f
        elif "best-prec" in fname:
            target_ckpts["best_prec"] = f

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    seed = config.get("seed", "unknown")
    loss_name = config["training"].get("loss", "unknown")
    csv_filename = os.path.join(results_dir, f"{seed}_{loss_name}.csv")
    
    fieldnames = [
        "ckpt_type", "ckpt_path", 
        "test_map", "test_prec1", 
        "ood_auroc", "ood_fpr95", 
        "combined_map", "combined_prec1", 
        "seen_map", "seen_prec1", 
        "unseen_map", "unseen_prec1"
    ]

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ckpt_type, ckpt_path in target_ckpts.items():
            print(f"\nProcessing {ckpt_type}: {ckpt_path}")
            
            test_res = test_config(config, ckpt_path=ckpt_path)
            
            ood_res = ood_detection_main(config, ckpt_path=ckpt_path)
            
            row = {
                "ckpt_type": ckpt_type,
                "ckpt_path": ckpt_path,
                "test_map": test_res["map_r"],
                "test_prec1": test_res["prec1"],
                "ood_auroc": ood_res["auroc"],
                "ood_fpr95": ood_res["fpr95"],
                "combined_map": ood_res["combined_map"],
                "combined_prec1": ood_res["combined_prec1"],
                "seen_map": ood_res["seen_map"],
                "seen_prec1": ood_res["seen_prec1"],
                "unseen_map": ood_res["unseen_map"],
                "unseen_prec1": ood_res["unseen_prec1"]
            }
            writer.writerow(row)
            print(f"Saved results for {ckpt_type} to {csv_filename}")


if __name__ == '__main__':
    config_dir = "./configs"
    configs = []

    for file in os.listdir(config_dir):
        if file.endswith(".yaml"):
            config_path = os.path.join(config_dir, file)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                configs.append(config)

    for config in configs:
        train_test_ood(config)