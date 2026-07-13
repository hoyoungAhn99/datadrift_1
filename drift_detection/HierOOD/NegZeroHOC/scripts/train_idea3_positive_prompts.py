from __future__ import annotations

import argparse
from argparse import Namespace
import random
import sys
from pathlib import Path

import torch
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import save_idea3_checkpoint
from negzerohoc.clip_backend import ClipBackend
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.losses import positive_ce_loss, prompt_metric_loss
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.soft_prompting import SoftPromptTextEncoder, soft_prompt_smoke_test
from negzerohoc.training_data import build_positive_edge_examples, gather_image_features, group_examples_by_parent, node_path, sample_examples


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    features_cfg = cfg.get("features", {})
    prompt_cfg = cfg.get("prompt", {})
    train_cfg = cfg.get("positive_training", {})
    loss_cfg = train_cfg.get("loss", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea3")
    output_root = experiment_cfg.get("output_root", "outputs")
    checkpoint = train_cfg.get("checkpoint") or str(Path(output_root) / "checkpoints" / f"{experiment_name}-positive.pt")
    result_path = train_cfg.get("result_path") or str(Path(output_root) / "results" / f"{experiment_name}-positive_child_only.result")
    diagnostics_path = train_cfg.get("diagnostics_path") or str(Path(output_root) / "diagnostics" / f"{experiment_name}-positive-diagnostics.json")

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=output_root,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        features_dir=features_cfg.get("dir") or inference_cfg.get("features_dir"),
        device=runtime_cfg.get("device", "cuda"),
        seed=int(runtime_cfg.get("seed", 0)),
        prompt=prompt_cfg,
        epochs=int(train_cfg.get("epochs", 20)),
        batch_size=int(train_cfg.get("batch_size", 1024)),
        parents_per_step=int(train_cfg.get("parents_per_step", 4)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        loss_name=loss_cfg.get("name", "ce"),
        loss_alpha=float(loss_cfg.get("alpha", 2.0)),
        loss_beta=float(loss_cfg.get("beta", 50.0)),
        loss_lam=float(loss_cfg.get("lam", 0.5)),
        loss_mining_margin=float(loss_cfg.get("mining_margin", 0.1)),
        loss_minimum_mode=loss_cfg.get("minimum_mode", "sample"),
        loss_dist_scale=float(loss_cfg.get("dist_scale", 2.0)),
        loss_dist_pow=float(loss_cfg.get("dist_pow", 1.0)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        metric_weight=float(loss_cfg.get("metric_weight", 1.0)),
        eval_after_training=bool(train_cfg.get("eval_after_training", True)),
        inference_batch_size=int(inference_cfg.get("batch_size", 1024)),
        inference_tau=float(inference_cfg.get("tau", 1.0)),
        checkpoint=checkpoint,
        result_path=result_path,
        diagnostics_path=diagnostics_path,
    )


def _path_label_tensor(hierarchy, nodes: list[str], max_len: int, device: str | torch.device) -> torch.Tensor:
    node_to_idx = {node: idx for idx, node in enumerate(hierarchy.id_node_list)}
    rows = []
    for node in nodes:
        path = node_path(hierarchy, node)
        if len(path) < max_len:
            path = path + [path[-1]] * (max_len - len(path))
        elif len(path) > max_len:
            path = path[:max_len]
        rows.append([node_to_idx[item] for item in path])
    return torch.tensor(rows, dtype=torch.long, device=device)


def compute_positive_loss(
    args,
    hierarchy,
    learner,
    image_features,
    child_features,
    parent_examples,
    children,
    targets,
    device,
):
    loss_name = args.loss_name.lower()
    logits = image_features @ child_features.t() / float(args.tau)
    local_acc = (logits.argmax(dim=1) == targets.to(logits.device)).float().mean()

    if loss_name == "ce":
        loss, stats = positive_ce_loss(image_features, child_features, targets, tau=args.tau)
        stats["metric_loss"] = 0.0
        return loss, stats

    metric_names = {"ms", "hims_min", "himsmin", "weihims", "hims_min_wei"}
    combo_names = {"ce_ms", "ce_hims_min", "ce_himsmin", "ce_weihims"}
    if loss_name not in metric_names and loss_name not in combo_names:
        raise ValueError(f"Unsupported positive_training.loss.name: {args.loss_name}")

    metric_name = loss_name
    use_ce = False
    if loss_name.startswith("ce_"):
        use_ce = True
        metric_name = loss_name[3:]

    max_len = hierarchy.max_depth + 1
    image_nodes = [example.leaf for example in parent_examples]
    image_path_labels = _path_label_tensor(hierarchy, image_nodes, max_len, device)
    prompt_path_labels = _path_label_tensor(hierarchy, children, max_len, device)
    metric_loss, metric_stats = prompt_metric_loss(
        image_features,
        child_features,
        image_path_labels,
        prompt_path_labels,
        loss_name=metric_name,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        lam=args.loss_lam,
        mining_margin=args.loss_mining_margin,
        minimum_mode=args.loss_minimum_mode,
        dist_scale=args.loss_dist_scale,
        dist_pow=args.loss_dist_pow,
    )

    if use_ce:
        ce_loss, _ = positive_ce_loss(image_features, child_features, targets, tau=args.tau)
        loss = args.ce_weight * ce_loss + args.metric_weight * metric_loss
        ce_value = float(ce_loss.detach().cpu())
    else:
        loss = metric_loss
        ce_value = 0.0

    return loss, {
        "acc": float(local_acc.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "ce_loss": ce_value,
        "metric_loss": metric_stats["metric_loss"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


@torch.no_grad()
def evaluate_positive(args, hierarchy, learner, features_dir: Path, device: str) -> dict:
    dists_mats = make_distance_mats(hierarchy)
    semantic_index = build_idea3_semantic_index(hierarchy, learner, mode="positive_child_only")
    results = {
        "args": vars(args),
        "mode": "positive_child_only",
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
    }
    for split_name in ["val", "ood"]:
        payload = load_feature_file(features_dir / f"{split_name}-features.pt")
        preds = []
        stop_depth_counts = {}
        stop_node_counts = {}
        features = payload["features"]
        for start in range(0, int(features.shape[0]), args.inference_batch_size):
            end = min(start + args.inference_batch_size, int(features.shape[0]))
            out = predict_features_idea3(
                features[start:end].to(device),
                hierarchy,
                semantic_index,
                mode="positive_child_only",
                tau=args.inference_tau,
            )
            preds.append(out["preds"].cpu())
            for key, value in out["diagnostics"]["stop_depth_counts"].items():
                stop_depth_counts[int(key)] = stop_depth_counts.get(int(key), 0) + int(value)
            for key, value in out["diagnostics"]["stop_node_counts"].items():
                stop_node_counts[str(key)] = stop_node_counts.get(str(key), 0) + int(value)
        preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
        node_labels, metrics = evaluate_split(hierarchy, payload, preds.cpu(), dists_mats=dists_mats)
        results[split_name] = {
            "preds": preds.cpu(),
            "targets": node_labels.cpu(),
            "metrics": metrics,
            "diagnostics": {
                "stop_depth_counts": dict(sorted(stop_depth_counts.items())),
                "stop_node_counts": dict(sorted(stop_node_counts.items(), key=lambda x: x[1], reverse=True)),
            },
        }
    results["mixed"] = mixed_summary(results["val"]["metrics"], results["ood"]["metrics"])
    return results


def main():
    args = parse_args()
    if not args.features_dir:
        raise ValueError("Missing features.dir or inference.features_dir in config")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    features_dir = Path(args.features_dir)
    train_payload = load_feature_file(features_dir / "train-features.pt")
    train_features = train_payload["features"].float()
    examples = build_positive_edge_examples(hierarchy, train_payload)
    grouped = group_examples_by_parent(examples)
    parents = sorted(grouped)
    if not parents:
        raise RuntimeError("No positive edge examples were built from train-features.pt")

    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    smoke = soft_prompt_smoke_test(backend, max_length=int(args.prompt.get("max_length", 77)))
    if not smoke["ok"]:
        raise RuntimeError(f"Soft prompt smoke test failed: {smoke}")

    text_encoder = SoftPromptTextEncoder(
        backend.model,
        backend.processor.tokenizer,
        max_length=int(args.prompt.get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(args.prompt)
    learner = PositivePromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    n_per_parent = max(1, args.batch_size // max(1, args.parents_per_step))
    history = []

    for epoch in range(1, args.epochs + 1):
        learner.train()
        shuffled = list(parents)
        rng.shuffle(shuffled)
        chunks = [
            shuffled[i:i + args.parents_per_step]
            for i in range(0, len(shuffled), args.parents_per_step)
        ]
        iterator = tqdm(chunks, desc=f"positive epoch {epoch}/{args.epochs}", leave=False) if tqdm else chunks
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        for parent_chunk in iterator:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            step_accs = []
            for parent in parent_chunk:
                parent_examples = sample_examples(grouped[parent], n_per_parent, rng)
                if not parent_examples:
                    continue
                children = list(hierarchy.parent2children[parent])
                child_to_idx = {child: idx for idx, child in enumerate(children)}
                targets = torch.tensor([child_to_idx[ex.child] for ex in parent_examples], dtype=torch.long, device=device)
                image_features = gather_image_features(train_features, parent_examples, device)
                child_features = learner.encode_children(parent, children)
                loss, stats = compute_positive_loss(
                    args,
                    hierarchy,
                    learner,
                    image_features,
                    child_features,
                    parent_examples,
                    children,
                    targets,
                    device,
                )
                losses.append(loss)
                step_accs.append(stats["acc"])

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_acc += sum(step_accs) / max(1, len(step_accs))
            steps += 1

        epoch_stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "local_acc": epoch_acc / max(1, steps),
            "steps": steps,
        }
        history.append(epoch_stats)
        print(f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, local_acc={epoch_stats['local_acc']:.6f}")

    metrics = {"train_history": history, "soft_prompt_smoke": smoke}
    result = None
    if args.eval_after_training:
        result = evaluate_positive(args, hierarchy, learner, features_dir, device)
        ensure_dir(Path(args.result_path).parent)
        torch.save(result, args.result_path)
        metrics["final"] = {
            "val_balanced_acc": float(result["val"]["metrics"]["balanced_acc"]),
            "ood_balanced_acc": float(result["ood"]["metrics"]["balanced_acc"]),
            "mixed_balanced_acc": float(result["mixed"]["mixed_balanced_acc"]),
            "val_balanced_hdist": float(result["val"]["metrics"]["balanced_hdist"]),
            "ood_balanced_hdist": float(result["ood"]["metrics"]["balanced_hdist"]),
            "mixed_balanced_hdist": float(result["mixed"]["mixed_balanced_hdist"]),
        }
        print(f"saved result: {args.result_path}")

    ckpt_path = save_idea3_checkpoint(
        args.checkpoint,
        stage="positive",
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=learner.state_dict(),
        metrics=metrics,
        args=vars(args),
    )
    save_json(args.diagnostics_path, metrics)
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved diagnostics: {args.diagnostics_path}")


if __name__ == "__main__":
    main()
