from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank, compute_prototypes
from sacil.config import get_required
from sacil.data import ClassOrderProtocol, build_data_module
from sacil.engine.checkpoint import save_checkpoint
from sacil.engine.evaluator import (
    compute_nme_class_means,
    evaluate,
    evaluate_nme,
)
from sacil.features import collect_features
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.memory import ExemplarMemory, herding_select, icarl_herding_select
from sacil.methods import (
    AnchorGeometryLoss,
    afc_nca_loss,
    afc_pod_loss,
    casper_spectral_loss,
    compute_conflict_weights,
    controlled_transfer_loss,
    create_classification_loss,
    create_contrastive_loss,
    cross_space_clustering_loss,
    fgp_graph_preservation_loss,
    icarl_bce_loss,
    old_logit_kl_loss,
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
    prototype_cross_entropy,
    reconstruction_confidence_weights,
    scheduled_afc_factor,
    scheduled_fgp_weight,
)
from sacil.metrics import CILMetricsTracker
from sacil.models import (
    AFCIncrementalNet,
    CREATEIncrementalNet,
    CSCCTIncrementalNet,
    ExpandableLinearNet,
    FGPIncrementalNet,
    kmeans_imprinted_weights,
)
from sacil.utils import (
    dump_json,
    ensure_dir,
    git_commit,
    make_generator,
    resolved_device,
    seed_worker,
    set_seed,
)


SUPPORTED_TABLE1_METHODS = frozenset(
    {
        "joint",
        "finetune",
        "replay",
        "icarl",
        "podnet",
        "afc",
        "create",
        "fgp",
        "cscct",
        "casper",
        "sacil",
    }
)


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Draw a fixed number of classes before drawing replay samples.

    This is the sampling contract used by CaSpeR.  It intentionally differs
    from ordinary class-balanced weighting: every graph contains exactly
    ``classes_per_batch`` old classes.
    """

    def __init__(
        self,
        labels: list[int],
        *,
        batch_size: int,
        classes_per_batch: int,
        batches: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.classes_per_batch = int(classes_per_batch)
        self.batches = int(batches)
        self.seed = int(seed)
        positions: dict[int, list[int]] = defaultdict(list)
        for position, label in enumerate(labels):
            positions[int(label)].append(position)
        self.positions = dict(positions)
        if self.classes_per_batch > len(self.positions):
            raise ValueError("CaSpeR requests more classes than memory contains")
        if self.classes_per_batch <= 0 or self.batch_size <= self.classes_per_batch:
            raise ValueError("invalid CaSpeR balanced batch dimensions")
        self._iteration = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        classes = np.asarray(sorted(self.positions), dtype=np.int64)
        base, remainder = divmod(self.batch_size, self.classes_per_batch)
        for _ in range(self.batches):
            selected = rng.choice(
                classes, size=self.classes_per_batch, replace=False
            )
            batch: list[int] = []
            for offset, class_id in enumerate(selected.tolist()):
                count = base + int(offset < remainder)
                choices = rng.choice(
                    self.positions[int(class_id)], size=count, replace=True
                )
                batch.extend(int(value) for value in choices.tolist())
            rng.shuffle(batch)
            yield batch


class StandaloneTable1Trainer:
    """PyCIL-free Table-1 trainer with method-specific model lifecycles."""

    def __init__(
        self,
        config: dict[str, Any],
        project_root: str | Path,
        *,
        max_sessions: int | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.project_root = Path(project_root).resolve()
        self.method = str(get_required(config, "method.name")).lower()
        if self.method not in SUPPORTED_TABLE1_METHODS:
            raise ValueError(f"unsupported standalone method: {self.method}")
        self.seed = int(config.get("seed", 1))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = resolved_device(str(config.get("device", "cuda:0")))
        self.protocol = ClassOrderProtocol.from_json(
            self._project_path(get_required(config, "data.protocol"))
        )
        self.data = build_data_module(
            str(config["data"].get("name", "cifar100")),
            self._project_path(get_required(config, "data.root")),
            self.protocol,
            download=bool(config["data"].get("download", False)),
            color_jitter=bool(config["data"].get("color_jitter", True)),
        )
        initial_limit = self._memory_limit(self.protocol.session(0).stop)
        self.memory = ExemplarMemory(initial_limit)
        self.metrics = CILMetricsTracker()
        self.model: nn.Module | None = None
        self.class_means: Tensor | None = None
        self.sacil_tree: HierarchyTree | None = None
        self.sacil_prototypes: PrototypeBank | None = None
        self.sacil_anchors: HierarchicalAnchorBank | None = None
        self.max_sessions = (
            self.protocol.num_sessions
            if max_sessions is None
            else min(int(max_sessions), self.protocol.num_sessions)
        )
        output = config["output"]
        self.run_dir = ensure_dir(
            self._project_path(output["directory"])
            / str(output["run_name"])
            / f"seed_{self.seed}"
        )
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        dump_json(
            {
                "config": self.config,
                "framework": "sacil-standalone",
                "pycil_used": False,
                "protocol_id": self.protocol.protocol_id,
                "device": str(self.device),
                "git_commit": git_commit(self.project_root),
            },
            self.run_dir / "resolved_config.json",
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _memory_limit(self, seen_classes: int) -> int:
        memory = self.config["memory"]
        mode = str(memory.get("mode", "per_class"))
        if mode == "per_class":
            return int(memory.get("exemplars_per_class", 20))
        if mode == "fixed_total":
            return max(1, int(memory.get("capacity", 2000)) // int(seen_classes))
        raise ValueError(f"unknown memory mode: {mode}")

    @property
    def debug_train_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("train_samples_per_class")
        return None if value is None else int(value)

    @property
    def debug_test_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("test_samples_per_class")
        return None if value is None else int(value)

    def _loader(
        self,
        dataset,
        *,
        shuffle: bool,
        session_id: int,
        batch_size: int | None = None,
        batch_sampler=None,
    ) -> DataLoader:
        training = self.config["training"]
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "num_workers": int(training.get("num_workers", 0)),
            "pin_memory": bool(training.get("pin_memory", True)),
            "worker_init_fn": seed_worker,
            "generator": make_generator(self.seed * 1000 + session_id),
        }
        workers = kwargs["num_workers"]
        if workers > 0:
            kwargs["persistent_workers"] = bool(
                training.get("persistent_workers", False)
            )
        if batch_sampler is not None:
            kwargs["batch_sampler"] = batch_sampler
        else:
            kwargs.update(
                batch_size=int(batch_size or training.get("batch_size", 128)),
                shuffle=bool(shuffle),
            )
        return DataLoader(**kwargs)

    def _new_model(self, num_classes: int) -> nn.Module:
        model = self.config.get("model", {})
        if self.method in {"podnet", "afc"}:
            return AFCIncrementalNet(
                num_classes,
                initial_size=self.protocol.session(0).size,
                increment_size=self.protocol.session(1).size,
                proxies_per_class=int(model.get("proxies_per_class", 10)),
                classifier_scale=float(model.get("nca_scale", 1.0)),
                distance_scale=float(model.get("distance_scale", 3.0)),
            )
        if self.method == "create":
            return CREATEIncrementalNet(
                num_classes,
                hidden_layers=tuple(model.get("hidden_layers", [])),
                latent_features=int(model.get("latent_features", 32)),
                reconstruction_scale=float(model.get("reconstruction_scale", 0.1)),
            )
        if self.method == "fgp":
            return FGPIncrementalNet(num_classes)
        if self.method == "cscct":
            return CSCCTIncrementalNet(num_classes)
        return ExpandableLinearNet(
            num_classes, backbone=str(model.get("backbone", "resnet32"))
        )

    def _phase(self, session_id: int) -> dict[str, Any]:
        name = "base" if session_id == 0 else "incremental"
        return dict(self.config["training"][name])

    @staticmethod
    def _scheduler(optimizer: SGD, phase: dict[str, Any], epochs: int):
        kind = str(phase.get("scheduler", "multistep"))
        if kind == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        if kind == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=[int(v) for v in phase.get("milestones", [])],
                gamma=float(phase.get("lr_decay", 0.1)),
            )
        raise ValueError(f"unknown scheduler: {kind}")

    def _training_dataset(self, session_id: int, old_indices: list[int]):
        if self.method == "joint":
            return self.data.train_dataset_for_classes(
                self.protocol.seen_classes(session_id),
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        if self.method == "finetune":
            return self.data.new_train_dataset(
                session_id,
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        return self.data.training_dataset(
            session_id,
            old_indices,
            samples_per_class=self.debug_train_samples_per_class,
        )

    def _prototype_dataset(self, session_id: int, old_indices: list[int]):
        new = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        if not old_indices:
            return new
        return ConcatDataset(
            [new, self.data.replay_dataset(old_indices, augment=False)]
        )

    @torch.inference_mode()
    def _incoming_collection(self, session_id: int):
        if self.model is None:
            raise RuntimeError("model is missing")
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        return collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 4000),
            self.device,
        )

    def _expand_model(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("cannot expand a missing model")
        total = self.protocol.session(session_id).stop
        if isinstance(self.model, AFCIncrementalNet):
            collection = self._incoming_collection(session_id)
            class_features = [
                collection.features[
                    collection.original_targets == int(class_id)
                ]
                for class_id in self.protocol.classes_for_session(session_id)
            ]
            weights = kmeans_imprinted_weights(
                class_features,
                self.model.classifier.weights,
                proxies_per_class=self.model.classifier.proxies_per_class,
                random_state=self.seed * 1000 + session_id * 100,
            )
            self.model.expand_classes(total, weights.to(self.device))
        elif isinstance(self.model, CSCCTIncrementalNet):
            collection = self._incoming_collection(session_id)
            old_norm = self.model.classifier.weight.detach().norm(dim=1).mean()
            values = []
            for class_id in self.protocol.classes_for_session(session_id):
                feature = collection.features[
                    collection.original_targets == int(class_id)
                ].mean(dim=0, keepdim=True)
                values.append(F.normalize(feature, dim=1) * old_norm.cpu())
            self.model.expand_classes(torch.cat(values).to(self.device))
        elif hasattr(self.model, "expand_classes"):
            self.model.expand_classes(total)
        else:
            raise TypeError("model has no expansion contract")

    def run(self) -> dict[str, Any]:
        session_log = self.run_dir / "sessions.jsonl"
        metrics_path = self.run_dir / "metrics.json"
        existing_checkpoints = tuple(self.checkpoint_dir.glob("session_*.pt"))
        if session_log.exists() or metrics_path.exists() or existing_checkpoints:
            raise FileExistsError(
                "the standalone run directory already contains training "
                f"artifacts: {self.run_dir}; choose a new --run-name"
            )
        for session_id in range(self.max_sessions):
            started = time.time()
            record = self._run_session(session_id)
            record["elapsed_seconds"] = time.time() - started
            with session_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = self.metrics.summary()
        dump_json(
            {"summary": summary, "sessions": self.metrics.records},
            metrics_path,
        )
        return summary

    def _run_session(self, session_id: int) -> dict[str, Any]:
        session = self.protocol.session(session_id)
        old_indices = self.memory.all_indices(self.protocol.class_order)
        if session_id == 0 or self.method == "joint":
            self.model = self._new_model(session.stop).to(self.device)
            teacher = None
        else:
            if self.model is None:
                raise RuntimeError("incremental session has no model")
            teacher = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._expand_model(session_id)
            self.model.to(self.device)

        geometry = self._prepare_sacil_geometry(session_id, teacher)
        train_dataset = self._training_dataset(session_id, old_indices)
        train_loader = self._loader(
            train_dataset, shuffle=True, session_id=session_id
        )
        prototype_loader = (
            self._loader(
                self._prototype_dataset(session_id, old_indices),
                shuffle=False,
                session_id=session_id + 5000,
            )
            if self.method == "sacil"
            else None
        )
        training = self._train_session(
            session_id,
            train_loader,
            teacher,
            geometry,
            prototype_loader,
        )
        self._update_memory(session_id)
        post = self._post_training(session_id, train_loader)
        self._build_evaluation_state(session_id)
        evaluation = self._evaluate_session(session_id)
        metric = self.metrics.update(session_id, evaluation)
        checkpoint = self._save_checkpoint(session_id)
        return {
            "session_id": session_id,
            "method": self.method,
            "seen_class_count": session.stop,
            "memory_size": len(self.memory),
            "training": training,
            "post_training": post,
            "evaluation": metric,
            "checkpoint": str(checkpoint),
        }

    def _prepare_sacil_geometry(
        self, session_id: int, teacher: nn.Module | None
    ) -> AnchorGeometryLoss | None:
        if self.method != "sacil" or session_id == 0:
            return None
        if (
            teacher is None
            or self.sacil_anchors is None
            or self.sacil_tree is None
        ):
            raise RuntimeError("SACIL incremental session lacks old anchors")
        collection = self._incoming_collection(session_id)
        incoming = compute_prototypes(
            collection.features,
            collection.original_targets,
            self.protocol.classes_for_session(session_id),
        )
        conflict = self.config["method"].get("conflict", {})
        weights = compute_conflict_weights(
            incoming,
            self.sacil_anchors,
            self.sacil_tree,
            max_neighbors=int(conflict.get("max_neighbors", 5)),
            old_class_ratio=float(conflict.get("old_class_ratio", 0.1)),
            temperature=float(conflict.get("temperature", 0.05)),
            min_preservation_weight=float(
                conflict.get("min_preservation_weight", 0.1)
            ),
            ancestor_decay=float(conflict.get("ancestor_decay", 0.5)),
        )
        return AnchorGeometryLoss(
            self.sacil_anchors,
            weights.leaf_weights,
            weights.internal_weights,
            use_internal_anchors=True,
        ).to(self.device)

    def _train_session(
        self,
        session_id: int,
        loader: DataLoader,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototype_loader: DataLoader | None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("training has no model")
        phase = self._phase(session_id)
        parameters = self._main_parameters()
        optimizer = SGD(
            parameters,
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 5e-4)),
            nesterov=bool(phase.get("nesterov", False)),
        )
        epochs = int(phase["epochs"])
        scheduler = self._scheduler(optimizer, phase, epochs)
        fusion_loader = self._cscct_balanced_loader(session_id)
        fusion_optimizer = (
            SGD(
                list(self.model.fusion.parameters()),
                lr=float(self.config["method"].get("fusion_lr", 1e-8)),
                momentum=float(phase.get("momentum", 0.9)),
                weight_decay=float(phase.get("weight_decay", 5e-4)),
            )
            if isinstance(self.model, CSCCTIncrementalNet)
            and fusion_loader is not None
            else None
        )
        fusion_scheduler = (
            self._scheduler(fusion_optimizer, phase, epochs)
            if fusion_optimizer is not None
            else None
        )
        casper_loader = self._casper_loader(session_id, max(1, len(loader)))
        max_batches_value = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = (
            None if max_batches_value is None else int(max_batches_value)
        )
        logs = []
        progress = tqdm(
            range(epochs), disable=bool(self.config.get("disable_tqdm", False))
        )
        for epoch in progress:
            prototypes = (
                self._refresh_training_prototypes(prototype_loader)
                if prototype_loader is not None
                else None
            )
            self.model.train()
            totals: dict[str, float] = defaultdict(float)
            count = batches = correct = 0
            casper_iterator = None if casper_loader is None else iter(casper_loader)
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True).long()
                replay_mask = batch["is_replay"].to(self.device).bool()
                replay_images = None
                if casper_iterator is not None:
                    try:
                        replay_batch = next(casper_iterator)
                    except StopIteration:
                        casper_iterator = iter(casper_loader)
                        replay_batch = next(casper_iterator)
                    replay_images = replay_batch["image"].to(
                        self.device, non_blocking=True
                    )
                components, prediction = self._loss_components(
                    session_id,
                    images,
                    targets,
                    replay_mask,
                    teacher,
                    geometry,
                    prototypes,
                    replay_images,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_count = targets.numel()
                count += batch_count
                batches += 1
                correct += int(prediction.argmax(1).eq(targets).sum().item())
                totals["loss"] += float(loss.detach()) * batch_count
                for name, value in components.items():
                    totals[name] += float(value.detach()) * batch_count
            scheduler.step()
            if fusion_optimizer is not None and fusion_loader is not None:
                self._update_cscct_fusion(fusion_loader, fusion_optimizer)
                if fusion_scheduler is not None:
                    fusion_scheduler.step()
            if count == 0:
                raise RuntimeError("standalone training processed no samples")
            record = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batches": batches,
                "train_accuracy": correct / count,
                **{key: value / count for key, value in totals.items()},
            }
            logs.append(record)
            progress.set_description(
                f"{self.method} s{session_id} e{epoch + 1}/{epochs} "
                f"loss={record['loss']:.4f} acc={record['train_accuracy']:.3f}"
            )
        return {"epochs": epochs, "samples": len(loader.dataset), "epoch_logs": logs}

    def _main_parameters(self) -> list[nn.Parameter]:
        if self.model is None:
            raise RuntimeError("model is missing")
        if isinstance(self.model, AFCIncrementalNet):
            self.model.postprocessor_scale.requires_grad_(False)
            if self.method == "afc":
                for parameter in self.model.classifier.old_weights:
                    parameter.requires_grad_(False)
                return [
                    *[p for p in self.model.backbone.parameters() if p.requires_grad],
                    self.model.classifier.new_weights,
                ]
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad_(True)
            return [
                *[p for p in self.model.backbone.parameters() if p.requires_grad],
                *list(self.model.classifier.parameters()),
            ]
        if isinstance(self.model, CSCCTIncrementalNet):
            for parameter in self.model.classifier.old_weights:
                parameter.requires_grad_(False)
            return self.model.main_parameters()
        if self.method == "sacil" and isinstance(
            self.model, ExpandableLinearNet
        ):
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad_(False)
            return list(self.model.backbone.parameters())
        return [p for p in self.model.parameters() if p.requires_grad]

    def _loss_components(
        self,
        session_id: int,
        images: Tensor,
        targets: Tensor,
        replay_mask: Tensor,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototypes: Tensor | None,
        replay_images: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        if self.model is None:
            raise RuntimeError("loss has no model")
        known = self.protocol.session(session_id).start
        total = self.protocol.session(session_id).stop
        new_count = self.protocol.session(session_id).size

        if isinstance(self.model, ExpandableLinearNet):
            output = self.model.forward_detailed(images)
            if self.method in {"joint", "finetune", "replay"}:
                return {"classification": F.cross_entropy(output.logits, targets)}, output.logits
            reference = (
                None
                if teacher is None
                else teacher.forward_detailed(images)
            )
            if self.method in {"icarl", "casper"}:
                loss = icarl_bce_loss(
                    output.logits,
                    targets,
                    old_logits=(None if reference is None else reference.logits),
                    known_classes=known,
                )
                components = {"classification": loss}
                if self.method == "casper" and replay_images is not None:
                    casper = self.config["method"].get("casper", {})
                    replay_features = self.model.extract_features(replay_images)
                    components["spectral"] = float(
                        casper.get("weight", 0.01)
                    ) * casper_spectral_loss(
                        replay_features,
                        num_classes=int(casper.get("classes_per_graph", new_count)),
                        k=int(casper.get("knn", 10)),
                        solver=str(casper.get("solver", "partial")),
                    )
                return components, output.logits
            if self.method == "sacil":
                if prototypes is None:
                    raise RuntimeError("SACIL training prototypes are missing")
                classification, prediction = prototype_cross_entropy(
                    output.features,
                    targets,
                    prototypes,
                    temperature=float(
                        self.config["method"].get("prototype_temperature", 0.1)
                    ),
                )
                components = {"classification": classification}
                if (
                    geometry is not None
                    and reference is not None
                    and bool(replay_mask.any())
                ):
                    components["geometry"] = float(
                        self.config["method"].get("lambda_geo", 1.0)
                    ) * geometry(
                        output.features[replay_mask],
                        reference.features[replay_mask],
                    )
                return components, prediction
            raise RuntimeError(f"unhandled linear method: {self.method}")

        if isinstance(self.model, AFCIncrementalNet):
            output = self.model.forward_detailed(images)
            if self.method == "podnet":
                components = {
                    "classification": podnet_nca_loss(
                        output.logits, targets, scale=1.0
                    )
                }
                if teacher is not None:
                    reference = teacher.forward_detailed(images)
                    factor = math.sqrt(total / new_count)
                    pod = self.config["method"].get("podnet", {})
                    components["pod_flat"] = float(
                        pod.get("flat_weight", 1.0)
                    ) * factor * pod_flat_loss(
                        output.features, reference.features
                    )
                    components["pod_spatial"] = float(
                        pod.get("spatial_weight", 5.0)
                    ) * factor * pod_spatial_loss(
                        output.attentions, reference.attentions
                    )
                return components, output.logits
            components = {
                "classification": afc_nca_loss(
                    output.logits, targets, 1.0
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                afc = self.config["method"].get("afc", {})
                factor = scheduled_afc_factor(
                    total, new_count, float(afc.get("pod_base_factor", 4.0))
                )
                components["afc"] = factor * afc_pod_loss(
                    reference.attentions,
                    output.attentions,
                    reference.importance[:-1],
                )
            return components, output.logits

        if isinstance(self.model, CREATEIncrementalNet):
            output = self.model.forward_detailed(images)
            create = self.config["method"].get("create", {})
            components = {
                "classification": create_classification_loss(
                    output["logits"], targets
                )
            }
            weights = reconstruction_confidence_weights(
                output["error_logits"],
                alpha=float(create.get("confidence_alpha", 2.0)),
            )
            components["contrastive"] = float(
                create.get("contrastive_weight", 1.0)
            ) * create_contrastive_loss(
                output["latents"], targets, sample_weights=weights
            )
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                components["kd"] = float(create.get("kd_weight", 1.0)) * old_logit_kl_loss(
                    output["error_logits"][:, :known],
                    reference["error_logits"],
                    temperature=float(create.get("kd_temperature", 2.0)),
                )
            return components, output["logits"]

        if isinstance(self.model, FGPIncrementalNet):
            output = self.model.forward_detailed(images)
            one_hot = F.one_hot(targets, num_classes=total).to(output.logits)
            components = {
                "classification": F.binary_cross_entropy_with_logits(
                    output.logits, one_hot
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                weight = scheduled_fgp_weight(
                    known,
                    total,
                    base_weight=float(
                        self.config["method"].get("fgp_weight", 0.1)
                    ),
                )
                components["graph"] = weight * fgp_graph_preservation_loss(
                    output.features,
                    reference.features,
                    self.model.classifier.weight[:known],
                    teacher.classifier.weight,
                )
            return components, output.logits

        if isinstance(self.model, CSCCTIncrementalNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": F.cross_entropy(output.logits, targets)
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                method = self.config["method"]
                components["kd"] = float(method.get("kd_weight", 0.25)) * old_logit_kl_loss(
                    output.logits[:, :known],
                    reference.logits,
                    temperature=float(method.get("kd_temperature", 2.0)),
                )
                components["csc"] = float(method.get("csc_weight", 3.0)) * cross_space_clustering_loss(
                    output.features, reference.features, targets
                )
                components["ct"] = float(method.get("ct_weight", 1.5)) * controlled_transfer_loss(
                    output.features,
                    reference.features,
                    targets,
                    known_classes=known,
                    temperature=float(method.get("ct_temperature", 2.0)),
                )
            return components, output.logits
        raise TypeError(f"unsupported model type: {type(self.model).__name__}")

    def _refresh_training_prototypes(self, loader: DataLoader | None) -> Tensor:
        if loader is None or self.model is None:
            raise RuntimeError("prototype refresh lacks model or loader")
        collection = collect_features(self.model, loader, self.device)
        prototypes = compute_prototypes(
            collection.features,
            collection.targets,
            range(self.model.num_classes),
        )
        return prototypes.to(self.device)

    def _casper_loader(
        self, session_id: int, batches: int
    ) -> DataLoader | None:
        if self.method != "casper" or session_id == 0:
            return None
        indices = self.memory.all_indices(self.protocol.class_order)
        dataset = self.data.replay_dataset(indices, augment=True)
        labels = [
            self.protocol.incremental_label(self.data.train_aug.targets[index])
            for index in indices
        ]
        config = self.config["method"].get("casper", {})
        sampler = BalancedClassBatchSampler(
            labels,
            batch_size=int(config.get("replay_batch_size", 64)),
            classes_per_batch=int(
                config.get(
                    "classes_per_graph", self.protocol.session(session_id).size
                )
            ),
            batches=batches,
            seed=self.seed * 10000 + session_id * 100,
        )
        return self._loader(
            dataset,
            shuffle=False,
            session_id=session_id + 6000,
            batch_sampler=sampler,
        )

    def _cscct_balanced_loader(self, session_id: int) -> DataLoader | None:
        if self.method != "cscct" or session_id == 0:
            return None
        old_indices = self.memory.all_indices(self.protocol.class_order)
        # CSCCT's official fusion update draws ``increment * exemplars`` new
        # images with replacement, then concatenates them with the complete
        # old memory (base_trainer.py::gen_balanced_loader).
        new_candidates = self.data.new_train_dataset(
            session_id,
            augment=True,
            samples_per_class=self.debug_train_samples_per_class,
        )
        rng = np.random.default_rng(self.seed * 10000 + session_id * 100 + 71)
        new_sample_count = (
            self.protocol.session(session_id).size
            * self._memory_limit(self.protocol.session(session_id).stop)
        )
        new_positions = rng.choice(
            len(new_candidates), size=new_sample_count, replace=True
        )
        new_indices = [
            new_candidates.indices[int(position)]
            for position in new_positions.tolist()
        ]
        dataset = ConcatDataset(
            [
                self.data.replay_dataset(old_indices, augment=True),
                self.data.train_dataset_from_indices(
                    new_indices, augment=True, is_replay=False
                ),
            ]
        )
        return self._loader(
            dataset, shuffle=True, session_id=session_id + 7000
        )

    def _update_cscct_fusion(
        self, loader: DataLoader, optimizer: SGD
    ) -> None:
        if not isinstance(self.model, CSCCTIncrementalNet):
            return
        self.model.eval()
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device).long()
            loss = F.cross_entropy(self.model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def _update_memory(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("memory update has no model")
        limit = self._memory_limit(self.protocol.session(session_id).stop)
        self.memory.resize_limit(limit)
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        collection = collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 8000),
            self.device,
        )
        selection = str(self.config["memory"].get("selection", "icarl_herding"))
        for class_id in self.protocol.classes_for_session(session_id):
            mask = collection.original_targets == int(class_id)
            function = (
                icarl_herding_select if selection == "icarl_herding" else herding_select
            )
            selected = function(
                collection.features[mask],
                collection.indices[mask].tolist(),
                limit,
            )
            self.memory.set_class_indices(class_id, selected)

    def _post_training(
        self, session_id: int, train_loader: DataLoader
    ) -> dict[str, Any] | None:
        if self.method == "afc":
            return {
                "finetuning": self._afc_finetune(session_id),
                "importance": self._afc_importance(train_loader),
            }
        if self.method == "create":
            return {"finetuning": self._create_finetune(session_id)}
        return None

    def _memory_loader(self, session_id: int, *, augment: bool) -> DataLoader:
        indices = self.memory.all_indices(self.protocol.class_order)
        return self._loader(
            self.data.replay_dataset(indices, augment=augment),
            shuffle=augment,
            session_id=session_id + 9000 + int(augment),
        )

    def _afc_finetune(self, session_id: int) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, AFCIncrementalNet):
            return None
        config = self.config["method"].get("afc", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.05)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                loss = afc_nca_loss(self.model(images), targets, 1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _afc_importance(self, loader: DataLoader) -> dict[str, Any]:
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC importance needs AFCIncrementalNet")
        self.model.backbone.reset_importance()
        self.model.backbone.start_importance_collection()
        self.model.train()
        batches = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device).long()
            self.model.zero_grad(set_to_none=True)
            loss = afc_nca_loss(self.model(images), targets, 1.0)
            loss.backward()
            batches += 1
        self.model.backbone.stop_importance_collection()
        self.model.backbone.normalize_importance()
        return {
            "batches": batches,
            "layer_means": [
                float(layer.importance.mean())
                for layer in self.model.backbone.importance_layers
            ],
        }

    def _create_finetune(self, session_id: int) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, CREATEIncrementalNet):
            return None
        config = self.config["method"].get("create", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                loss = create_classification_loss(self.model(images), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            scheduler.step()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _build_evaluation_state(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("evaluation state has no model")
        if self.method == "create":
            self.class_means = None
        else:
            if self.method == "joint":
                dataset = self.data.train_eval_dataset_for_classes(
                    self.protocol.seen_classes(session_id),
                    samples_per_class=self.debug_train_samples_per_class,
                )
                loader = self._loader(
                    dataset, shuffle=False, session_id=session_id + 10000
                )
            else:
                loader = self._memory_loader(session_id, augment=False)
            self.class_means = compute_nme_class_means(
                self.model,
                loader,
                self.device,
                self.protocol.session(session_id).stop,
                horizontal_flip=bool(
                    self.config.get("evaluation", {}).get(
                        "horizontal_flip", True
                    )
                ),
            ).cpu()
        if self.method == "sacil":
            self._build_sacil_artifacts(session_id)

    def _build_sacil_artifacts(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("SACIL artifacts have no model")
        loader = self._memory_loader(session_id, augment=False)
        collection = collect_features(self.model, loader, self.device)
        class_ids = self.protocol.seen_classes(session_id)
        values = compute_prototypes(
            collection.features, collection.original_targets, class_ids
        )
        prototypes = PrototypeBank(class_ids, values)
        hierarchy = self.config["method"].get("hierarchy", {})
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            values,
            temperature=float(hierarchy.get("taxonomy_temperature", 0.2)),
        )
        tree = GriffinPeronaGreedy().build(
            class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        self.sacil_tree = tree
        self.sacil_prototypes = prototypes
        self.sacil_anchors = anchors
        dump_json(
            tree.state_dict(), self.run_dir / f"tree_session_{session_id:02d}.json"
        )

    def _evaluate_session(self, session_id: int):
        if self.model is None:
            raise RuntimeError("evaluation has no model")
        dataset = self.data.cumulative_test_dataset(
            session_id, samples_per_class=self.debug_test_samples_per_class
        )
        loader = self._loader(
            dataset, shuffle=False, session_id=session_id + 11000
        )
        old = self.protocol.session(session_id).start
        if self.method == "create":
            return evaluate(self.model, loader, self.device, old)
        if self.class_means is None:
            raise RuntimeError("NME evaluation has no class means")
        return evaluate_nme(
            self.model, loader, self.device, old, self.class_means
        )

    def _save_checkpoint(self, session_id: int) -> Path:
        if self.model is None:
            raise RuntimeError("checkpoint has no model")
        path = self.checkpoint_dir / f"session_{session_id:02d}.pt"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "framework": "sacil-standalone",
            "pycil_used": False,
            "method": self.method,
            "session_id": session_id,
            "protocol_id": self.protocol.protocol_id,
            "config": self.config,
            "model": self.model.state_dict(),
            "model_type": type(self.model).__name__,
            "memory": self.memory.state_dict(),
            "metrics": self.metrics.state_dict(),
            "class_means": self.class_means,
        }
        if self.sacil_tree is not None:
            payload.update(
                tree=self.sacil_tree.state_dict(),
                prototypes=self.sacil_prototypes.state_dict(),
                anchors=self.sacil_anchors.state_dict(),
            )
        save_checkpoint(payload, path)
        return path
