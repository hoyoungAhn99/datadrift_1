from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from sacil.anchors import (
    HierarchicalAnchorBank,
    PrototypeBank,
    compute_prototypes,
)
from sacil.config import get_required
from sacil.data import ClassOrderProtocol, build_data_module
from sacil.engine.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from sacil.engine.evaluator import evaluate
from sacil.features import FeatureCollection, collect_features
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.memory import ExemplarMemory, herding_select, icarl_herding_select
from sacil.methods import (
    AnchorGeometryLoss,
    ConflictWeights,
    afc_nca_loss,
    afc_pod_loss,
    compute_conflict_weights,
    global_preservation_weights,
    method_uses_afc,
    method_uses_geometry,
    scheduled_afc_factor,
)
from sacil.metrics import CILMetricsTracker, summarize_geometry_drift
from sacil.models import (
    AFCIncrementalNet,
    IncrementalNet,
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


@dataclass
class SessionArtifacts:
    tree: HierarchyTree
    prototypes: PrototypeBank
    anchors: HierarchicalAnchorBank


class SACILTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        project_root: str | Path,
        *,
        resume: str | Path | None = None,
        max_sessions: int | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.project_root = Path(project_root).expanduser().resolve()
        self.seed = int(self.config.get("seed", 1))
        set_seed(
            self.seed,
            deterministic=bool(self.config.get("deterministic", True)),
        )
        self.device = resolved_device(str(self.config.get("device", "cuda:0")))
        self.protocol = ClassOrderProtocol.from_json(
            self._project_path(get_required(self.config, "data.protocol"))
        )
        self.method_name = str(get_required(self.config, "method.name"))
        method_uses_geometry(self.method_name)
        self.is_afc = method_uses_afc(self.method_name)
        self.data = build_data_module(
            str(self.config["data"].get("name", "cifar100")),
            self._project_path(get_required(self.config, "data.root")),
            self.protocol,
            download=bool(self.config["data"].get("download", False)),
            color_jitter=bool(
                self.config["data"].get("color_jitter", False)
            ),
        )
        self.exemplars_per_class = int(
            get_required(self.config, "memory.exemplars_per_class")
        )
        self.memory = ExemplarMemory(self.exemplars_per_class)
        self.metrics = CILMetricsTracker()
        self.artifacts: SessionArtifacts | None = None
        self.model: IncrementalNet | AFCIncrementalNet | None = None
        self.start_session = 0
        self.max_sessions = (
            self.protocol.num_sessions
            if max_sessions is None
            else min(int(max_sessions), self.protocol.num_sessions)
        )
        self.run_dir = ensure_dir(
            self._project_path(get_required(self.config, "output.directory"))
            / str(get_required(self.config, "output.run_name"))
            / f"seed_{self.seed}"
        )
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self._write_run_metadata()
        if resume is not None:
            self._resume(resume)

    def _project_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _write_run_metadata(self) -> None:
        metadata = {
            "config": self.config,
            "project_root": str(self.project_root),
            "protocol_id": self.protocol.protocol_id,
            "device": str(self.device),
            "git_commit": git_commit(self.project_root),
        }
        dump_json(metadata, self.run_dir / "resolved_config.json")

    def _resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, map_location="cpu")
        if checkpoint["protocol_id"] != self.protocol.protocol_id:
            raise ValueError("checkpoint protocol does not match configuration")
        checkpoint_method = str(
            checkpoint.get("config", {})
            .get("method", {})
            .get("name", self.method_name)
        )
        if method_uses_afc(checkpoint_method) != self.is_afc:
            raise ValueError("checkpoint model family does not match configuration")
        num_classes = int(checkpoint["num_classes"])
        self.model = self._new_model(num_classes)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])
        self.metrics = CILMetricsTracker.from_state_dict(
            checkpoint["metrics"]
        )
        self.artifacts = SessionArtifacts(
            tree=HierarchyTree.from_state_dict(checkpoint["tree"]),
            prototypes=PrototypeBank.from_state_dict(
                checkpoint["prototypes"]
            ),
            anchors=HierarchicalAnchorBank.from_state_dict(
                checkpoint["anchors"]
            ),
        )
        self.start_session = int(checkpoint["session_id"]) + 1
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

    def _new_model(
        self, num_classes: int
    ) -> IncrementalNet | AFCIncrementalNet:
        model_config = self.config["model"]
        if self.is_afc:
            if str(model_config.get("backbone")) != "afc_resnet32":
                raise ValueError("AFC methods require backbone=afc_resnet32")
            return AFCIncrementalNet(
                num_classes=num_classes,
                initial_size=self.protocol.session(0).size,
                increment_size=self.protocol.session(1).size,
                proxies_per_class=int(
                    model_config.get("proxies_per_class", 10)
                ),
                classifier_scale=float(
                    model_config.get("classifier_scale", 1.0)
                ),
            )
        return IncrementalNet(
            num_classes=num_classes,
            backbone=str(model_config.get("backbone", "resnet32")),
            classifier_scale=float(model_config.get("classifier_scale", 10.0)),
            learnable_scale=bool(
                model_config.get("learnable_scale", True)
            ),
        )

    def _loader(
        self,
        dataset,
        *,
        shuffle: bool,
        session_id: int,
        batch_size: int | None = None,
    ) -> DataLoader:
        training = self.config["training"]
        return DataLoader(
            dataset,
            batch_size=int(batch_size or training["batch_size"]),
            shuffle=shuffle,
            num_workers=int(training.get("num_workers", 0)),
            pin_memory=bool(training.get("pin_memory", True)),
            persistent_workers=(
                int(training.get("num_workers", 0)) > 0
                and bool(training.get("persistent_workers", False))
            ),
            worker_init_fn=seed_worker,
            generator=make_generator(self.seed * 1000 + session_id),
            drop_last=False,
        )

    @property
    def debug_train_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("train_samples_per_class")
        return None if value is None else int(value)

    @property
    def debug_test_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("test_samples_per_class")
        return None if value is None else int(value)

    def run(self) -> dict:
        if self.start_session >= self.max_sessions:
            return self.metrics.summary()
        for session_id in range(self.start_session, self.max_sessions):
            started = time.time()
            session_log = self._run_session(session_id)
            session_log["elapsed_seconds"] = time.time() - started
            self._append_session_log(session_log)
        summary = self.metrics.summary()
        dump_json(
            {
                "summary": summary,
                "sessions": self.metrics.records,
            },
            self.run_dir / "metrics.json",
        )
        return summary

    def _run_session(self, session_id: int) -> dict:
        session = self.protocol.session(session_id)
        old_class_count = session.start
        seen_class_count = session.stop
        old_memory_indices = self.memory.all_indices(
            self.protocol.class_order
        )
        teacher: IncrementalNet | AFCIncrementalNet | None = None
        geometry_loss: AnchorGeometryLoss | None = None
        conflict_weights: ConflictWeights | None = None
        incoming_prototypes: Tensor | None = None

        if session_id == 0:
            self.model = self._new_model(seen_class_count).to(self.device)
        else:
            if self.model is None or self.artifacts is None:
                raise RuntimeError("incremental session requires previous state")
            teacher = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            incoming_collection = self._incoming_feature_collection(
                teacher, session_id
            )
            incoming_prototypes = compute_prototypes(
                incoming_collection.features,
                incoming_collection.original_targets,
                self.protocol.classes_for_session(session_id),
            )
            if self.is_afc:
                if not isinstance(self.model, AFCIncrementalNet):
                    raise TypeError("AFC method requires AFCIncrementalNet")
                class_features = [
                    incoming_collection.features[
                        incoming_collection.original_targets == int(class_id)
                    ]
                    for class_id in self.protocol.classes_for_session(
                        session_id
                    )
                ]
                imprinted = kmeans_imprinted_weights(
                    class_features,
                    self.model.classifier.weights,
                    proxies_per_class=self.model.classifier.proxies_per_class,
                    random_state=self.seed * 1000 + session_id * 100,
                )
                self.model.expand_classes(
                    seen_class_count, imprinted.to(self.device)
                )
            else:
                if not isinstance(self.model, IncrementalNet):
                    raise TypeError("standard method requires IncrementalNet")
                self.model.expand_classes(seen_class_count)
                self.model.classifier.initialize_rows(
                    old_class_count, incoming_prototypes.to(self.device)
                )
            self.model.to(self.device)
            if method_uses_geometry(self.method_name):
                if self.method_name in {"global_hap", "afc_global_hap"}:
                    conflict_weights = global_preservation_weights(
                        self.artifacts.anchors
                    )
                else:
                    conflict_weights = self._conflict_weights(
                        incoming_prototypes
                    )
                use_internal = self.method_name not in {
                    "flat_lrhap",
                    "afc_flat_lrhap",
                }
                geometry_loss = AnchorGeometryLoss(
                    self.artifacts.anchors,
                    conflict_weights.leaf_weights,
                    conflict_weights.internal_weights,
                    use_internal_anchors=use_internal,
                ).to(self.device)

        train_dataset = self.data.training_dataset(
            session_id,
            old_memory_indices,
            samples_per_class=self.debug_train_samples_per_class,
        )
        train_loader = self._loader(
            train_dataset, shuffle=True, session_id=session_id
        )
        training_log = self._train_session(
            session_id, train_loader, teacher, geometry_loss
        )

        self._update_memory(session_id)
        finetuning_log = None
        importance_log = None
        if self.is_afc:
            finetuning_log = self._afc_finetune_classifier(session_id)
            importance_log = self._update_afc_importance(train_loader)

        geometry_log = None
        if (
            session_id > 0
            and teacher is not None
            and geometry_loss is not None
            and conflict_weights is not None
            and old_memory_indices
        ):
            geometry_log = self._geometry_diagnostics(
                old_memory_indices,
                teacher,
                geometry_loss,
                conflict_weights,
                session_id,
            )

        self.artifacts = self._build_posthoc_artifacts(session_id)
        evaluation = self._evaluate(session_id)
        metric_record = self.metrics.update(session_id, evaluation)
        checkpoint_path = self._save_session_checkpoint(
            session_id, conflict_weights
        )
        return {
            "session_id": session_id,
            "kind": session.kind,
            "new_class_ids": list(session.class_ids),
            "seen_class_count": seen_class_count,
            "memory_size": len(self.memory),
            "training": training_log,
            "finetuning": finetuning_log,
            "importance": importance_log,
            "evaluation": metric_record,
            "geometry": geometry_log,
            "conflict": (
                None
                if conflict_weights is None
                else self._conflict_log(conflict_weights)
            ),
            "checkpoint": str(checkpoint_path),
        }

    def _phase_training_config(self, session_id: int) -> dict:
        phase = "base" if session_id == 0 else "incremental"
        return dict(self.config["training"][phase])

    def _train_session(
        self,
        session_id: int,
        loader: DataLoader,
        teacher: IncrementalNet | AFCIncrementalNet | None,
        geometry_loss: AnchorGeometryLoss | None,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        if self.is_afc:
            if not isinstance(self.model, AFCIncrementalNet):
                raise TypeError("AFC method requires AFCIncrementalNet")
            if teacher is not None and not isinstance(
                teacher, AFCIncrementalNet
            ):
                raise TypeError("AFC teacher has the wrong model type")
            return self._train_afc_session(
                session_id, loader, teacher, geometry_loss
            )
        phase = self._phase_training_config(session_id)
        optimizer = SGD(
            self.model.parameters(),
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 5e-4)),
            nesterov=bool(phase.get("nesterov", True)),
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[int(value) for value in phase.get("milestones", [])],
            gamma=float(phase.get("gamma", 0.1)),
        )
        epochs = int(phase["epochs"])
        max_batches = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = None if max_batches is None else int(max_batches)
        lambda_geo = float(self.config["method"].get("lambda_geo", 1.0))
        epoch_logs = []
        self.model.train()
        for epoch in range(epochs):
            totals = {"loss": 0.0, "classification": 0.0, "geometry": 0.0}
            sample_count = 0
            batch_count = 0
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(
                    self.device, non_blocking=True
                )
                targets = batch["target"].to(
                    self.device, non_blocking=True
                ).long()
                replay_mask = batch["is_replay"].to(
                    self.device, non_blocking=True
                ).bool()
                logits, features = self.model(
                    images, return_features=True
                )
                classification = F.cross_entropy(logits, targets)
                geometry = features.sum() * 0.0
                if (
                    teacher is not None
                    and geometry_loss is not None
                    and bool(replay_mask.any())
                ):
                    with torch.no_grad():
                        reference_features = teacher.extract_features(
                            images[replay_mask]
                        )
                    geometry = geometry_loss(
                        features[replay_mask], reference_features
                    )
                loss = classification + lambda_geo * geometry
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = targets.numel()
                sample_count += batch_size
                batch_count += 1
                totals["loss"] += float(loss.detach().item()) * batch_size
                totals["classification"] += (
                    float(classification.detach().item()) * batch_size
                )
                totals["geometry"] += (
                    float(geometry.detach().item()) * batch_size
                )
            scheduler.step()
            if sample_count == 0:
                raise RuntimeError("training loop processed no samples")
            epoch_logs.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "batches": batch_count,
                    **{
                        key: value / sample_count
                        for key, value in totals.items()
                    },
                }
            )
        return {
            "epochs": epochs,
            "samples_in_dataset": len(loader.dataset),
            "epoch_logs": epoch_logs,
        }

    @staticmethod
    def _scheduler(optimizer: SGD, phase: dict, epochs: int):
        scheduler_name = str(phase.get("scheduler", "step")).lower()
        if scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler_name == "step":
            return MultiStepLR(
                optimizer,
                milestones=[
                    int(value) for value in phase.get("milestones", [])
                ],
                gamma=float(phase.get("gamma", 0.1)),
            )
        raise ValueError(f"unknown scheduler: {scheduler_name}")

    def _train_afc_session(
        self,
        session_id: int,
        loader: DataLoader,
        teacher: AFCIncrementalNet | None,
        geometry_loss: AnchorGeometryLoss | None,
    ) -> dict:
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC training requires AFCIncrementalNet")
        phase = self._phase_training_config(session_id)
        for parameter in self.model.classifier.old_weights:
            parameter.requires_grad_(False)
        optimizer = SGD(
            self.model.main_trainable_parameters(),
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 5e-4)),
            nesterov=bool(phase.get("nesterov", False)),
        )
        epochs = int(phase["epochs"])
        scheduler = self._scheduler(optimizer, phase, epochs)
        max_batches = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = None if max_batches is None else int(max_batches)
        method_config = self.config["method"]
        afc_config = method_config.get("afc", {})
        lambda_geo = float(method_config.get("lambda_geo", 1.0))
        nca_margin = float(afc_config.get("nca_margin", 0.6))
        pod_base_factor = float(afc_config.get("pod_base_factor", 4.0))
        seen_class_count = self.protocol.session(session_id).stop
        new_class_count = self.protocol.session(session_id).size
        pod_factor = (
            0.0
            if teacher is None
            else scheduled_afc_factor(
                seen_class_count, new_class_count, pod_base_factor
            )
        )
        epoch_logs = []
        self.model.train()
        for epoch in range(epochs):
            totals = {
                "loss": 0.0,
                "classification": 0.0,
                "distillation": 0.0,
                "geometry": 0.0,
            }
            sample_count = 0
            batch_count = 0
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(
                    self.device, non_blocking=True
                )
                targets = batch["target"].to(
                    self.device, non_blocking=True
                ).long()
                replay_mask = batch["is_replay"].to(
                    self.device, non_blocking=True
                ).bool()
                output = self.model.forward_detailed(images)
                classification = afc_nca_loss(
                    output.logits,
                    targets,
                    self.model.postprocessor_scale,
                    margin=nca_margin,
                )
                distillation = output.features.sum() * 0.0
                geometry = output.features.sum() * 0.0
                if teacher is not None:
                    with torch.no_grad():
                        reference = teacher.forward_detailed(images)
                    distillation = pod_factor * afc_pod_loss(
                        reference.attentions,
                        output.attentions,
                        reference.importance[:-1],
                    )
                    if (
                        geometry_loss is not None
                        and bool(replay_mask.any())
                    ):
                        geometry = geometry_loss(
                            output.features[replay_mask],
                            reference.features[replay_mask],
                        )
                loss = (
                    classification
                    + distillation
                    + lambda_geo * geometry
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in self.model.main_trainable_parameters():
                    if parameter.grad is not None:
                        parameter.grad.clamp_(min=-5.0, max=5.0)
                optimizer.step()
                batch_size = targets.numel()
                sample_count += batch_size
                batch_count += 1
                for key, value in (
                    ("loss", loss),
                    ("classification", classification),
                    ("distillation", distillation),
                    ("geometry", geometry),
                ):
                    totals[key] += float(value.detach().item()) * batch_size
            scheduler.step()
            if sample_count == 0:
                raise RuntimeError("AFC training loop processed no samples")
            epoch_logs.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "batches": batch_count,
                    **{
                        key: value / sample_count
                        for key, value in totals.items()
                    },
                }
            )
        for parameter in self.model.classifier.old_weights:
            parameter.requires_grad_(True)
        return {
            "epochs": epochs,
            "samples_in_dataset": len(loader.dataset),
            "pod_factor": pod_factor,
            "epoch_logs": epoch_logs,
        }

    def _afc_finetune_classifier(self, session_id: int) -> dict | None:
        if session_id == 0:
            return None
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC fine-tuning requires AFCIncrementalNet")
        config = self.config["method"].get("afc", {}).get(
            "finetuning", {}
        )
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        memory_indices = self.memory.all_indices(self.protocol.class_order)
        dataset = self.data.replay_dataset(memory_indices, augment=True)
        loader = self._loader(
            dataset,
            shuffle=True,
            session_id=session_id + 10_000,
        )
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        self.model.postprocessor_scale.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        optimizer = SGD(
            self.model.classifier_parameters(),
            lr=float(config.get("lr", 0.05)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            nesterov=bool(config.get("nesterov", False)),
        )
        max_batches = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = None if max_batches is None else int(max_batches)
        nca_margin = float(
            self.config["method"].get("afc", {}).get("nca_margin", 0.6)
        )
        epoch_logs = []
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            sample_count = 0
            batch_count = 0
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(
                    self.device, non_blocking=True
                )
                targets = batch["target"].to(
                    self.device, non_blocking=True
                ).long()
                logits = self.model(images)
                loss = afc_nca_loss(
                    logits,
                    targets,
                    self.model.postprocessor_scale,
                    margin=nca_margin,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in self.model.classifier_parameters():
                    if parameter.grad is not None:
                        parameter.grad.clamp_(min=-5.0, max=5.0)
                optimizer.step()
                sample_count += targets.numel()
                batch_count += 1
                total_loss += float(loss.detach().item()) * targets.numel()
            if sample_count == 0:
                raise RuntimeError("AFC fine-tuning processed no samples")
            epoch_logs.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "batches": batch_count,
                    "loss": total_loss / sample_count,
                }
            )
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        self.model.postprocessor_scale.requires_grad_(True)
        return {
            "epochs": epochs,
            "samples_in_dataset": len(dataset),
            "epoch_logs": epoch_logs,
        }

    def _update_afc_importance(self, loader: DataLoader) -> dict:
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC importance requires AFCIncrementalNet")
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(False)
        self.model.postprocessor_scale.requires_grad_(False)
        self.model.backbone.reset_importance()
        self.model.backbone.start_importance_collection()
        self.model.train()
        nca_margin = float(
            self.config["method"].get("afc", {}).get("nca_margin", 0.6)
        )
        max_batches = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = None if max_batches is None else int(max_batches)
        batch_count = 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(
                self.device, non_blocking=True
            ).long()
            self.model.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = afc_nca_loss(
                logits,
                targets,
                self.model.postprocessor_scale,
                margin=nca_margin,
            )
            loss.backward()
            batch_count += 1
        self.model.backbone.stop_importance_collection()
        self.model.backbone.normalize_importance()
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        self.model.postprocessor_scale.requires_grad_(True)
        means = [
            float(layer.importance.mean().item())
            for layer in self.model.backbone.importance_layers
        ]
        return {
            "batches": batch_count,
            "normalized_layer_means": means,
        }

    def _incoming_feature_collection(
        self,
        teacher: IncrementalNet | AFCIncrementalNet,
        session_id: int,
    ) -> FeatureCollection:
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        return collect_features(
            teacher,
            self._loader(
                dataset, shuffle=False, session_id=session_id
            ),
            self.device,
        )

    def _conflict_weights(
        self, incoming_prototypes: Tensor
    ) -> ConflictWeights:
        if self.artifacts is None:
            raise RuntimeError("conflict weights require previous artifacts")
        conflict = self.config["method"]["conflict"]
        return compute_conflict_weights(
            incoming_prototypes,
            self.artifacts.anchors,
            self.artifacts.tree,
            max_neighbors=int(conflict.get("max_neighbors", 5)),
            old_class_ratio=float(conflict.get("old_class_ratio", 0.1)),
            temperature=float(conflict.get("temperature", 0.05)),
            min_preservation_weight=float(
                conflict.get("min_preservation_weight", 0.1)
            ),
            ancestor_decay=float(conflict.get("ancestor_decay", 0.5)),
        )

    def _update_memory(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        collection = collect_features(
            self.model,
            self._loader(
                dataset, shuffle=False, session_id=session_id
            ),
            self.device,
        )
        for class_id in self.protocol.classes_for_session(session_id):
            mask = collection.original_targets == int(class_id)
            selection = str(
                self.config["memory"].get("selection", "herding")
            )
            if selection == "herding":
                selected = herding_select(
                    collection.features[mask],
                    collection.indices[mask].tolist(),
                    self.exemplars_per_class,
                )
            elif selection == "icarl_herding":
                selected = icarl_herding_select(
                    collection.features[mask],
                    collection.indices[mask].tolist(),
                    self.exemplars_per_class,
                )
            else:
                raise ValueError(
                    f"unsupported exemplar selection: {selection}"
                )
            self.memory.set_class_indices(class_id, selected)

    def _build_posthoc_artifacts(
        self, session_id: int
    ) -> SessionArtifacts:
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        seen_classes = self.protocol.seen_classes(session_id)
        memory_indices = self.memory.all_indices(self.protocol.class_order)
        dataset = self.data.train_eval_dataset_from_indices(memory_indices)
        collection = collect_features(
            self.model,
            self._loader(
                dataset, shuffle=False, session_id=session_id
            ),
            self.device,
        )
        prototype_tensor = compute_prototypes(
            collection.features,
            collection.original_targets,
            seen_classes,
        )
        prototypes = PrototypeBank(seen_classes, prototype_tensor)
        hierarchy_config = self.config["method"].get("hierarchy", {})
        temperature = float(
            hierarchy_config.get("taxonomy_temperature", 0.2)
        )
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            self.model.classifier.weight.detach().cpu(),
            temperature=temperature,
        )
        affinity = symmetric_affinity(confusion)
        tree = GriffinPeronaGreedy().build(seen_classes, affinity)
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        dump_json(
            tree.state_dict(),
            self.run_dir / f"tree_session_{session_id:02d}.json",
        )
        return SessionArtifacts(tree, prototypes, anchors)

    def _evaluate(self, session_id: int):
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        dataset = self.data.cumulative_test_dataset(
            session_id,
            samples_per_class=self.debug_test_samples_per_class,
        )
        return evaluate(
            self.model,
            self._loader(
                dataset, shuffle=False, session_id=session_id
            ),
            self.device,
            old_class_count=self.protocol.session(session_id).start,
        )

    def _geometry_diagnostics(
        self,
        old_memory_indices: list[int],
        teacher: IncrementalNet,
        geometry_loss: AnchorGeometryLoss,
        conflict_weights: ConflictWeights,
        session_id: int,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        dataset = self.data.train_eval_dataset_from_indices(
            old_memory_indices
        )
        loader = self._loader(
            dataset, shuffle=False, session_id=session_id
        )
        current = collect_features(self.model, loader, self.device)
        reference = collect_features(teacher, loader, self.device)
        per_anchor = geometry_loss.per_anchor_drift(
            current.features.to(self.device),
            reference.features.to(self.device),
        )
        return summarize_geometry_drift(
            per_anchor,
            conflict_weights.leaf_weights,
            conflict_weights.internal_weights,
        )

    @staticmethod
    def _conflict_log(weights: ConflictWeights) -> dict:
        leaf_weights = weights.leaf_weights
        ranked = torch.argsort(leaf_weights)
        return {
            "neighbors_per_new_class": weights.neighbors_per_new_class,
            "leaf_weight_min": float(leaf_weights.min().item()),
            "leaf_weight_mean": float(leaf_weights.mean().item()),
            "leaf_weight_max": float(leaf_weights.max().item()),
            "most_relaxed_leaf_classes": [
                weights.leaf_class_ids[int(position)]
                for position in ranked[: min(10, ranked.numel())]
            ],
            "internal_weight_min": (
                None
                if weights.internal_weights.numel() == 0
                else float(weights.internal_weights.min().item())
            ),
            "internal_weight_mean": (
                None
                if weights.internal_weights.numel() == 0
                else float(weights.internal_weights.mean().item())
            ),
        }

    def _save_session_checkpoint(
        self,
        session_id: int,
        conflict_weights: ConflictWeights | None,
    ) -> Path:
        if self.model is None or self.artifacts is None:
            raise RuntimeError("cannot checkpoint incomplete session state")
        path = self.checkpoint_dir / f"session_{session_id:02d}.pt"
        save_checkpoint(
            {
                "schema_version": 1,
                "protocol_id": self.protocol.protocol_id,
                "session_id": session_id,
                "num_classes": self.model.num_classes,
                "model": self.model.state_dict(),
                "memory": self.memory.state_dict(),
                "metrics": self.metrics.state_dict(),
                "tree": self.artifacts.tree.state_dict(),
                "prototypes": self.artifacts.prototypes.state_dict(),
                "anchors": self.artifacts.anchors.state_dict(),
                "conflict_weights": (
                    None
                    if conflict_weights is None
                    else conflict_weights.state_dict()
                ),
                "config": self.config,
            },
            path,
        )
        return path

    def _append_session_log(self, session_log: dict) -> None:
        jsonl_path = self.run_dir / "sessions.jsonl"
        with jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(session_log, ensure_ascii=False))
            handle.write("\n")
        evaluation = session_log["evaluation"]
        csv_path = self.run_dir / "session_metrics.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "session_id",
                    "accuracy",
                    "old_accuracy",
                    "new_accuracy",
                    "harmonic_mean",
                    "memory_size",
                    "elapsed_seconds",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "session_id": session_log["session_id"],
                    "accuracy": evaluation["accuracy"],
                    "old_accuracy": evaluation["old_accuracy"],
                    "new_accuracy": evaluation["new_accuracy"],
                    "harmonic_mean": evaluation["harmonic_mean"],
                    "memory_size": session_log["memory_size"],
                    "elapsed_seconds": session_log["elapsed_seconds"],
                }
            )
