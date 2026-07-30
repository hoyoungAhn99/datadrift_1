from __future__ import annotations

import copy
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# These are intentionally the official PyCIL top-level imports. The runtime
# adapter places the pinned PyCIL checkout at the front of sys.path before
# importing this module.
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

from sacil.anchors import (
    HierarchicalAnchorBank,
    PrototypeBank,
    compute_prototypes,
)
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.methods import (
    AnchorGeometryLoss,
    ConflictWeights,
    compute_conflict_weights,
    global_preservation_weights,
)
from sacil.utils import dump_json, ensure_dir


@dataclass
class _FeatureCollection:
    features: Tensor
    targets: Tensor


@dataclass
class _SessionArtifacts:
    tree: HierarchyTree
    prototypes: PrototypeBank
    anchors: HierarchicalAnchorBank


@contextmanager
def preserve_rng_state():
    """Keep SACIL's deterministic post-hoc work RNG-neutral to PyCIL."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def pycil_kd_loss(pred: Tensor, soft: Tensor, temperature: float) -> Tensor:
    """The old-logit distillation used by PyCIL's official iCaRL learner."""

    if pred.ndim != 2 or soft.ndim != 2 or pred.shape != soft.shape:
        raise ValueError("student and teacher old-logit shapes must match")
    if pred.shape[0] == 0:
        return pred.sum() * 0.0
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    log_prob = F.log_softmax(pred / float(temperature), dim=1)
    probability = F.softmax(soft / float(temperature), dim=1)
    return -(probability * log_prob).sum() / pred.shape[0]


@torch.inference_mode()
def collect_pycil_features(
    network: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> _FeatureCollection:
    was_training = network.training
    network.eval()
    features: list[Tensor] = []
    targets: list[Tensor] = []
    for _, images, labels in loader:
        output = network(images.to(device, non_blocking=True))
        features.append(output["features"].detach().cpu())
        targets.append(labels.detach().cpu().long())
    network.train(was_training)
    if not features:
        raise ValueError("cannot collect features from an empty PyCIL loader")
    return _FeatureCollection(
        features=torch.cat(features, dim=0),
        targets=torch.cat(targets, dim=0),
    )


class SACIL(BaseLearner):
    """SACIL on the official PyCIL iCaRL/replay lifecycle.

    PyCIL owns the dataset/task split, classifier expansion, exemplar memory,
    herding, and evaluation. SACIL contributes only the post-hoc hierarchical
    anchors, conflict-aware relaxation, and geometry-preservation loss.
    """

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._artifacts: _SessionArtifacts | None = None
        self._geometry_loss: AnchorGeometryLoss | None = None
        self._conflict_weights: ConflictWeights | None = None

        self.batch_size = int(args.get("batch_size", 128))
        self.num_workers = int(args.get("num_workers", 0))
        self.pin_memory = bool(args.get("pin_memory", True))

        self.init_epochs = int(args.get("init_epochs", 200))
        self.init_lr = float(args.get("init_lr", 0.1))
        self.init_milestones = [
            int(value)
            for value in args.get("init_milestones", [60, 120, 170])
        ]
        self.init_weight_decay = float(
            args.get("init_weight_decay", 5e-4)
        )

        self.epochs = int(args.get("epochs", 170))
        self.lr = float(args.get("lr", 0.1))
        self.milestones = [
            int(value) for value in args.get("milestones", [80, 120])
        ]
        self.weight_decay = float(args.get("weight_decay", 2e-4))
        self.momentum = float(args.get("momentum", 0.9))
        self.lr_decay = float(args.get("lr_decay", 0.1))
        self.nesterov = bool(args.get("nesterov", False))

        self.lambda_kd = float(args.get("lambda_kd", 1.0))
        self.kd_temperature = float(args.get("kd_temperature", 2.0))
        self.kd_scope = str(args.get("kd_scope", "all")).lower()
        if self.kd_scope not in {"all", "replay"}:
            raise ValueError("kd_scope must be 'all' or 'replay'")

        self.lambda_geo = float(args.get("lambda_geo", 1.0))
        self.geometry_mode = str(
            args.get("geometry_mode", "sacil")
        ).lower()
        if self.geometry_mode not in {"none", "global", "flat", "sacil"}:
            raise ValueError(
                "geometry_mode must be none, global, flat, or sacil"
            )
        self.taxonomy_temperature = float(
            args.get("taxonomy_temperature", 0.2)
        )
        self.conflict_max_neighbors = int(
            args.get("conflict_max_neighbors", 5)
        )
        self.conflict_old_class_ratio = float(
            args.get("conflict_old_class_ratio", 0.1)
        )
        self.conflict_temperature = float(
            args.get("conflict_temperature", 0.05)
        )
        self.min_preservation_weight = float(
            args.get("min_preservation_weight", 0.1)
        )
        self.ancestor_decay = float(args.get("ancestor_decay", 0.5))

        self.eval_interval = int(args.get("eval_interval", 5))
        self.max_batches_per_epoch = args.get("max_batches_per_epoch")
        if self.max_batches_per_epoch is not None:
            self.max_batches_per_epoch = int(self.max_batches_per_epoch)
        self.disable_tqdm = bool(args.get("disable_tqdm", False))
        self.save_checkpoints = bool(args.get("save_checkpoints", True))
        self.artifact_dir = ensure_dir(
            Path(args.get("artifact_dir", "outputs/pycil/sacil"))
        )

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: %d", self.exemplar_size)
        if self.save_checkpoints:
            self._save_checkpoint()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on %d-%d", self._known_classes, self._total_classes
        )

        self._geometry_loss = None
        self._conflict_weights = None
        if self._cur_task > 0:
            if self._old_network is None or self._artifacts is None:
                raise RuntimeError(
                    "incremental SACIL task requires teacher and anchors"
                )
            # Keep this extra feature pass from perturbing PyCIL's classifier
            # initialization and DataLoader RNG stream. It is executed for
            # every geometry mode so the controlled variants have the same
            # non-training lifecycle.
            with preserve_rng_state():
                incoming = self._incoming_prototypes(data_manager)
            if self.geometry_mode == "global":
                self._conflict_weights = global_preservation_weights(
                    self._artifacts.anchors
                )
            elif self.geometry_mode in {"flat", "sacil"}:
                self._conflict_weights = compute_conflict_weights(
                    incoming,
                    self._artifacts.anchors,
                    self._artifacts.tree,
                    max_neighbors=self.conflict_max_neighbors,
                    old_class_ratio=self.conflict_old_class_ratio,
                    temperature=self.conflict_temperature,
                    min_preservation_weight=self.min_preservation_weight,
                    ancestor_decay=self.ancestor_decay,
                )
            if (
                self.geometry_mode != "none"
                and self.lambda_geo != 0
                and self._conflict_weights is not None
            ):
                self._geometry_loss = AnchorGeometryLoss(
                    self._artifacts.anchors,
                    self._conflict_weights.leaf_weights,
                    self._conflict_weights.internal_weights,
                    use_internal_anchors=self.geometry_mode != "flat",
                ).to(self._device)
                logging.info(
                    "%s weights: leaf min=%.4f mean=%.4f",
                    self.geometry_mode,
                    self._conflict_weights.leaf_weights.min().item(),
                    self._conflict_weights.leaf_weights.mean().item(),
                )

        self._network.update_fc(self._total_classes)
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = self._loader(train_dataset, shuffle=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = self._loader(test_dataset, shuffle=False)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(
                self._network, self._multiple_gpus
            )
        self._train(self.train_loader, self.test_loader)

        # These calls deliberately use the official PyCIL BaseLearner
        # implementation and its iCaRL herding policy.
        self.build_rehearsal_memory(
            data_manager, self.samples_per_class
        )
        # Artifact construction is analysis-only and must not shift PyCIL's
        # RNG stream before the following task.
        with preserve_rng_state():
            self._artifacts = self._build_posthoc_artifacts(data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _network_without_parallel(self) -> nn.Module:
        if isinstance(self._network, nn.DataParallel):
            return self._network.module
        return self._network

    def _classifier_weights(self) -> Tensor:
        network = self._network_without_parallel()
        classifier = network.fc
        if not hasattr(classifier, "weight"):
            raise TypeError(
                "PyCIL SACIL currently requires a single SimpleLinear head"
            )
        return classifier.weight.detach().cpu()

    def _incoming_prototypes(self, data_manager) -> Tensor:
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        loader = self._loader(dataset, shuffle=False)
        collection = collect_pycil_features(
            self._old_network, loader, self._device
        )
        return compute_prototypes(
            collection.features,
            collection.targets,
            range(self._known_classes, self._total_classes),
        )

    def _phase(self) -> tuple[int, float, list[int], float]:
        if self._cur_task == 0:
            return (
                self.init_epochs,
                self.init_lr,
                self.init_milestones,
                self.init_weight_decay,
            )
        return self.epochs, self.lr, self.milestones, self.weight_decay

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        epochs, learning_rate, milestones, weight_decay = self._phase()
        optimizer = optim.SGD(
            self._network.parameters(),
            lr=learning_rate,
            momentum=self.momentum,
            weight_decay=weight_decay,
            nesterov=self.nesterov,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.lr_decay
        )

        iterator = tqdm(range(epochs), disable=self.disable_tqdm)
        for epoch in iterator:
            self._network.train()
            total_loss = 0.0
            total_clf = 0.0
            total_kd = 0.0
            total_geo = 0.0
            correct, total, batches = 0, 0, 0
            for batch_index, (_, inputs, targets) in enumerate(train_loader):
                if (
                    self.max_batches_per_epoch is not None
                    and batch_index >= self.max_batches_per_epoch
                ):
                    break
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True).long()
                output = self._network(inputs)
                logits = output["logits"]
                features = output["features"]
                classification = F.cross_entropy(logits, targets)
                distillation = features.sum() * 0.0
                geometry = features.sum() * 0.0

                if self._old_network is not None:
                    with torch.no_grad():
                        reference = self._old_network(inputs)
                    replay_mask = targets < self._known_classes
                    if self.lambda_kd != 0:
                        if self.kd_scope == "all":
                            kd_current = logits[:, : self._known_classes]
                            kd_reference = reference["logits"]
                        else:
                            kd_current = logits[
                                replay_mask, : self._known_classes
                            ]
                            kd_reference = reference["logits"][replay_mask]
                        distillation = pycil_kd_loss(
                            kd_current,
                            kd_reference,
                            self.kd_temperature,
                        )
                    if (
                        self._geometry_loss is not None
                        and bool(replay_mask.any())
                    ):
                        geometry = self._geometry_loss(
                            features[replay_mask],
                            reference["features"][replay_mask],
                        )

                loss = (
                    classification
                    + self.lambda_kd * distillation
                    + self.lambda_geo * geometry
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = targets.numel()
                total += count
                batches += 1
                total_loss += float(loss.detach().item()) * count
                total_clf += float(classification.detach().item()) * count
                total_kd += float(distillation.detach().item()) * count
                total_geo += float(geometry.detach().item()) * count
                correct += (
                    logits.argmax(dim=1).eq(targets).detach().cpu().sum()
                )

            scheduler.step()
            if total == 0:
                raise RuntimeError("PyCIL SACIL processed no training samples")
            train_accuracy = np.around(
                tensor2numpy(correct) * 100 / total, decimals=2
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.4f}, CE {:.4f}, "
                "KD {:.4f}, Geo {:.6f}, Train_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                epochs,
                total_loss / total,
                total_clf / total,
                total_kd / total,
                total_geo / total,
                train_accuracy,
            )
            if (
                self.eval_interval > 0
                and epoch % self.eval_interval == 0
            ):
                test_accuracy = self._compute_accuracy(
                    self._network, test_loader
                )
                info += ", Test_accy {:.2f}".format(test_accuracy)
            iterator.set_description(info)
            logging.info(info)

    def _build_posthoc_artifacts(self, data_manager) -> _SessionArtifacts:
        memory = self._get_memory()
        if memory is None:
            raise RuntimeError("SACIL anchors require exemplar memory")
        dataset = data_manager.get_dataset(
            [], source="train", mode="test", appendent=memory
        )
        collection = collect_pycil_features(
            self._network,
            self._loader(dataset, shuffle=False),
            self._device,
        )
        class_ids: Sequence[int] = tuple(range(self._total_classes))
        prototype_tensor = compute_prototypes(
            collection.features, collection.targets, class_ids
        )
        prototypes = PrototypeBank(class_ids, prototype_tensor)
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            self._classifier_weights(),
            temperature=self.taxonomy_temperature,
        )
        tree = GriffinPeronaGreedy().build(
            class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        dump_json(
            tree.state_dict(),
            self.artifact_dir
            / f"tree_task_{self._cur_task:02d}.json",
        )
        return _SessionArtifacts(tree, prototypes, anchors)

    def _save_checkpoint(self) -> None:
        if self._artifacts is None:
            raise RuntimeError("cannot save SACIL without anchor artifacts")
        path = self.artifact_dir / f"task_{self._cur_task:02d}.pt"
        torch.save(
            {
                "schema_version": 1,
                "framework": "PyCIL",
                "task": self._cur_task,
                "known_classes": self._known_classes,
                "model_state_dict": self._network.state_dict(),
                "memory_data": copy.deepcopy(self._data_memory),
                "memory_targets": copy.deepcopy(self._targets_memory),
                "tree": self._artifacts.tree.state_dict(),
                "prototypes": self._artifacts.prototypes.state_dict(),
                "anchors": self._artifacts.anchors.state_dict(),
                "conflict": (
                    None
                    if self._conflict_weights is None
                    else self._conflict_weights.state_dict()
                ),
            },
            path,
        )
