from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.anchors import compute_prototypes
from sacil.engine.checkpoint import load_checkpoint
from sacil.engine.evaluator import EvaluationResult
from sacil.engine.evaluator import evaluate as evaluate_classifier
from sacil.engine.evaluator import evaluate_nme
from sacil.engine.table1_trainer import UnifiedTable1Trainer
from sacil.features import collect_features
from sacil.memory import ExemplarMemory
from sacil.methods import normalized_cosine_classifier_logits
from sacil.methods.prototype_transport import (
    affine_class_residual_transport,
    affine_ridge_transport,
    classwise_translation_transport,
    rigid_procrustes_transport,
)
from sacil.provenance import build_exploration_provenance
from sacil.utils import dump_json, git_commit


_CHECKPOINT_PATTERN = re.compile(r"^session_(\d+)\.pt$")


@dataclass(frozen=True)
class CMPTExperimentSettings:
    learner: str
    checkpoint_directory: Path
    output_file: Path
    expected_checkpoint_method: str
    expected_sessions: int = 11
    expected_exemplars_per_class: int = 20
    expected_casper_enabled: bool = False
    device: str = "cuda:0"
    transport: str = "rigid_procrustes"
    affine_ridge: float = 1.0e-2
    prototype_horizontal_flip: bool = True
    support_horizontal_flip: bool = True
    query_horizontal_flip: bool = False
    center_strength: float = 0.0
    strict_parity: bool = True
    parity_tolerance: float = 1.0e-6
    cpu_threads: int = 6
    prototype_interpolation_alphas: tuple[float, ...] = ()
    adaptive_alpha_enabled: bool = False
    adaptive_alpha_folds: int = 5
    accuracy_oracle_enabled: bool = False
    class_geometric_oracle_enabled: bool = False
    oracle_alpha_grid: tuple[float, ...] = ()
    component_ablation_enabled: bool = False

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        project_root: str | Path,
    ) -> "CMPTExperimentSettings":
        root = Path(project_root).expanduser().resolve()
        experiment = _required_mapping(config, "experiment")
        cmpt = _required_mapping(config, "cmpt")
        runtime = config.get("runtime", {})
        if runtime is None:
            runtime = {}
        if not isinstance(runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        cpu_threads = int(runtime.get("cpu_threads", torch.get_num_threads()))
        if cpu_threads <= 0:
            raise ValueError("runtime.cpu_threads must be positive")

        def project_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        transport = str(cmpt.get("transport", "rigid_procrustes")).lower()
        if transport not in {"rigid_procrustes", "affine_ridge"}:
            raise ValueError(
                "cmpt.transport must be rigid_procrustes or affine_ridge"
            )
        affine_ridge = float(cmpt.get("affine_ridge", 1.0e-2))
        if affine_ridge <= 0.0:
            raise ValueError("cmpt.affine_ridge must be positive")
        if not bool(cmpt.get("full_introduction_prototypes", True)):
            raise ValueError(
                "CMPT requires full_introduction_prototypes=true"
            )
        if not bool(cmpt.get("replace_old_classes_only", True)):
            raise ValueError("CMPT primary evaluator replaces old classes only")
        center_strength = float(cmpt.get("center_strength", 0.0))
        if not 0.0 <= center_strength <= 1.0:
            raise ValueError("cmpt.center_strength must be in [0, 1]")
        parity_tolerance = float(cmpt.get("parity_tolerance", 1.0e-6))
        if parity_tolerance < 0.0:
            raise ValueError("cmpt.parity_tolerance must be non-negative")
        raw_alphas = cmpt.get("prototype_interpolation_alphas", ())
        if raw_alphas is None:
            raw_alphas = ()
        if isinstance(raw_alphas, (str, bytes)) or not isinstance(
            raw_alphas, Sequence
        ):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must be a sequence"
            )
        interpolation_alphas = tuple(float(value) for value in raw_alphas)
        if any(
            not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
            for alpha in interpolation_alphas
        ):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must lie in [0, 1]"
            )
        if len(set(interpolation_alphas)) != len(interpolation_alphas):
            raise ValueError(
                "cmpt.prototype_interpolation_alphas must be unique"
            )
        if interpolation_alphas and not (
            0.0 in interpolation_alphas and 1.0 in interpolation_alphas
        ):
            raise ValueError(
                "an interpolation sweep must include alpha=0 and alpha=1"
            )
        adaptive_alpha = cmpt.get("adaptive_alpha", {})
        if adaptive_alpha is None:
            adaptive_alpha = {}
        if not isinstance(adaptive_alpha, Mapping):
            raise ValueError("cmpt.adaptive_alpha must be a mapping")
        adaptive_alpha_folds = int(adaptive_alpha.get("folds", 5))
        if adaptive_alpha_folds < 2:
            raise ValueError("cmpt.adaptive_alpha.folds must be at least 2")
        oracle = cmpt.get("oracle_diagnostics", {})
        if oracle is None:
            oracle = {}
        if not isinstance(oracle, Mapping):
            raise ValueError("cmpt.oracle_diagnostics must be a mapping")
        raw_oracle_grid = oracle.get(
            "alpha_grid", interpolation_alphas
        )
        if raw_oracle_grid is None:
            raw_oracle_grid = ()
        if isinstance(raw_oracle_grid, (str, bytes)) or not isinstance(
            raw_oracle_grid, Sequence
        ):
            raise ValueError(
                "cmpt.oracle_diagnostics.alpha_grid must be a sequence"
            )
        oracle_alpha_grid = tuple(float(value) for value in raw_oracle_grid)
        if any(
            not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
            for alpha in oracle_alpha_grid
        ):
            raise ValueError("oracle alpha grid values must lie in [0, 1]")
        if len(set(oracle_alpha_grid)) != len(oracle_alpha_grid):
            raise ValueError("oracle alpha grid values must be unique")
        accuracy_oracle_enabled = bool(oracle.get("accuracy", False))
        class_geometric_oracle_enabled = bool(
            oracle.get("class_geometric", False)
        )
        if (accuracy_oracle_enabled or class_geometric_oracle_enabled) and not (
            oracle_alpha_grid
            and 0.0 in oracle_alpha_grid
            and 1.0 in oracle_alpha_grid
        ):
            raise ValueError(
                "oracle diagnostics require a grid containing 0 and 1"
            )
        if accuracy_oracle_enabled and not interpolation_alphas:
            raise ValueError(
                "accuracy oracle requires prototype_interpolation_alphas"
            )
        if accuracy_oracle_enabled and set(oracle_alpha_grid) != set(
            interpolation_alphas
        ):
            raise ValueError(
                "accuracy oracle grid must equal the interpolation grid"
            )
        component_ablation = cmpt.get("component_ablation", {})
        if component_ablation is None:
            component_ablation = {}
        if not isinstance(component_ablation, Mapping):
            raise ValueError("cmpt.component_ablation must be a mapping")
        component_ablation_enabled = bool(
            component_ablation.get("enabled", False)
        )
        if component_ablation_enabled and transport != "affine_ridge":
            raise ValueError(
                "CMPT component ablation requires affine_ridge transport"
            )

        return cls(
            learner=str(experiment["learner"]),
            checkpoint_directory=project_path(experiment["checkpoints"]),
            output_file=project_path(experiment["output"]),
            expected_checkpoint_method=str(
                experiment["expected_checkpoint_method"]
            ).lower(),
            expected_sessions=int(experiment.get("expected_sessions", 11)),
            expected_exemplars_per_class=int(
                experiment.get("expected_exemplars_per_class", 20)
            ),
            expected_casper_enabled=bool(
                experiment.get("expected_casper_enabled", False)
            ),
            device=str(config.get("device", "cuda:0")),
            transport=transport,
            affine_ridge=affine_ridge,
            prototype_horizontal_flip=bool(
                cmpt.get("prototype_horizontal_flip", True)
            ),
            support_horizontal_flip=bool(
                cmpt.get("support_horizontal_flip", True)
            ),
            query_horizontal_flip=bool(
                cmpt.get("query_horizontal_flip", False)
            ),
            center_strength=center_strength,
            strict_parity=bool(cmpt.get("strict_parity", True)),
            parity_tolerance=parity_tolerance,
            cpu_threads=cpu_threads,
            prototype_interpolation_alphas=interpolation_alphas,
            adaptive_alpha_enabled=bool(
                adaptive_alpha.get("enabled", False)
            ),
            adaptive_alpha_folds=adaptive_alpha_folds,
            accuracy_oracle_enabled=accuracy_oracle_enabled,
            class_geometric_oracle_enabled=(
                class_geometric_oracle_enabled
            ),
            oracle_alpha_grid=oracle_alpha_grid,
            component_ablation_enabled=component_ablation_enabled,
        )


@dataclass(frozen=True)
class TrajectoryAudit:
    learner: str
    checkpoint_count: int
    session_ids: tuple[int, ...]
    protocol_id: str
    checkpoint_method: str
    evaluation_classifier: str
    feature_dim: int
    exemplars_per_class: int
    final_memory_classes: int
    old_exemplar_identities_stable: bool
    casper_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_ids"] = list(self.session_ids)
        return payload


@dataclass(frozen=True)
class NativeClassifierSpec:
    """Test-time classifier that belongs to a learner's native contract."""

    classifier: str
    implementation: str
    distinct_from_nme: bool
    uses_learned_parameters: bool
    query_horizontal_flip: bool = False
    scale: float | None = None
    epsilon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedTransportSupport:
    """Paired replay features for affine fitting and uncertainty estimates."""

    old_fit_features: Tensor
    current_fit_features: Tensor
    fit_targets: Tensor
    old_exemplar_features: Tensor
    current_exemplar_features: Tensor
    targets: Tensor

    @property
    def fit_support_count(self) -> int:
        return int(self.old_fit_features.shape[0])

    @property
    def exemplar_count(self) -> int:
        return int(self.old_exemplar_features.shape[0])


def resolve_native_classifier(
    checkpoint: Mapping[str, Any],
) -> NativeClassifierSpec:
    """Resolve native inference without confusing a training-only FC with it.

    The common-recipe LUCIR adapter intentionally uses the iCaRL substrate
    name, so its normalized-cosine native head is identified from the saved
    feature-cosine configuration rather than from a learner-name string.
    """

    contract = _required_mapping(checkpoint, "method_contract")
    method = str(contract.get("name", "")).lower()
    config = _required_mapping(checkpoint, "config")
    method_config = config.get("method", {})
    if not isinstance(method_config, Mapping):
        raise ValueError("checkpoint config.method must be a mapping")

    if method == "icarl":
        feature_cosine = method_config.get(
            "feature_cosine_distillation", {}
        )
        if feature_cosine is None:
            feature_cosine = {}
        if not isinstance(feature_cosine, Mapping):
            raise ValueError(
                "method.feature_cosine_distillation must be a mapping"
            )
        if bool(feature_cosine.get("enabled", False)):
            training_classifier = str(
                feature_cosine.get("training_classifier", "")
            ).lower()
            if training_classifier != "normalized_cosine":
                raise ValueError(
                    "enabled LUCIR feature distillation requires its saved "
                    "normalized_cosine training classifier"
                )
            return NativeClassifierSpec(
                classifier="normalized_cosine_head",
                implementation="lucir_training_cosine_logits",
                distinct_from_nme=True,
                uses_learned_parameters=True,
                scale=float(feature_cosine.get("cosine_scale", 10.0)),
                epsilon=float(feature_cosine.get("epsilon", 1.0e-12)),
            )
        return NativeClassifierSpec(
            classifier="exemplar_nme",
            implementation="native_nme_alias",
            distinct_from_nme=False,
            uses_learned_parameters=False,
        )

    if method in {"casper", "sacil"}:
        return NativeClassifierSpec(
            classifier="exemplar_nme",
            implementation="native_nme_alias",
            distinct_from_nme=False,
            uses_learned_parameters=False,
        )

    if method == "create":
        return NativeClassifierSpec(
            classifier="classwise_reconstruction_error",
            implementation="model_forward_logits",
            distinct_from_nme=True,
            uses_learned_parameters=True,
        )

    if method in {
        "joint",
        "finetune",
        "replay",
        "podnet",
        "afc",
        "fgp",
        "cscct",
    }:
        return NativeClassifierSpec(
            classifier="learned_classification_head",
            implementation="model_forward_logits",
            distinct_from_nme=True,
            uses_learned_parameters=True,
        )

    raise ValueError(f"unsupported native classifier contract: {method}")


class _NativeClassifierView(nn.Module):
    """Expose exactly the saved learner's native logits to the evaluator."""

    def __init__(self, model: nn.Module, spec: NativeClassifierSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.eval()

    def forward(self, images: Tensor) -> Tensor:
        if self.spec.implementation == "model_forward_logits":
            logits = self.model(images)
        elif self.spec.implementation == "lucir_training_cosine_logits":
            classifier = getattr(self.model, "classifier", None)
            class_weights = getattr(classifier, "weight", None)
            if not isinstance(class_weights, Tensor):
                raise TypeError("LUCIR native evaluation requires FC weights")
            features = self.model.extract_features(images)
            logits = normalized_cosine_classifier_logits(
                features,
                class_weights,
                scale=float(self.spec.scale),
                epsilon=float(self.spec.epsilon),
            )
        else:
            raise RuntimeError(
                "NME aliases must not be sent through a native-head view"
            )
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise TypeError("native classifier must return a logits matrix")
        return logits


def _required_mapping(
    mapping: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def discover_checkpoint_paths(
    checkpoint_directory: str | Path,
) -> list[Path]:
    directory = Path(checkpoint_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    indexed: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_PATTERN.match(path.name)
        if path.is_file() and match is not None:
            indexed.append((int(match.group(1)), path.resolve()))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        raise FileNotFoundError(f"no session checkpoints found in {directory}")
    session_ids = [item[0] for item in indexed]
    if session_ids != list(range(len(session_ids))):
        raise ValueError(
            "checkpoint sequence must be contiguous from S0; found "
            f"{session_ids}"
        )
    return [item[1] for item in indexed]


def _memory_indices(checkpoint: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    memory = _required_mapping(checkpoint, "memory")
    raw = _required_mapping(memory, "indices")
    return {
        int(class_id): tuple(int(index) for index in indices)
        for class_id, indices in raw.items()
    }


def _checkpoint_session_metric(
    checkpoint: Mapping[str, Any], session_id: int
) -> Mapping[str, Any]:
    metrics = _required_mapping(checkpoint, "metrics")
    records = metrics.get("records")
    if not isinstance(records, Sequence):
        raise ValueError("checkpoint metrics.records must be a sequence")
    for record in records:
        if isinstance(record, Mapping) and int(record["session_id"]) == session_id:
            return record
    raise ValueError(f"checkpoint lacks stored metric for session {session_id}")


def audit_checkpoint_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
    settings: CMPTExperimentSettings,
) -> TrajectoryAudit:
    if len(checkpoints) != settings.expected_sessions:
        raise ValueError(
            f"{settings.learner}: expected {settings.expected_sessions} "
            f"checkpoints, found {len(checkpoints)}"
        )
    session_ids = tuple(int(value["session_id"]) for value in checkpoints)
    if session_ids != tuple(range(len(checkpoints))):
        raise ValueError(
            f"{settings.learner}: non-contiguous session IDs {session_ids}"
        )

    protocols: set[str] = set()
    methods: set[str] = set()
    classifiers: set[str] = set()
    dimensions: set[int] = set()
    casper_flags: set[bool] = set()
    prototype_flip_flags: set[bool] = set()
    previous_memory: dict[int, tuple[int, ...]] | None = None
    stable = True

    for session_id, checkpoint in enumerate(checkpoints):
        protocols.add(str(checkpoint.get("protocol_id", "")))
        contract = _required_mapping(checkpoint, "method_contract")
        methods.add(str(contract.get("name", "")).lower())
        classifiers.add(str(contract.get("evaluation_classifier", "")).lower())
        means = checkpoint.get("class_means")
        if not isinstance(means, Tensor) or means.ndim != 2:
            raise ValueError(
                f"{settings.learner} S{session_id}: class_means is missing"
            )
        if not bool(torch.isfinite(means).all()):
            raise ValueError(
                f"{settings.learner} S{session_id}: non-finite class means"
            )
        dimensions.add(int(means.shape[1]))
        memory = _memory_indices(checkpoint)
        memory_state = _required_mapping(checkpoint, "memory")
        if int(memory_state["exemplars_per_class"]) != (
            settings.expected_exemplars_per_class
        ):
            raise ValueError(
                f"{settings.learner} S{session_id}: memory limit does not "
                "match the expected exemplar count"
            )
        if len(memory) != int(means.shape[0]):
            raise ValueError(
                f"{settings.learner} S{session_id}: memory/class-mean count "
                f"mismatch ({len(memory)} vs {means.shape[0]})"
            )
        counts = {len(indices) for indices in memory.values()}
        if counts != {settings.expected_exemplars_per_class}:
            raise ValueError(
                f"{settings.learner} S{session_id}: expected "
                f"{settings.expected_exemplars_per_class} exemplars/class, "
                f"found {sorted(counts)}"
            )
        if previous_memory is not None:
            for class_id, indices in previous_memory.items():
                if memory.get(class_id) != indices:
                    stable = False
        previous_memory = memory
        casper = checkpoint.get("casper_options", {})
        casper_flags.add(
            bool(casper.get("enabled", False))
            if isinstance(casper, Mapping)
            else False
        )
        checkpoint_config = _required_mapping(checkpoint, "config")
        evaluation = _required_mapping(checkpoint_config, "evaluation")
        prototype_flip_flags.add(bool(evaluation.get("horizontal_flip", False)))
        _checkpoint_session_metric(checkpoint, session_id)

    if len(protocols) != 1 or "" in protocols:
        raise ValueError(f"{settings.learner}: inconsistent protocol IDs")
    if methods != {settings.expected_checkpoint_method}:
        raise ValueError(
            f"{settings.learner}: expected checkpoint method "
            f"{settings.expected_checkpoint_method!r}, found {sorted(methods)}"
        )
    if classifiers != {"nme"}:
        raise ValueError(
            f"{settings.learner}: CMPT requires NME checkpoints, found "
            f"{sorted(classifiers)}"
        )
    if len(dimensions) != 1:
        raise ValueError(
            f"{settings.learner}: feature dimensions change across sessions: "
            f"{sorted(dimensions)}"
        )
    if prototype_flip_flags != {settings.prototype_horizontal_flip}:
        raise ValueError(
            f"{settings.learner}: checkpoint NME prototype flip convention "
            f"{sorted(prototype_flip_flags)} does not match CMPT setting "
            f"{settings.prototype_horizontal_flip}"
        )
    if not stable:
        raise ValueError(
            f"{settings.learner}: old exemplar identities change across sessions"
        )
    if len(casper_flags) != 1:
        raise ValueError(
            f"{settings.learner}: CaSpeR flag changes across sessions: "
            f"{sorted(casper_flags)}"
        )
    casper_enabled = casper_flags == {True}
    if settings.expected_casper_enabled != casper_enabled:
        raise ValueError(
            f"{settings.learner}: expected_casper_enabled="
            f"{settings.expected_casper_enabled}, observed {sorted(casper_flags)}"
        )
    assert previous_memory is not None
    return TrajectoryAudit(
        learner=settings.learner,
        checkpoint_count=len(checkpoints),
        session_ids=session_ids,
        protocol_id=next(iter(protocols)),
        checkpoint_method=next(iter(methods)),
        evaluation_classifier="nme",
        feature_dim=next(iter(dimensions)),
        exemplars_per_class=settings.expected_exemplars_per_class,
        final_memory_classes=len(previous_memory),
        old_exemplar_identities_stable=stable,
        casper_enabled=casper_enabled,
    )


def _load_model(
    trainer: UnifiedTable1Trainer,
    checkpoint: Mapping[str, Any],
    session_id: int,
) -> nn.Module:
    seen_classes = trainer.protocol.session(int(session_id)).stop
    base_classes = trainer.protocol.session(0).stop

    # Some incremental classifiers cannot be reconstructed by passing the
    # final class count to their constructor.  PODNet stores one consolidated
    # old chunk plus the latest chunk, while CSCCT stores every task chunk and
    # creates its scale-shift branch at the first expansion.  Replaying only
    # these parameter-free architecture transitions yields the exact module
    # graph expected by the checkpoint; it does not run training or use data.
    if trainer.method == "podnet" and session_id > 0:
        model = trainer._new_model(base_classes).to(trainer.device)
        for step in range(1, int(session_id) + 1):
            model.expand_classes(trainer.protocol.session(step).stop)
    elif trainer.method == "cscct" and session_id > 0:
        model = trainer._new_model(base_classes).to(trainer.device)
        for step in range(1, int(session_id) + 1):
            increment = trainer.protocol.session(step).size
            dummy = torch.zeros(
                increment,
                int(model.feature_dim),
                device=trainer.device,
            )
            model.expand_classes(dummy)
    else:
        model = trainer._new_model(int(seen_classes)).to(trainer.device)
    expected_type = str(checkpoint.get("model_type", ""))
    if expected_type and type(model).__name__ != expected_type:
        raise TypeError(
            f"checkpoint expects {expected_type}, reconstructed "
            f"{type(model).__name__}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _full_introduction_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> Tensor:
    class_ids = trainer.protocol.classes_for_session(session_id)
    dataset = trainer.data.train_eval_dataset_for_classes(
        class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 13000
    )
    regular = collect_features(model, loader, trainer.device)
    features = regular.features
    targets = regular.original_targets
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("full-prototype flip rows are misaligned")
        features = torch.cat([features, flipped.features], dim=0)
        targets = torch.cat(
            [targets, flipped.original_targets], dim=0
        )
    return compute_prototypes(features, targets, class_ids).cpu()


def _full_current_old_prototypes(
    trainer: UnifiedTable1Trainer,
    model: nn.Module,
    session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, int]:
    """Oracle-only current means from all old-class training images."""

    class_ids = trainer.protocol.old_classes(session_id)
    if not class_ids:
        raise ValueError("class-geometric oracle requires old classes")
    dataset = trainer.data.train_eval_dataset_for_classes(
        class_ids,
        samples_per_class=trainer.debug_train_samples_per_class,
    )
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 15000
    )
    regular = collect_features(model, loader, trainer.device)
    features = regular.features
    targets = regular.original_targets
    if horizontal_flip:
        flipped = collect_features(
            model, loader, trainer.device, horizontal_flip=True
        )
        if not torch.equal(regular.indices, flipped.indices):
            raise RuntimeError("geometric-oracle flip rows are misaligned")
        features = torch.cat([features, flipped.features], dim=0)
        targets = torch.cat(
            [targets, flipped.original_targets], dim=0
        )
    return (
        compute_prototypes(features, targets, class_ids).cpu(),
        len(dataset),
    )


def _paired_support_features(
    trainer: UnifiedTable1Trainer,
    previous_model: nn.Module,
    current_model: nn.Module,
    previous_session_id: int,
    *,
    horizontal_flip: bool,
) -> PairedTransportSupport:
    loader = trainer._memory_loader(previous_session_id, augment=False)
    old = collect_features(previous_model, loader, trainer.device)
    current = collect_features(current_model, loader, trainer.device)
    if not torch.equal(old.indices, current.indices):
        raise RuntimeError("old/current transport support rows are misaligned")
    if not torch.equal(old.targets, current.targets):
        raise RuntimeError("old/current transport support labels are misaligned")
    old_features = old.features
    current_features = current.features
    fit_targets = current.targets.detach().cpu().long()
    old_exemplar_features = F.normalize(old.features.float(), dim=1)
    current_exemplar_features = F.normalize(current.features.float(), dim=1)
    if horizontal_flip:
        old_flip = collect_features(
            previous_model, loader, trainer.device, horizontal_flip=True
        )
        current_flip = collect_features(
            current_model, loader, trainer.device, horizontal_flip=True
        )
        if not (
            torch.equal(old.indices, old_flip.indices)
            and torch.equal(old.indices, current_flip.indices)
        ):
            raise RuntimeError("transport-support flip rows are misaligned")
        old_features = torch.cat([old_features, old_flip.features], dim=0)
        current_features = torch.cat(
            [current_features, current_flip.features], dim=0
        )
        fit_targets = torch.cat([fit_targets, fit_targets], dim=0)
        old_exemplar_features = F.normalize(
            old_exemplar_features
            + F.normalize(old_flip.features.float(), dim=1),
            dim=1,
        )
        current_exemplar_features = F.normalize(
            current_exemplar_features
            + F.normalize(current_flip.features.float(), dim=1),
            dim=1,
        )
    return PairedTransportSupport(
        old_fit_features=old_features,
        current_fit_features=current_features,
        fit_targets=fit_targets,
        old_exemplar_features=old_exemplar_features,
        current_exemplar_features=current_exemplar_features,
        targets=current.targets.detach().cpu().long(),
    )


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    cmpt = [float(record["cmpt"]["accuracy"]) for record in records]
    native = [float(record["native"]["accuracy"]) for record in records]
    parity = [abs(float(record["parity_error"])) for record in records]
    incremental_baseline = baseline[1:] if len(baseline) > 1 else baseline
    incremental_cmpt = cmpt[1:] if len(cmpt) > 1 else cmpt
    incremental_native = native[1:] if len(native) > 1 else native
    baseline_aia = sum(baseline) / len(baseline)
    cmpt_aia = sum(cmpt) / len(cmpt)
    native_aia = sum(native) / len(native)
    return {
        "native_aia_percent": 100.0 * native_aia,
        "baseline_aia_percent": 100.0 * baseline_aia,
        "cmpt_aia_percent": 100.0 * cmpt_aia,
        "aia_delta_percent_points": 100.0 * (cmpt_aia - baseline_aia),
        "nme_minus_native_aia_percent_points": 100.0
        * (baseline_aia - native_aia),
        "cmpt_minus_native_aia_percent_points": 100.0
        * (cmpt_aia - native_aia),
        "native_incremental_aia_percent": 100.0
        * sum(incremental_native)
        / len(incremental_native),
        "baseline_incremental_aia_percent": 100.0
        * sum(incremental_baseline)
        / len(incremental_baseline),
        "cmpt_incremental_aia_percent": 100.0
        * sum(incremental_cmpt)
        / len(incremental_cmpt),
        "native_final_percent": 100.0 * native[-1],
        "baseline_final_percent": 100.0 * baseline[-1],
        "cmpt_final_percent": 100.0 * cmpt[-1],
        "final_delta_percent_points": 100.0 * (cmpt[-1] - baseline[-1]),
        "nme_minus_native_final_percent_points": 100.0
        * (baseline[-1] - native[-1]),
        "cmpt_minus_native_final_percent_points": 100.0
        * (cmpt[-1] - native[-1]),
        "max_parity_error_percent_points": 100.0 * max(parity),
    }


def _aggregate_component_ablation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate matched NME/global/class/combined prototype estimators."""

    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    global_affine = [float(record["cmpt"]["accuracy"]) for record in records]
    class_translation = [
        float(record["class_translation"]["accuracy"]) for record in records
    ]
    combined = [
        float(record["combined_cmpt"]["accuracy"]) for record in records
    ]
    baseline_aia = sum(baseline) / len(baseline)
    global_aia = sum(global_affine) / len(global_affine)
    class_aia = sum(class_translation) / len(class_translation)
    combined_aia = sum(combined) / len(combined)

    def metrics(values: Sequence[float]) -> dict[str, float]:
        incremental = values[1:] if len(values) > 1 else values
        aia = sum(values) / len(values)
        return {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
        }

    class_metrics = metrics(class_translation)
    class_metrics.update(
        {
            "aia_delta_vs_nme_percent_points": 100.0
            * (class_aia - baseline_aia),
            "aia_delta_vs_global_percent_points": 100.0
            * (class_aia - global_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (class_translation[-1] - baseline[-1]),
            "final_delta_vs_global_percent_points": 100.0
            * (class_translation[-1] - global_affine[-1]),
        }
    )
    combined_metrics = metrics(combined)
    combined_metrics.update(
        {
            "aia_delta_vs_nme_percent_points": 100.0
            * (combined_aia - baseline_aia),
            "aia_delta_vs_global_percent_points": 100.0
            * (combined_aia - global_aia),
            "aia_delta_vs_class_percent_points": 100.0
            * (combined_aia - class_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (combined[-1] - baseline[-1]),
            "final_delta_vs_global_percent_points": 100.0
            * (combined[-1] - global_affine[-1]),
            "final_delta_vs_class_percent_points": 100.0
            * (combined[-1] - class_translation[-1]),
        }
    )
    return {
        "nme": metrics(baseline),
        "global_affine": metrics(global_affine),
        "class_translation": class_metrics,
        "combined_cmpt": combined_metrics,
    }


def _aggregate_interpolation(
    records: Sequence[Mapping[str, Any]],
    alphas: Sequence[float],
) -> dict[str, dict[str, float]]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    baseline_final = baseline[-1]
    result: dict[str, dict[str, float]] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        values = [
            float(record["prototype_interpolation"][key]["accuracy"])
            for record in records
        ]
        incremental = values[1:] if len(values) > 1 else values
        aia = sum(values) / len(values)
        result[key] = {
            "alpha": float(alpha),
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (values[-1] - baseline_final),
        }
    return result


def _aggregate_adaptive_alpha(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    baseline_final = baseline[-1]
    result: dict[str, dict[str, float]] = {}
    for mode in ("session", "raw_class", "shrunken_class"):
        values = [
            float(record["adaptive_alpha"][mode]["metrics"]["accuracy"])
            for record in records
        ]
        incremental = values[1:] if len(values) > 1 else values
        alpha_means = [
            float(record["adaptive_alpha"][mode]["alpha_mean"])
            for record in records[1:]
        ]
        aia = sum(values) / len(values)
        result[mode] = {
            "aia_percent": 100.0 * aia,
            "incremental_aia_percent": 100.0
            * sum(incremental)
            / len(incremental),
            "final_percent": 100.0 * values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (aia - baseline_aia),
            "final_delta_vs_nme_percent_points": 100.0
            * (values[-1] - baseline_final),
            "mean_incremental_alpha": (
                sum(alpha_means) / len(alpha_means)
                if alpha_means
                else 0.0
            ),
        }
    return result


def _aggregate_accuracy_oracle(
    records: Sequence[Mapping[str, Any]],
    alphas: Sequence[float],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    pure = [float(record["cmpt"]["accuracy"]) for record in records]
    baseline_aia = sum(baseline) / len(baseline)
    pure_aia = sum(pure) / len(pure)

    trajectories: dict[float, list[float]] = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        trajectories[float(alpha)] = [
            float(record["prototype_interpolation"][key]["accuracy"])
            for record in records
        ]
    global_alpha, global_values = max(
        trajectories.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), item[0]),
    )
    global_aia = sum(global_values) / len(global_values)

    session_alphas: list[float] = []
    session_values: list[float] = []
    for index in range(len(records)):
        alpha, value = max(
            (
                (alpha, values[index])
                for alpha, values in trajectories.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        session_alphas.append(float(alpha))
        session_values.append(float(value))
    session_aia = sum(session_values) / len(session_values)
    return {
        "learner_global": {
            "alpha": global_alpha,
            "aia_percent": 100.0 * global_aia,
            "final_percent": 100.0 * global_values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (global_aia - baseline_aia),
            "aia_delta_vs_pure_cmpt_percent_points": 100.0
            * (global_aia - pure_aia),
        },
        "session": {
            "alphas": session_alphas,
            "mean_incremental_alpha": (
                sum(session_alphas[1:]) / len(session_alphas[1:])
                if len(session_alphas) > 1
                else session_alphas[0]
            ),
            "aia_percent": 100.0 * session_aia,
            "final_percent": 100.0 * session_values[-1],
            "aia_delta_vs_nme_percent_points": 100.0
            * (session_aia - baseline_aia),
            "aia_delta_vs_pure_cmpt_percent_points": 100.0
            * (session_aia - pure_aia),
        },
        "selection_uses_test_accuracy": True,
        "oracle_only": True,
    }


def _aggregate_class_geometric_oracle(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    pure = [float(record["cmpt"]["accuracy"]) for record in records]
    values = [
        float(record["class_geometric_oracle"]["metrics"]["accuracy"])
        for record in records
    ]
    alpha_means = [
        float(record["class_geometric_oracle"]["alpha_mean"])
        for record in records[1:]
    ]
    baseline_aia = sum(baseline) / len(baseline)
    pure_aia = sum(pure) / len(pure)
    aia = sum(values) / len(values)
    return {
        "aia_percent": 100.0 * aia,
        "final_percent": 100.0 * values[-1],
        "aia_delta_vs_nme_percent_points": 100.0 * (aia - baseline_aia),
        "aia_delta_vs_pure_cmpt_percent_points": 100.0
        * (aia - pure_aia),
        "mean_incremental_alpha": (
            sum(alpha_means) / len(alpha_means) if alpha_means else 0.0
        ),
        "uses_full_old_training_data": True,
        "oracle_only": True,
    }


def build_old_class_cmpt_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
) -> Tensor:
    """Replace only old rows, keeping current-session NME means unchanged."""

    if baseline_means.ndim != 2 or transported_means.ndim != 2:
        raise ValueError("NME and transported means must be matrices")
    if baseline_means.shape != transported_means.shape:
        raise ValueError("NME and transported means must have one shape")
    boundary = int(old_class_count)
    if not 0 <= boundary <= baseline_means.shape[0]:
        raise ValueError("old_class_count is outside the prototype bank")
    result = baseline_means.detach().clone()
    if boundary > 0:
        result[:boundary] = transported_means[:boundary]
    return result


def build_old_class_interpolated_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
    alpha: float,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Interpolate NME and CMPT means for old classes only.

    ``alpha=0`` is exactly the current-memory NME bank and ``alpha=1`` is
    exactly the existing CMPT bank.  Current-session rows are never mixed.
    """

    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("prototype interpolation alpha must lie in [0, 1]")
    if epsilon <= 0.0:
        raise ValueError("prototype interpolation epsilon must be positive")
    cmpt_means = build_old_class_cmpt_means(
        baseline_means,
        transported_means,
        old_class_count,
    )
    if value == 0.0:
        return baseline_means.detach().clone()
    if value == 1.0:
        return cmpt_means

    boundary = int(old_class_count)
    result = baseline_means.detach().clone()
    if boundary == 0:
        return result
    baseline_old = F.normalize(
        baseline_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    cmpt_old = F.normalize(
        cmpt_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    mixed = (1.0 - value) * baseline_old + value * cmpt_old
    if bool((mixed.norm(dim=1) <= float(epsilon)).any()):
        raise ValueError(
            "prototype interpolation produced a zero-norm old-class row"
        )
    result[:boundary] = F.normalize(mixed, dim=1, eps=epsilon).to(
        dtype=result.dtype, device=result.device
    )
    return result


def build_old_class_adaptive_means(
    baseline_means: Tensor,
    transported_means: Tensor,
    old_class_count: int,
    old_class_alphas: Tensor | Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Interpolate each old-class row with its own test-independent alpha."""

    boundary = int(old_class_count)
    cmpt_means = build_old_class_cmpt_means(
        baseline_means, transported_means, boundary
    )
    alphas = torch.as_tensor(
        old_class_alphas,
        dtype=torch.float32,
        device=baseline_means.device,
    ).flatten()
    if alphas.numel() != boundary:
        raise ValueError("adaptive alpha count must equal old_class_count")
    if bool((~torch.isfinite(alphas)).any()) or bool(
        ((alphas < 0.0) | (alphas > 1.0)).any()
    ):
        raise ValueError("adaptive class alphas must lie in [0, 1]")
    result = baseline_means.detach().clone()
    if boundary == 0:
        return result
    baseline_old = F.normalize(
        baseline_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    cmpt_old = F.normalize(
        cmpt_means[:boundary].detach().float(), dim=1, eps=epsilon
    )
    mixed = (
        (1.0 - alphas[:, None]) * baseline_old
        + alphas[:, None] * cmpt_old
    )
    if bool((mixed.norm(dim=1) <= float(epsilon)).any()):
        raise ValueError("adaptive interpolation produced a zero-norm row")
    result[:boundary] = F.normalize(mixed, dim=1, eps=epsilon).to(
        dtype=result.dtype, device=result.device
    )
    return result


def class_geometric_oracle_alphas(
    baseline_old_means: Tensor,
    transported_old_means: Tensor,
    full_current_old_means: Tensor,
    alpha_grid: Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, dict[str, float]]:
    """Select per-class alpha by unavailable full-current prototype targets."""

    if not (
        baseline_old_means.ndim == 2
        and baseline_old_means.shape == transported_old_means.shape
        and baseline_old_means.shape == full_current_old_means.shape
    ):
        raise ValueError("geometric-oracle prototype shapes must match")
    grid = torch.as_tensor(
        tuple(float(value) for value in alpha_grid), dtype=torch.float32
    )
    if grid.numel() == 0 or bool((grid < 0.0).any()) or bool(
        (grid > 1.0).any()
    ):
        raise ValueError("geometric-oracle alpha grid must lie in [0, 1]")
    baseline = F.normalize(baseline_old_means.detach().cpu().float(), dim=1)
    transported = F.normalize(
        transported_old_means.detach().cpu().float(), dim=1
    )
    target = F.normalize(
        full_current_old_means.detach().cpu().float(), dim=1
    )
    candidates = (
        (1.0 - grid[:, None, None]) * baseline[None]
        + grid[:, None, None] * transported[None]
    )
    norms = candidates.norm(dim=2)
    normalized = F.normalize(candidates, dim=2, eps=epsilon)
    similarities = (normalized * target[None]).sum(dim=2)
    similarities = similarities.masked_fill(norms <= float(epsilon), -1.0)
    best_indices = similarities.argmax(dim=0)
    class_indices = torch.arange(baseline.shape[0])
    best_alphas = grid[best_indices]
    selected_similarity = similarities[best_indices, class_indices]
    return best_alphas, {
        "baseline_mean_cosine_distance": float(
            (1.0 - (baseline * target).sum(dim=1)).mean().item()
        ),
        "cmpt_mean_cosine_distance": float(
            (1.0 - (transported * target).sum(dim=1)).mean().item()
        ),
        "oracle_mean_cosine_distance": float(
            (1.0 - selected_similarity).mean().item()
        ),
    }


def adaptive_alpha_from_uncertainties(
    memory_uncertainty: Tensor,
    transport_uncertainty: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Return the inverse-risk weight assigned to transported CMPT means."""

    if memory_uncertainty.shape != transport_uncertainty.shape:
        raise ValueError("adaptive uncertainty tensors must have one shape")
    if epsilon <= 0.0:
        raise ValueError("adaptive alpha epsilon must be positive")
    memory = memory_uncertainty.detach().float()
    transport = transport_uncertainty.detach().float()
    if bool((~torch.isfinite(memory)).any()) or bool(
        (~torch.isfinite(transport)).any()
    ):
        raise ValueError("adaptive uncertainties must be finite")
    if bool((memory < 0.0).any()) or bool((transport < 0.0).any()):
        raise ValueError("adaptive uncertainties must be non-negative")
    return (
        (memory + float(epsilon))
        / (memory + transport + 2.0 * float(epsilon))
    ).clamp(0.0, 1.0)


def empirical_bayes_shrink_alphas(
    raw_alphas: Tensor,
    jackknife_variances: Tensor,
    session_alpha: float,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, float]:
    """Shrink noisy class alphas toward their pooled session estimate."""

    raw = raw_alphas.detach().float().flatten()
    variances = jackknife_variances.detach().float().flatten()
    if raw.shape != variances.shape or raw.numel() == 0:
        raise ValueError("raw alpha and jackknife variance shapes must match")
    if bool((variances < 0.0).any()) or bool((~torch.isfinite(variances)).any()):
        raise ValueError("jackknife variances must be finite and non-negative")
    observed = (
        raw.var(unbiased=True)
        if raw.numel() > 1
        else torch.zeros((), dtype=raw.dtype, device=raw.device)
    )
    between_class_variance = (observed - variances.mean()).clamp_min(0.0)
    credibility = between_class_variance / (
        between_class_variance + variances + float(epsilon)
    )
    center = torch.full_like(raw, float(session_alpha))
    shrunken = center + credibility * (raw - center)
    return (
        shrunken.clamp(0.0, 1.0),
        credibility,
        float(between_class_variance.item()),
    )


@dataclass(frozen=True)
class AdaptiveAlphaEstimate:
    session_alpha: float
    raw_class_alphas: Tensor
    shrunken_class_alphas: Tensor
    memory_uncertainties: Tensor
    transport_uncertainties: Tensor
    jackknife_variances: Tensor
    shrinkage_credibilities: Tensor
    between_class_variance: float
    folds: int


def _fit_affine_mapping(
    old_features: Tensor,
    current_features: Tensor,
    *,
    ridge: float,
) -> Tensor:
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, dtype=old.dtype)], dim=1
    )
    regularizer = torch.eye(design.shape[1], dtype=design.dtype) * float(ridge)
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )


def _apply_affine_mapping(features: Tensor, mapping: Tensor) -> Tensor:
    values = F.normalize(features.detach().float(), dim=1)
    design = torch.cat(
        [values, torch.ones(values.shape[0], 1, dtype=values.dtype)], dim=1
    )
    return F.normalize(design @ mapping, dim=1)


def _mean_estimation_uncertainty(values: Tensor) -> Tensor:
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("mean uncertainty requires at least two exemplars")
    normalized = F.normalize(values.detach().float(), dim=1)
    center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=1)
    return (normalized - center).square().sum() / float(count * (count - 1))


def _transport_estimation_uncertainty(residuals: Tensor) -> Tensor:
    count = int(residuals.shape[0])
    if count < 2:
        raise ValueError("transport uncertainty requires two exemplars")
    mean = residuals.mean(dim=0, keepdim=True)
    squared_bias = mean.square().sum()
    variance_of_mean = (residuals - mean).square().sum() / float(
        count * (count - 1)
    )
    return squared_bias + variance_of_mean


def _cross_fitted_affine_residuals(
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    folds: int,
    ridge: float,
) -> tuple[Tensor, int]:
    old = F.normalize(old_features.detach().cpu().float(), dim=1)
    current = F.normalize(current_features.detach().cpu().float(), dim=1)
    labels = targets.detach().cpu().long().flatten()
    if old.shape != current.shape or old.shape[0] != labels.numel():
        raise ValueError("cross-fitted affine support shapes do not match")
    expected = set(range(int(num_classes)))
    observed = set(int(value) for value in labels.unique().tolist())
    if observed != expected:
        raise ValueError("adaptive support must cover every contiguous old class")
    minimum_count = min(
        int((labels == class_index).sum().item())
        for class_index in range(int(num_classes))
    )
    effective_folds = min(int(folds), minimum_count)
    if effective_folds < 2:
        raise ValueError("adaptive alpha requires at least two folds")
    fold_ids = torch.empty_like(labels)
    for class_index in range(int(num_classes)):
        positions = torch.where(labels == class_index)[0]
        fold_ids[positions] = torch.arange(positions.numel()) % effective_folds
    residuals = torch.empty_like(current)
    for fold in range(effective_folds):
        held_out = fold_ids == fold
        mapping = _fit_affine_mapping(
            old[~held_out], current[~held_out], ridge=ridge
        )
        predicted = _apply_affine_mapping(old[held_out], mapping)
        residuals[held_out] = current[held_out] - predicted
    return residuals, effective_folds


def estimate_adaptive_alphas(
    support: PairedTransportSupport,
    *,
    num_classes: int,
    folds: int,
    ridge: float,
    epsilon: float = 1.0e-12,
) -> AdaptiveAlphaEstimate:
    """Estimate session, raw class, and EB-shrunken class CMPT weights."""

    residuals, effective_folds = _cross_fitted_affine_residuals(
        support.old_exemplar_features,
        support.current_exemplar_features,
        support.targets,
        num_classes=num_classes,
        folds=folds,
        ridge=ridge,
    )
    labels = support.targets.detach().cpu().long()
    current = support.current_exemplar_features.detach().cpu().float()
    memory_uncertainties: list[Tensor] = []
    transport_uncertainties: list[Tensor] = []
    jackknife_variances: list[Tensor] = []
    for class_index in range(int(num_classes)):
        values = current[labels == class_index]
        class_residuals = residuals[labels == class_index]
        count = int(values.shape[0])
        if count < 3:
            raise ValueError(
                "class-wise adaptive alpha requires at least three exemplars"
            )
        memory_risk = _mean_estimation_uncertainty(values)
        transport_risk = _transport_estimation_uncertainty(class_residuals)
        memory_uncertainties.append(memory_risk)
        transport_uncertainties.append(transport_risk)

        leave_one_out: list[Tensor] = []
        for omitted in range(count):
            keep = torch.arange(count) != omitted
            leave_memory = _mean_estimation_uncertainty(values[keep])
            leave_transport = _transport_estimation_uncertainty(
                class_residuals[keep]
            )
            leave_one_out.append(
                adaptive_alpha_from_uncertainties(
                    leave_memory[None],
                    leave_transport[None],
                    epsilon=epsilon,
                )[0]
            )
        jackknife = torch.stack(leave_one_out)
        jackknife_variances.append(
            float(count - 1)
            / float(count)
            * (jackknife - jackknife.mean()).square().sum()
        )

    memory = torch.stack(memory_uncertainties)
    transport = torch.stack(transport_uncertainties)
    raw = adaptive_alpha_from_uncertainties(
        memory, transport, epsilon=epsilon
    )
    session = float(
        adaptive_alpha_from_uncertainties(
            memory.mean()[None], transport.mean()[None], epsilon=epsilon
        )[0].item()
    )
    shrunken, credibility, between = empirical_bayes_shrink_alphas(
        raw,
        torch.stack(jackknife_variances),
        session,
        epsilon=epsilon,
    )
    return AdaptiveAlphaEstimate(
        session_alpha=session,
        raw_class_alphas=raw,
        shrunken_class_alphas=shrunken,
        memory_uncertainties=memory,
        transport_uncertainties=transport,
        jackknife_variances=torch.stack(jackknife_variances),
        shrinkage_credibilities=credibility,
        between_class_variance=between,
        folds=effective_folds,
    )


def _alpha_key(alpha: float) -> str:
    return format(float(alpha), ".6g")


def _alpha_statistics(values: Tensor) -> dict[str, Any]:
    flattened = values.detach().cpu().float().flatten()
    if flattened.numel() == 0:
        return {
            "alpha_mean": None,
            "alpha_min": None,
            "alpha_max": None,
            "alphas": [],
        }
    return {
        "alpha_mean": float(flattened.mean().item()),
        "alpha_min": float(flattened.min().item()),
        "alpha_max": float(flattened.max().item()),
        "alphas": [float(value) for value in flattened.tolist()],
    }


@torch.inference_mode()
def _evaluate_nme_banks(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    old_class_count: int,
    class_mean_banks: Mapping[str, Tensor],
    *,
    center_strength: float,
    horizontal_flip_query: bool,
) -> dict[str, dict[str, Any]]:
    """Evaluate several prototype banks from one query-feature extraction."""

    if not class_mean_banks:
        raise ValueError("NME bank evaluation requires at least one bank")
    if not 0.0 <= float(center_strength) <= 1.0:
        raise ValueError("NME center_strength must be in [0, 1]")

    collection = collect_features(model, loader, device)
    features = F.normalize(collection.features.to(device).float(), dim=1)
    if horizontal_flip_query:
        flipped = collect_features(
            model, loader, device, horizontal_flip=True
        )
        if not torch.equal(collection.indices, flipped.indices):
            raise RuntimeError("NME query flip rows are misaligned")
        features = F.normalize(
            features
            + F.normalize(flipped.features.to(device).float(), dim=1),
            dim=1,
        )

    targets = collection.targets.to(device).long()
    original_targets = collection.original_targets.long()
    old_mask = targets < int(old_class_count)
    new_mask = ~old_mask
    total_count = int(targets.numel())
    old_count = int(old_mask.sum().item())
    new_count = int(new_mask.sum().item())
    if total_count == 0 or new_count == 0:
        raise ValueError("cannot evaluate an empty NME split")

    results: dict[str, dict[str, Any]] = {}
    expected_shape: tuple[int, int] | None = None
    for name, class_means in class_mean_banks.items():
        means = F.normalize(class_means.to(device).float(), dim=1)
        shape = (int(means.shape[0]), int(means.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all NME prototype banks must have one shape")
        center = means.mean(dim=0, keepdim=True)
        query = features
        if center_strength > 0.0:
            means = F.normalize(
                means - float(center_strength) * center, dim=1
            )
            query = F.normalize(
                features - float(center_strength) * center, dim=1
            )
        predictions = (query @ means.T).argmax(dim=1)
        correct = predictions.eq(targets)
        old_accuracy = (
            None
            if old_count == 0
            else int(correct[old_mask].sum().item()) / old_count
        )
        new_accuracy = int(correct[new_mask].sum().item()) / new_count
        harmonic = None
        if old_accuracy is not None:
            denominator = old_accuracy + new_accuracy
            harmonic = (
                0.0
                if denominator == 0.0
                else 2.0 * old_accuracy * new_accuracy / denominator
            )
        cpu_correct = correct.cpu()
        per_class_accuracy: dict[int, float] = {}
        for class_id in original_targets.unique().tolist():
            mask = original_targets == int(class_id)
            per_class_accuracy[int(class_id)] = int(
                cpu_correct[mask].sum().item()
            ) / int(mask.sum().item())
        results[str(name)] = EvaluationResult(
            accuracy=int(correct.sum().item()) / total_count,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            harmonic_mean=harmonic,
            per_class_accuracy=per_class_accuracy,
            sample_count=total_count,
        ).to_dict()
    return results


class CMPTCheckpointEvaluator:
    """Evaluate native, current-memory NME, and CMPT on one trajectory."""

    def __init__(
        self,
        settings: CMPTExperimentSettings,
        project_root: str | Path,
        *,
        source_root: str | Path,
        max_sessions: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.project_root = Path(project_root).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        self.progress = progress or (lambda _: None)
        torch.set_num_threads(int(settings.cpu_threads))
        self.checkpoint_paths = discover_checkpoint_paths(
            settings.checkpoint_directory
        )
        self.checkpoints = [
            load_checkpoint(path, map_location="cpu")
            for path in self.checkpoint_paths
        ]
        self.audit = audit_checkpoint_trajectory(self.checkpoints, settings)
        native_specs = [
            resolve_native_classifier(checkpoint)
            for checkpoint in self.checkpoints
        ]
        self.native_classifier = native_specs[0]
        if any(
            spec != self.native_classifier for spec in native_specs[1:]
        ):
            raise ValueError(
                f"{settings.learner}: native classifier contract changes "
                "across sessions"
            )
        if max_sessions is not None:
            count = int(max_sessions)
            if count <= 0:
                raise ValueError("max_sessions must be positive")
            self.checkpoint_paths = self.checkpoint_paths[:count]
            self.checkpoints = self.checkpoints[:count]

    def validation_payload(self) -> dict[str, Any]:
        return {
            "status": "validated",
            "trajectory": self.audit.to_dict(),
            "checkpoint_directory": str(
                self.settings.checkpoint_directory
            ),
            "output_file": str(self.settings.output_file),
            "cmpt": self._cmpt_metadata(),
            "native_classifier": self.native_classifier.to_dict(),
        }

    def _cmpt_metadata(self) -> dict[str, Any]:
        return {
            "transport": self.settings.transport,
            "affine_ridge": (
                self.settings.affine_ridge
                if self.settings.transport == "affine_ridge"
                else None
            ),
            "full_introduction_prototypes": True,
            "evaluation_replacement": "transported_old_classes_only",
            "current_session_prototypes": "current_exemplar_nme",
            "prototype_horizontal_flip": (
                self.settings.prototype_horizontal_flip
            ),
            "support_horizontal_flip": self.settings.support_horizontal_flip,
            "query_horizontal_flip": self.settings.query_horizontal_flip,
            "center_strength": self.settings.center_strength,
            "strict_parity": self.settings.strict_parity,
            "parity_tolerance": self.settings.parity_tolerance,
            "cpu_threads": self.settings.cpu_threads,
            "prototype_interpolation_alphas": list(
                self.settings.prototype_interpolation_alphas
            ),
            "prototype_interpolation_formula": (
                "normalize((1-alpha)*current_memory_nme + "
                "alpha*transported_cmpt), old classes only"
            ),
            "prototype_interpolation_sweep_is_diagnostic": bool(
                self.settings.prototype_interpolation_alphas
            ),
            "adaptive_alpha": {
                "enabled": self.settings.adaptive_alpha_enabled,
                "folds": self.settings.adaptive_alpha_folds,
                "variants": [
                    "session",
                    "raw_class",
                    "shrunken_class",
                ],
                "memory_uncertainty": (
                    "directional exemplar-mean variance"
                ),
                "transport_uncertainty": (
                    "stratified out-of-fold affine residual bias plus "
                    "variance-of-mean"
                ),
                "class_shrinkage": (
                    "jackknife empirical-Bayes shrinkage toward session alpha"
                ),
                "uses_test_data": False,
            },
            "oracle_diagnostics": {
                "accuracy_enabled": self.settings.accuracy_oracle_enabled,
                "class_geometric_enabled": (
                    self.settings.class_geometric_oracle_enabled
                ),
                "alpha_grid": list(self.settings.oracle_alpha_grid),
                "oracle_only": bool(
                    self.settings.accuracy_oracle_enabled
                    or self.settings.class_geometric_oracle_enabled
                ),
                "accuracy_oracle_uses_test_labels": (
                    self.settings.accuracy_oracle_enabled
                ),
                "class_geometric_uses_full_old_training_data": (
                    self.settings.class_geometric_oracle_enabled
                ),
            },
            "component_ablation": {
                "enabled": self.settings.component_ablation_enabled,
                "class_translation_formula": (
                    "p_t = normalize(p_previous + "
                    "mean_current_exemplar - mean_old_exemplar)"
                ),
                "combined_formula": (
                    "p_t = normalize(T(p_previous) + "
                    "mean_current_exemplar - T(mean_old_exemplar))"
                ),
                "paired_old_exemplars_only": True,
                "uses_test_data": False,
            },
            "training_reused_without_changes": True,
        }

    def _trainer(self) -> UnifiedTable1Trainer:
        config = copy.deepcopy(self.checkpoints[0]["config"])
        config["device"] = self.settings.device
        config["output"] = {
            "directory": str(
                self.project_root / "outputs" / "cmpt" / "_runtime"
            ),
            "run_name": re.sub(
                r"[^a-zA-Z0-9_.-]+", "_", self.settings.learner.lower()
            ),
        }
        return UnifiedTable1Trainer(
            config,
            self.project_root,
            max_sessions=len(self.checkpoints),
        )

    def _partial_payload(
        self,
        records: list[dict[str, Any]],
        *,
        status: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 2,
            "status": status,
            "learner": self.settings.learner,
            "trajectory": self.audit.to_dict(),
            "native_classifier": self.native_classifier.to_dict(),
            "checkpoint_directory": str(
                self.settings.checkpoint_directory
            ),
            "evaluated_checkpoint_count": len(self.checkpoints),
            "cmpt": self._cmpt_metadata(),
            "cil_valid_data_access": not (
                self.settings.class_geometric_oracle_enabled
            ),
            "test_labels_used_for_selection": (
                self.settings.accuracy_oracle_enabled
            ),
            "oracle_only": bool(
                self.settings.accuracy_oracle_enabled
                or self.settings.class_geometric_oracle_enabled
            ),
            "checkpoint_weights_modified": False,
            "elapsed_seconds": elapsed_seconds,
            "git_commit": git_commit(self.project_root),
            "source_provenance": build_exploration_provenance(
                self.source_root, self.project_root / "src_explore"
            ),
            "records": records,
        }
        if records:
            payload["summary"] = _aggregate(records)
            if self.settings.prototype_interpolation_alphas:
                payload["summary"]["prototype_interpolation"] = (
                    _aggregate_interpolation(
                        records,
                        self.settings.prototype_interpolation_alphas,
                    )
                )
            if self.settings.adaptive_alpha_enabled:
                payload["summary"]["adaptive_alpha"] = (
                    _aggregate_adaptive_alpha(records)
                )
            if self.settings.accuracy_oracle_enabled:
                payload["summary"]["accuracy_oracle"] = (
                    _aggregate_accuracy_oracle(
                        records, self.settings.oracle_alpha_grid
                    )
                )
            if self.settings.class_geometric_oracle_enabled:
                payload["summary"]["class_geometric_oracle"] = (
                    _aggregate_class_geometric_oracle(records)
                )
            if self.settings.component_ablation_enabled:
                payload["summary"]["component_ablation"] = (
                    _aggregate_component_ablation(records)
                )
        return payload

    def _evaluate_native(
        self,
        model: nn.Module,
        loader: Any,
        device: torch.device,
        old_class_count: int,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.native_classifier.distinct_from_nme:
            return copy.deepcopy(dict(baseline))
        view = _NativeClassifierView(model, self.native_classifier)
        return evaluate_classifier(
            view,
            loader,
            device,
            old_class_count,
        ).to_dict()

    def _upgrade_existing_native_output(
        self,
        output: Path,
    ) -> dict[str, Any]:
        """Add native metrics to a completed schema-v1 result atomically."""

        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise FileExistsError(
                f"CMPT output exists but is not complete: {output}; "
                "pass --force to replace it"
            )
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != len(
            self.checkpoints
        ):
            raise ValueError(
                "existing CMPT result does not match the checkpoint count"
            )
        if any("native" in record for record in records):
            raise FileExistsError(
                f"CMPT output already contains native results: {output}; "
                "pass --force to replace it"
            )

        trainer = (
            self._trainer()
            if self.native_classifier.distinct_from_nme
            else None
        )
        started = time.perf_counter()
        for record, checkpoint in zip(records, self.checkpoints):
            session_id = int(checkpoint["session_id"])
            if int(record.get("session_id", -1)) != session_id:
                raise ValueError(
                    "existing CMPT record order does not match checkpoints"
                )
            baseline = record.get("baseline")
            if not isinstance(baseline, Mapping):
                raise ValueError("existing CMPT record lacks its NME result")
            if self.native_classifier.distinct_from_nme:
                assert trainer is not None
                old_class_count = trainer.protocol.session(session_id).start
                model = _load_model(trainer, checkpoint, session_id)
                test_dataset = trainer.data.cumulative_test_dataset(session_id)
                test_loader = trainer._loader(
                    test_dataset,
                    shuffle=False,
                    session_id=session_id + 11000,
                )
                native = self._evaluate_native(
                    model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    baseline,
                )
                del model
            else:
                native = copy.deepcopy(dict(baseline))
            record["native"] = native
            self.progress(
                f"{self.settings.learner} S{session_id}: "
                f"Native={100.0 * float(native['accuracy']):.3f} "
                "(existing NME/CMPT preserved)"
            )

        payload["schema_version"] = 2
        payload["native_classifier"] = self.native_classifier.to_dict()
        payload["summary"] = _aggregate(records)
        payload["native_evaluation"] = {
            "added_to_existing_result": True,
            "training_reused_without_changes": True,
            "checkpoint_weights_modified": False,
            "test_labels_used_for_selection": False,
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": git_commit(self.project_root),
            "source_provenance": build_exploration_provenance(
                self.source_root, self.project_root / "src_explore"
            ),
        }
        temporary = output.with_suffix(output.suffix + ".native-upgrade.tmp")
        dump_json(payload, temporary)
        temporary.replace(output)
        return payload

    def run(
        self,
        *,
        output_file: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        output = (
            self.settings.output_file
            if output_file is None
            else Path(output_file).expanduser().resolve()
        )
        if output.exists() and not force:
            return self._upgrade_existing_native_output(output)
        trainer = self._trainer()
        records: list[dict[str, Any]] = []
        transported: Tensor | None = None
        class_translation_bank: Tensor | None = None
        combined_bank: Tensor | None = None
        previous_checkpoint: Mapping[str, Any] | None = None
        started = time.perf_counter()

        for checkpoint_path, checkpoint in zip(
            self.checkpoint_paths, self.checkpoints
        ):
            session_started = time.perf_counter()
            session_id = int(checkpoint["session_id"])
            seen = trainer.protocol.session(session_id).stop
            old_class_count = trainer.protocol.session(session_id).start
            current_model = _load_model(trainer, checkpoint, session_id)
            trainer.model = current_model
            trainer.memory = ExemplarMemory.from_state_dict(
                checkpoint["memory"]
            )
            paired_support: PairedTransportSupport | None = None

            if session_id == 0:
                transported = _full_introduction_prototypes(
                    trainer,
                    current_model,
                    session_id,
                    horizontal_flip=(
                        self.settings.prototype_horizontal_flip
                    ),
                )
                transport_diagnostics = {
                    "initialized": True,
                    "support_count": 0,
                    "fit_residual": None,
                }
                if self.settings.component_ablation_enabled:
                    class_translation_bank = transported.detach().clone()
                    combined_bank = transported.detach().clone()
                    component_diagnostics = {
                        "initialized": True,
                        "class_translation_mean_norm": None,
                        "class_translation_max_norm": None,
                        "class_residual_mean_norm": None,
                        "class_residual_max_norm": None,
                    }
            else:
                if previous_checkpoint is None or transported is None:
                    raise RuntimeError("CMPT transition lacks previous state")
                previous_model = _load_model(
                    trainer, previous_checkpoint, session_id - 1
                )
                trainer.memory = ExemplarMemory.from_state_dict(
                    previous_checkpoint["memory"]
                )
                paired_support = _paired_support_features(
                    trainer,
                    previous_model,
                    current_model,
                    session_id - 1,
                    horizontal_flip=(
                        self.settings.support_horizontal_flip
                    ),
                )
                if self.settings.transport == "rigid_procrustes":
                    transported_old, rotation, translation, residual = (
                        rigid_procrustes_transport(
                            transported,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                        )
                    )
                    transport_diagnostics = {
                        "initialized": False,
                        "support_count": paired_support.fit_support_count,
                        "fit_residual": residual,
                        "rotation_orthogonality_error": float(
                            (
                                rotation.T @ rotation
                                - torch.eye(rotation.shape[0])
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        "translation_norm": float(translation.norm().item()),
                    }
                else:
                    transported_old, mapping, residual = affine_ridge_transport(
                        transported,
                        paired_support.old_fit_features,
                        paired_support.current_fit_features,
                        ridge=self.settings.affine_ridge,
                    )
                    linear = mapping[:-1]
                    bias = mapping[-1]
                    identity = torch.eye(
                        linear.shape[0],
                        device=linear.device,
                        dtype=linear.dtype,
                    )
                    transport_diagnostics = {
                        "initialized": False,
                        "support_count": paired_support.fit_support_count,
                        "fit_residual": residual,
                        "affine_ridge": self.settings.affine_ridge,
                        "linear_identity_deviation": float(
                            (linear - identity).norm().item()
                            / linear.shape[0] ** 0.5
                        ),
                        "bias_norm": float(bias.norm().item()),
                    }
                if self.settings.component_ablation_enabled:
                    if (
                        class_translation_bank is None
                        or combined_bank is None
                    ):
                        raise RuntimeError(
                            "component ablation transition lacks prototype banks"
                        )
                    if self.settings.transport != "affine_ridge":
                        raise RuntimeError(
                            "component ablation transition lacks affine mapping"
                        )
                    class_translation_old, class_translations = (
                        classwise_translation_transport(
                            class_translation_bank,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                            paired_support.fit_targets,
                        )
                    )
                    combined_old, class_residuals = (
                        affine_class_residual_transport(
                            combined_bank,
                            mapping,
                            paired_support.old_fit_features,
                            paired_support.current_fit_features,
                            paired_support.fit_targets,
                        )
                    )
                    translation_norms = class_translations.norm(dim=1)
                    residual_norms = class_residuals.norm(dim=1)
                    component_diagnostics = {
                        "initialized": False,
                        "class_translation_mean_norm": float(
                            translation_norms.mean().item()
                        ),
                        "class_translation_max_norm": float(
                            translation_norms.max().item()
                        ),
                        "class_residual_mean_norm": float(
                            residual_norms.mean().item()
                        ),
                        "class_residual_max_norm": float(
                            residual_norms.max().item()
                        ),
                    }
                trainer.model = current_model
                new_full = _full_introduction_prototypes(
                    trainer,
                    current_model,
                    session_id,
                    horizontal_flip=(
                        self.settings.prototype_horizontal_flip
                    ),
                )
                transported = torch.cat(
                    [transported_old.cpu(), new_full.cpu()], dim=0
                )
                if self.settings.component_ablation_enabled:
                    class_translation_bank = torch.cat(
                        [class_translation_old.cpu(), new_full.cpu()], dim=0
                    )
                    combined_bank = torch.cat(
                        [combined_old.cpu(), new_full.cpu()], dim=0
                    )
                del previous_model

            if transported is None or transported.shape[0] != seen:
                raise RuntimeError(
                    f"CMPT prototype bank has invalid shape at S{session_id}: "
                    f"{None if transported is None else tuple(transported.shape)}"
                )
            baseline_means = checkpoint["class_means"].detach().cpu()
            # The original exploration that established the CMPT gain changes
            # only old-class prototypes.  Current-session classes keep exactly
            # the same exemplar NME means as the control; their full-data
            # introduction prototypes are stored only for transport after they
            # become old.  Consequently S0 is identical by construction.
            cmpt_means = build_old_class_cmpt_means(
                baseline_means,
                transported,
                old_class_count,
            )
            class_translation_means: Tensor | None = None
            combined_means: Tensor | None = None
            if self.settings.component_ablation_enabled:
                if (
                    class_translation_bank is None
                    or combined_bank is None
                ):
                    raise RuntimeError(
                        "component ablation prototype banks are missing"
                    )
                if not (
                    class_translation_bank.shape == transported.shape
                    and combined_bank.shape == transported.shape
                ):
                    raise RuntimeError(
                        "component ablation prototype-bank shapes differ"
                    )
                class_translation_means = build_old_class_cmpt_means(
                    baseline_means,
                    class_translation_bank,
                    old_class_count,
                )
                combined_means = build_old_class_cmpt_means(
                    baseline_means,
                    combined_bank,
                    old_class_count,
                )
            adaptive_estimate: AdaptiveAlphaEstimate | None = None
            adaptive_banks: dict[str, Tensor] = {}
            if self.settings.adaptive_alpha_enabled:
                if session_id == 0:
                    adaptive_banks = {
                        "adaptive_session": baseline_means,
                        "adaptive_raw_class": baseline_means,
                        "adaptive_shrunken_class": baseline_means,
                    }
                else:
                    if paired_support is None:
                        raise RuntimeError(
                            "adaptive alpha transition lacks paired support"
                        )
                    if self.settings.transport != "affine_ridge":
                        raise ValueError(
                            "adaptive alpha currently requires affine_ridge"
                        )
                    adaptive_estimate = estimate_adaptive_alphas(
                        paired_support,
                        num_classes=old_class_count,
                        folds=self.settings.adaptive_alpha_folds,
                        ridge=self.settings.affine_ridge,
                    )
                    session_alphas = torch.full(
                        (old_class_count,),
                        adaptive_estimate.session_alpha,
                    )
                    adaptive_banks = {
                        "adaptive_session": build_old_class_adaptive_means(
                            baseline_means,
                            transported,
                            old_class_count,
                            session_alphas,
                        ),
                        "adaptive_raw_class": (
                            build_old_class_adaptive_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                adaptive_estimate.raw_class_alphas,
                            )
                        ),
                        "adaptive_shrunken_class": (
                            build_old_class_adaptive_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                adaptive_estimate.shrunken_class_alphas,
                            )
                        ),
                    }
            geometric_alphas = torch.empty(0)
            geometric_diagnostics: dict[str, Any] | None = None
            geometric_bank: Tensor | None = None
            if self.settings.class_geometric_oracle_enabled:
                if session_id == 0:
                    geometric_bank = baseline_means
                    geometric_diagnostics = {
                        "initialized": True,
                        "full_old_training_image_count": 0,
                        "baseline_mean_cosine_distance": None,
                        "cmpt_mean_cosine_distance": None,
                        "oracle_mean_cosine_distance": None,
                    }
                else:
                    full_current_old, full_image_count = (
                        _full_current_old_prototypes(
                            trainer,
                            current_model,
                            session_id,
                            horizontal_flip=(
                                self.settings.prototype_horizontal_flip
                            ),
                        )
                    )
                    geometric_alphas, distances = (
                        class_geometric_oracle_alphas(
                            baseline_means[:old_class_count],
                            cmpt_means[:old_class_count],
                            full_current_old,
                            self.settings.oracle_alpha_grid,
                        )
                    )
                    geometric_bank = build_old_class_adaptive_means(
                        baseline_means,
                        transported,
                        old_class_count,
                        geometric_alphas,
                    )
                    geometric_diagnostics = {
                        "initialized": False,
                        "full_old_training_image_count": full_image_count,
                        **distances,
                    }
            test_dataset = trainer.data.cumulative_test_dataset(session_id)
            test_loader = trainer._loader(
                test_dataset,
                shuffle=False,
                session_id=session_id + 11000,
            )
            prototype_interpolation: dict[str, dict[str, Any]] | None = None
            adaptive_metrics: dict[str, dict[str, Any]] | None = None
            component_metrics: dict[str, dict[str, Any]] | None = None
            if (
                self.settings.prototype_interpolation_alphas
                or self.settings.adaptive_alpha_enabled
                or self.settings.class_geometric_oracle_enabled
                or self.settings.component_ablation_enabled
            ):
                evaluation_banks: dict[str, Tensor] = {
                    "__baseline": baseline_means,
                    "__cmpt": cmpt_means,
                }
                if self.settings.component_ablation_enabled:
                    if (
                        class_translation_means is None
                        or combined_means is None
                    ):
                        raise RuntimeError(
                            "component ablation evaluation banks are missing"
                        )
                    evaluation_banks.update(
                        {
                            "class_translation": class_translation_means,
                            "combined_cmpt": combined_means,
                        }
                    )
                evaluation_banks.update(adaptive_banks)
                if geometric_bank is not None:
                    evaluation_banks["class_geometric_oracle"] = (
                        geometric_bank
                    )
                evaluation_banks.update(
                    {
                        _alpha_key(alpha): (
                            build_old_class_interpolated_means(
                                baseline_means,
                                transported,
                                old_class_count,
                                alpha,
                            )
                        )
                        for alpha in (
                            self.settings.prototype_interpolation_alphas
                        )
                    }
                )
                evaluated_banks = _evaluate_nme_banks(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    evaluation_banks,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                )
                baseline = copy.deepcopy(evaluated_banks["__baseline"])
                cmpt = copy.deepcopy(evaluated_banks["__cmpt"])
                if self.settings.prototype_interpolation_alphas:
                    prototype_interpolation = {
                        _alpha_key(alpha): copy.deepcopy(
                            evaluated_banks[_alpha_key(alpha)]
                        )
                        for alpha in (
                            self.settings.prototype_interpolation_alphas
                        )
                    }
                if self.settings.adaptive_alpha_enabled:
                    adaptive_metrics = {
                        "session": copy.deepcopy(
                            evaluated_banks["adaptive_session"]
                        ),
                        "raw_class": copy.deepcopy(
                            evaluated_banks["adaptive_raw_class"]
                        ),
                        "shrunken_class": copy.deepcopy(
                            evaluated_banks["adaptive_shrunken_class"]
                        ),
                    }
                if self.settings.component_ablation_enabled:
                    component_metrics = {
                        "class_translation": copy.deepcopy(
                            evaluated_banks["class_translation"]
                        ),
                        "combined_cmpt": copy.deepcopy(
                            evaluated_banks["combined_cmpt"]
                        ),
                    }
                geometric_metrics = (
                    copy.deepcopy(
                        evaluated_banks["class_geometric_oracle"]
                    )
                    if self.settings.class_geometric_oracle_enabled
                    else None
                )
            else:
                geometric_metrics = None
                baseline = evaluate_nme(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    baseline_means,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                ).to_dict()
                cmpt = evaluate_nme(
                    current_model,
                    test_loader,
                    trainer.device,
                    old_class_count,
                    cmpt_means,
                    center_strength=self.settings.center_strength,
                    horizontal_flip_query=(
                        self.settings.query_horizontal_flip
                    ),
                ).to_dict()
            native = self._evaluate_native(
                current_model,
                test_loader,
                trainer.device,
                old_class_count,
                baseline,
            )
            stored = _checkpoint_session_metric(checkpoint, session_id)
            parity_error = float(baseline["accuracy"]) - float(
                stored["accuracy"]
            )
            if (
                self.settings.strict_parity
                and abs(parity_error) > self.settings.parity_tolerance
            ):
                raise RuntimeError(
                    f"{self.settings.learner} S{session_id} NME parity failed: "
                    f"recomputed={baseline['accuracy']:.8f}, "
                    f"stored={float(stored['accuracy']):.8f}, "
                    f"error={parity_error:+.3e}"
                )

            adaptive_alpha_record: dict[str, Any] | None = None
            if self.settings.adaptive_alpha_enabled:
                if adaptive_metrics is None:
                    raise RuntimeError("adaptive alpha metrics are missing")
                if adaptive_estimate is None:
                    empty = torch.empty(0)
                    session_stats = _alpha_statistics(empty)
                    raw_stats = _alpha_statistics(empty)
                    shrunken_stats = _alpha_statistics(empty)
                    diagnostics: dict[str, Any] = {
                        "initialized": True,
                        "folds": None,
                        "old_exemplar_count": 0,
                        "memory_uncertainties": [],
                        "transport_uncertainties": [],
                        "jackknife_variances": [],
                        "shrinkage_credibilities": [],
                        "between_class_variance": None,
                    }
                else:
                    session_stats = _alpha_statistics(
                        torch.full(
                            (old_class_count,),
                            adaptive_estimate.session_alpha,
                        )
                    )
                    raw_stats = _alpha_statistics(
                        adaptive_estimate.raw_class_alphas
                    )
                    shrunken_stats = _alpha_statistics(
                        adaptive_estimate.shrunken_class_alphas
                    )
                    diagnostics = {
                        "initialized": False,
                        "folds": adaptive_estimate.folds,
                        "old_exemplar_count": (
                            paired_support.exemplar_count
                            if paired_support is not None
                            else 0
                        ),
                        "memory_uncertainties": [
                            float(value)
                            for value in adaptive_estimate.memory_uncertainties.tolist()
                        ],
                        "transport_uncertainties": [
                            float(value)
                            for value in adaptive_estimate.transport_uncertainties.tolist()
                        ],
                        "jackknife_variances": [
                            float(value)
                            for value in adaptive_estimate.jackknife_variances.tolist()
                        ],
                        "shrinkage_credibilities": [
                            float(value)
                            for value in adaptive_estimate.shrinkage_credibilities.tolist()
                        ],
                        "between_class_variance": (
                            adaptive_estimate.between_class_variance
                        ),
                    }
                adaptive_alpha_record = {
                    "session": {
                        **session_stats,
                        "metrics": adaptive_metrics["session"],
                    },
                    "raw_class": {
                        **raw_stats,
                        "metrics": adaptive_metrics["raw_class"],
                    },
                    "shrunken_class": {
                        **shrunken_stats,
                        "metrics": adaptive_metrics["shrunken_class"],
                    },
                    "diagnostics": diagnostics,
                }

            class_geometric_record: dict[str, Any] | None = None
            if self.settings.class_geometric_oracle_enabled:
                if geometric_metrics is None or geometric_diagnostics is None:
                    raise RuntimeError("class-geometric oracle result is missing")
                class_geometric_record = {
                    **_alpha_statistics(geometric_alphas),
                    "metrics": geometric_metrics,
                    "diagnostics": geometric_diagnostics,
                    "oracle_only": True,
                    "uses_full_old_training_data": session_id > 0,
                }

            record = {
                "checkpoint": str(checkpoint_path),
                "session_id": session_id,
                "seen_classes": seen,
                "native": native,
                "baseline": baseline,
                "cmpt": cmpt,
                "delta_accuracy": float(cmpt["accuracy"])
                - float(baseline["accuracy"]),
                "stored_nme_accuracy": float(stored["accuracy"]),
                "parity_error": parity_error,
                "transport_diagnostics": transport_diagnostics,
                "session_elapsed_seconds": time.perf_counter()
                - session_started,
            }
            if self.settings.component_ablation_enabled:
                if component_metrics is None:
                    raise RuntimeError(
                        "component ablation metrics are missing"
                    )
                record["class_translation"] = component_metrics[
                    "class_translation"
                ]
                record["combined_cmpt"] = component_metrics[
                    "combined_cmpt"
                ]
                record["component_ablation"] = {
                    "class_translation_delta_vs_nme": float(
                        component_metrics["class_translation"]["accuracy"]
                    )
                    - float(baseline["accuracy"]),
                    "combined_delta_vs_nme": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(baseline["accuracy"]),
                    "combined_delta_vs_global": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(cmpt["accuracy"]),
                    "combined_delta_vs_class": float(
                        component_metrics["combined_cmpt"]["accuracy"]
                    )
                    - float(
                        component_metrics["class_translation"]["accuracy"]
                    ),
                    "diagnostics": component_diagnostics,
                }
            if prototype_interpolation is not None:
                record["prototype_interpolation"] = (
                    prototype_interpolation
                )
            if adaptive_alpha_record is not None:
                record["adaptive_alpha"] = adaptive_alpha_record
            if class_geometric_record is not None:
                record["class_geometric_oracle"] = class_geometric_record
            records.append(record)
            elapsed = time.perf_counter() - started
            dump_json(
                self._partial_payload(
                    records, status="running", elapsed_seconds=elapsed
                ),
                output,
            )
            adaptive_progress = ""
            if adaptive_alpha_record is not None:
                adaptive_progress = (
                    ", Adapt-S="
                    f"{100.0 * float(adaptive_alpha_record['session']['metrics']['accuracy']):.3f}"
                    ", Adapt-C="
                    f"{100.0 * float(adaptive_alpha_record['raw_class']['metrics']['accuracy']):.3f}"
                    ", Adapt-CS="
                    f"{100.0 * float(adaptive_alpha_record['shrunken_class']['metrics']['accuracy']):.3f}"
                )
            oracle_progress = ""
            if class_geometric_record is not None:
                oracle_progress = (
                    ", Geo-Oracle="
                    f"{100.0 * float(class_geometric_record['metrics']['accuracy']):.3f}"
                )
            component_progress = ""
            if self.settings.component_ablation_enabled:
                assert component_metrics is not None
                component_progress = (
                    ", Class-T="
                    f"{100.0 * float(component_metrics['class_translation']['accuracy']):.3f}"
                    ", Combined="
                    f"{100.0 * float(component_metrics['combined_cmpt']['accuracy']):.3f}"
                )
            self.progress(
                f"{self.settings.learner} S{session_id}: "
                f"Native={100.0 * float(native['accuracy']):.3f}, "
                f"NME={100.0 * float(baseline['accuracy']):.3f}, "
                f"CMPT={100.0 * float(cmpt['accuracy']):.3f}, "
                f"delta={100.0 * record['delta_accuracy']:+.3f} pp"
                f"{adaptive_progress}"
                f"{oracle_progress}"
                f"{component_progress}"
            )
            previous_checkpoint = checkpoint
            del current_model

        payload = self._partial_payload(
            records,
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
        )
        dump_json(payload, output)
        return payload
