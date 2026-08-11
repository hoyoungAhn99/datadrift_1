from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sacil.anchors import compute_prototypes
from sacil.engine.checkpoint import load_checkpoint
from sacil.engine.evaluator import evaluate_nme
from sacil.engine.table1_trainer import UnifiedTable1Trainer
from sacil.features import collect_features
from sacil.memory import ExemplarMemory
from sacil.methods.prototype_transport import (
    affine_ridge_transport,
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

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        project_root: str | Path,
    ) -> "CMPTExperimentSettings":
        root = Path(project_root).expanduser().resolve()
        experiment = _required_mapping(config, "experiment")
        cmpt = _required_mapping(config, "cmpt")

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


def _paired_support_features(
    trainer: UnifiedTable1Trainer,
    previous_model: nn.Module,
    current_model: nn.Module,
    previous_session_id: int,
    *,
    horizontal_flip: bool,
) -> tuple[Tensor, Tensor, int]:
    loader = trainer._memory_loader(previous_session_id, augment=False)
    old = collect_features(previous_model, loader, trainer.device)
    current = collect_features(current_model, loader, trainer.device)
    if not torch.equal(old.indices, current.indices):
        raise RuntimeError("old/current transport support rows are misaligned")
    old_features = old.features
    current_features = current.features
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
    return old_features, current_features, int(old_features.shape[0])


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    baseline = [float(record["baseline"]["accuracy"]) for record in records]
    cmpt = [float(record["cmpt"]["accuracy"]) for record in records]
    parity = [abs(float(record["parity_error"])) for record in records]
    incremental_baseline = baseline[1:] if len(baseline) > 1 else baseline
    incremental_cmpt = cmpt[1:] if len(cmpt) > 1 else cmpt
    return {
        "baseline_aia_percent": 100.0 * sum(baseline) / len(baseline),
        "cmpt_aia_percent": 100.0 * sum(cmpt) / len(cmpt),
        "aia_delta_percent_points": 100.0
        * (sum(cmpt) - sum(baseline))
        / len(baseline),
        "baseline_incremental_aia_percent": 100.0
        * sum(incremental_baseline)
        / len(incremental_baseline),
        "cmpt_incremental_aia_percent": 100.0
        * sum(incremental_cmpt)
        / len(incremental_cmpt),
        "baseline_final_percent": 100.0 * baseline[-1],
        "cmpt_final_percent": 100.0 * cmpt[-1],
        "final_delta_percent_points": 100.0 * (cmpt[-1] - baseline[-1]),
        "max_parity_error_percent_points": 100.0 * max(parity),
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


class CMPTCheckpointEvaluator:
    """Evaluate current-memory NME and CMPT-NCM on one frozen trajectory."""

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
        self.checkpoint_paths = discover_checkpoint_paths(
            settings.checkpoint_directory
        )
        self.checkpoints = [
            load_checkpoint(path, map_location="cpu")
            for path in self.checkpoint_paths
        ]
        self.audit = audit_checkpoint_trajectory(self.checkpoints, settings)
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
            "schema_version": 1,
            "status": status,
            "learner": self.settings.learner,
            "trajectory": self.audit.to_dict(),
            "checkpoint_directory": str(
                self.settings.checkpoint_directory
            ),
            "evaluated_checkpoint_count": len(self.checkpoints),
            "cmpt": self._cmpt_metadata(),
            "cil_valid_data_access": True,
            "test_labels_used_for_selection": False,
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
            raise FileExistsError(
                f"CMPT output already exists: {output}; pass --force to replace"
            )
        trainer = self._trainer()
        records: list[dict[str, Any]] = []
        transported: Tensor | None = None
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
            else:
                if previous_checkpoint is None or transported is None:
                    raise RuntimeError("CMPT transition lacks previous state")
                previous_model = _load_model(
                    trainer, previous_checkpoint, session_id - 1
                )
                trainer.memory = ExemplarMemory.from_state_dict(
                    previous_checkpoint["memory"]
                )
                old_features, current_features, support_count = (
                    _paired_support_features(
                        trainer,
                        previous_model,
                        current_model,
                        session_id - 1,
                        horizontal_flip=(
                            self.settings.support_horizontal_flip
                        ),
                    )
                )
                if self.settings.transport == "rigid_procrustes":
                    transported_old, rotation, translation, residual = (
                        rigid_procrustes_transport(
                            transported,
                            old_features,
                            current_features,
                        )
                    )
                    transport_diagnostics = {
                        "initialized": False,
                        "support_count": support_count,
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
                        old_features,
                        current_features,
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
                        "support_count": support_count,
                        "fit_residual": residual,
                        "affine_ridge": self.settings.affine_ridge,
                        "linear_identity_deviation": float(
                            (linear - identity).norm().item()
                            / linear.shape[0] ** 0.5
                        ),
                        "bias_norm": float(bias.norm().item()),
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
            test_dataset = trainer.data.cumulative_test_dataset(session_id)
            test_loader = trainer._loader(
                test_dataset,
                shuffle=False,
                session_id=session_id + 11000,
            )
            baseline = evaluate_nme(
                current_model,
                test_loader,
                trainer.device,
                old_class_count,
                baseline_means,
                center_strength=self.settings.center_strength,
                horizontal_flip_query=self.settings.query_horizontal_flip,
            ).to_dict()
            cmpt = evaluate_nme(
                current_model,
                test_loader,
                trainer.device,
                old_class_count,
                cmpt_means,
                center_strength=self.settings.center_strength,
                horizontal_flip_query=self.settings.query_horizontal_flip,
            ).to_dict()
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

            record = {
                "checkpoint": str(checkpoint_path),
                "session_id": session_id,
                "seen_classes": seen,
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
            records.append(record)
            elapsed = time.perf_counter() - started
            dump_json(
                self._partial_payload(
                    records, status="running", elapsed_seconds=elapsed
                ),
                output,
            )
            self.progress(
                f"{self.settings.learner} S{session_id}: "
                f"NME={100.0 * float(baseline['accuracy']):.3f}, "
                f"CMPT={100.0 * float(cmpt['accuracy']):.3f}, "
                f"delta={100.0 * record['delta_accuracy']:+.3f} pp"
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
