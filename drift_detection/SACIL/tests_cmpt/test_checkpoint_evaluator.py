from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sacil.cmpt import (
    CMPTExperimentSettings,
    audit_checkpoint_trajectory,
    build_old_class_cmpt_means,
    discover_checkpoint_paths,
)


def _checkpoint(
    session_id: int,
    memories: dict[int, list[int]],
    *,
    casper: bool = False,
) -> dict:
    seen = len(memories)
    return {
        "session_id": session_id,
        "protocol_id": "test-protocol",
        "method_contract": {
            "name": "icarl",
            "evaluation_classifier": "nme",
        },
        "config": {"evaluation": {"horizontal_flip": True}},
        "class_means": torch.nn.functional.normalize(
            torch.randn(seen, 4), dim=1
        ),
        "memory": {
            "exemplars_per_class": 2,
            "indices": {
                str(class_id): indices
                for class_id, indices in memories.items()
            },
        },
        "casper_options": {"enabled": casper},
        "metrics": {
            "records": [
                {"session_id": value, "accuracy": 0.5}
                for value in range(session_id + 1)
            ]
        },
    }


def _settings(tmp_path: Path, *, casper: bool = False):
    return CMPTExperimentSettings(
        learner="test",
        checkpoint_directory=tmp_path,
        output_file=tmp_path / "result.json",
        expected_checkpoint_method="icarl",
        expected_sessions=2,
        expected_exemplars_per_class=2,
        expected_casper_enabled=casper,
        device="cpu",
    )


def test_discover_checkpoint_paths_requires_contiguous_sequence(
    tmp_path: Path,
) -> None:
    (tmp_path / "session_00.pt").touch()
    (tmp_path / "session_01.pt").touch()
    assert [path.name for path in discover_checkpoint_paths(tmp_path)] == [
        "session_00.pt",
        "session_01.pt",
    ]
    (tmp_path / "session_01.pt").unlink()
    (tmp_path / "session_02.pt").touch()
    with pytest.raises(ValueError, match="contiguous"):
        discover_checkpoint_paths(tmp_path)


def test_audit_accepts_stable_nme_trajectory(tmp_path: Path) -> None:
    base = {10: [1, 2], 11: [3, 4]}
    current = {**base, 12: [5, 6]}
    audit = audit_checkpoint_trajectory(
        [_checkpoint(0, base), _checkpoint(1, current)],
        _settings(tmp_path),
    )
    assert audit.feature_dim == 4
    assert audit.old_exemplar_identities_stable is True
    assert audit.final_memory_classes == 3


def test_audit_rejects_changed_old_exemplars(tmp_path: Path) -> None:
    base = {10: [1, 2], 11: [3, 4]}
    current = {10: [1, 9], 11: [3, 4], 12: [5, 6]}
    with pytest.raises(ValueError, match="identities change"):
        audit_checkpoint_trajectory(
            [_checkpoint(0, base), _checkpoint(1, current)],
            _settings(tmp_path),
        )


def test_audit_distinguishes_casper_trajectory(tmp_path: Path) -> None:
    base = {10: [1, 2], 11: [3, 4]}
    current = {**base, 12: [5, 6]}
    audit = audit_checkpoint_trajectory(
        [
            _checkpoint(0, base, casper=True),
            _checkpoint(1, current, casper=True),
        ],
        _settings(tmp_path, casper=True),
    )
    assert audit.casper_enabled is True


def test_s0_cmpt_means_are_exactly_the_nme_control() -> None:
    baseline = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    transported = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    result = build_old_class_cmpt_means(baseline, transported, 0)
    torch.testing.assert_close(result, baseline, rtol=0.0, atol=0.0)


def test_cmpt_replaces_old_rows_only() -> None:
    baseline = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    transported = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    result = build_old_class_cmpt_means(baseline, transported, 3)
    torch.testing.assert_close(result[:3], transported[:3])
    torch.testing.assert_close(result[3:], baseline[3:])


def test_affine_transport_settings_are_explicit(tmp_path: Path) -> None:
    config = {
        "device": "cpu",
        "experiment": {
            "learner": "affine-test",
            "checkpoints": str(tmp_path),
            "output": str(tmp_path / "affine.json"),
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "affine_ridge": 0.025,
            "full_introduction_prototypes": True,
            "replace_old_classes_only": True,
        },
    }
    settings = CMPTExperimentSettings.from_config(config, tmp_path)
    assert settings.transport == "affine_ridge"
    assert settings.affine_ridge == 0.025


def test_affine_transport_rejects_nonpositive_ridge(tmp_path: Path) -> None:
    config = {
        "experiment": {
            "learner": "affine-test",
            "checkpoints": str(tmp_path),
            "output": str(tmp_path / "affine.json"),
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "affine_ridge": 0.0,
        },
    }
    with pytest.raises(ValueError, match="affine_ridge must be positive"):
        CMPTExperimentSettings.from_config(config, tmp_path)
