from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMPT_SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(CMPT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CMPT_SOURCE_ROOT))

from sacil.cmpt import (  # noqa: E402
    CMPTExperimentSettings,
    adaptive_alpha_from_uncertainties,
    build_old_class_adaptive_means,
    build_old_class_interpolated_means,
    class_geometric_oracle_alphas,
    empirical_bayes_shrink_alphas,
)
from sacil.methods.prototype_transport import (  # noqa: E402
    affine_class_residual_transport,
    apply_affine_mapping,
    classwise_translation_transport,
)
from sacil.cmpt.drift_grouping import (  # noqa: E402
    direction_similarity_correlation,
)


def test_interpolation_endpoints_and_current_rows_are_exact() -> None:
    baseline = F.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
        ),
        dim=1,
    )
    transported = F.normalize(
        torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        dim=1,
    )

    alpha_zero = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=0.0
    )
    alpha_one = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=1.0
    )
    alpha_half = build_old_class_interpolated_means(
        baseline, transported, old_class_count=2, alpha=0.5
    )

    assert torch.equal(alpha_zero, baseline)
    assert torch.equal(alpha_one[:2], transported[:2])
    assert torch.equal(alpha_one[2:], baseline[2:])
    assert torch.equal(alpha_half[2:], baseline[2:])
    expected = F.normalize(baseline[:2] + transported[:2], dim=1)
    assert torch.allclose(alpha_half[:2], expected)


def test_interpolation_rejects_invalid_alpha() -> None:
    means = torch.eye(2)
    with pytest.raises(ValueError, match="alpha"):
        build_old_class_interpolated_means(
            means, means, old_class_count=1, alpha=1.01
        )


def test_settings_require_sweep_endpoints() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "prototype_interpolation_alphas": [0.25, 0.5, 0.75],
        },
    }
    with pytest.raises(ValueError, match="alpha=0 and alpha=1"):
        CMPTExperimentSettings.from_config(config, PROJECT_ROOT)


def test_uncertainty_ratio_weights_the_more_reliable_estimator() -> None:
    memory = torch.tensor([4.0, 1.0, 1.0])
    transport = torch.tensor([1.0, 4.0, 1.0])
    alpha = adaptive_alpha_from_uncertainties(memory, transport)
    assert torch.allclose(alpha, torch.tensor([0.8, 0.2, 0.5]))


def test_class_adaptive_interpolation_keeps_new_rows() -> None:
    baseline = torch.eye(4)
    transported = torch.flip(torch.eye(4), dims=(1,))
    result = build_old_class_adaptive_means(
        baseline,
        transported,
        old_class_count=2,
        old_class_alphas=torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(result[0], baseline[0])
    assert torch.equal(result[1], transported[1])
    assert torch.equal(result[2:], baseline[2:])


def test_empirical_bayes_shrinkage_respects_class_reliability() -> None:
    raw = torch.tensor([0.2, 0.8])
    jackknife = torch.tensor([1.0e-5, 0.2])
    shrunken, credibility, between = empirical_bayes_shrink_alphas(
        raw, jackknife, session_alpha=0.5
    )
    assert between > 0.0
    assert credibility[0] > credibility[1]
    assert abs(float(shrunken[1]) - 0.5) < abs(float(raw[1]) - 0.5)


def test_settings_parse_adaptive_alpha() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {"adaptive_alpha": {"enabled": True, "folds": 4}},
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.adaptive_alpha_enabled
    assert settings.adaptive_alpha_folds == 4


def test_class_geometric_oracle_selects_closest_segment_point() -> None:
    baseline = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    transported = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    target = torch.stack(
        [
            F.normalize(torch.tensor([1.0, 1.0]), dim=0),
            torch.tensor([0.0, 1.0]),
        ]
    )
    alphas, diagnostics = class_geometric_oracle_alphas(
        baseline,
        transported,
        target,
        alpha_grid=[0.0, 0.5, 1.0],
    )
    assert torch.equal(alphas, torch.tensor([0.5, 1.0]))
    assert diagnostics["oracle_mean_cosine_distance"] == pytest.approx(0.0)


def test_settings_parse_accuracy_oracle_grid() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "prototype_interpolation_alphas": [0.0, 0.5, 1.0],
            "oracle_diagnostics": {
                "accuracy": True,
                "alpha_grid": [0.0, 0.5, 1.0],
            },
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.accuracy_oracle_enabled
    assert settings.oracle_alpha_grid == (0.0, 0.5, 1.0)


def test_classwise_translation_uses_matched_class_mean_drifts() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    current = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])

    transported, translations = classwise_translation_transport(
        prototypes, old, current, targets
    )

    assert torch.allclose(
        translations,
        torch.tensor([[-1.0, 1.0], [-1.0, -1.0]]),
    )
    assert torch.allclose(transported, current[::2])


def test_affine_class_residual_is_zero_for_exact_global_map() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    current = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    mapping = torch.tensor(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    combined, residuals = affine_class_residual_transport(
        prototypes, mapping, old, current, targets
    )
    global_only = apply_affine_mapping(prototypes, mapping)

    assert torch.allclose(residuals, torch.zeros_like(residuals))
    assert torch.allclose(combined, global_only)


def test_affine_class_residual_adds_observed_class_offset() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = prototypes.repeat_interleave(2, dim=0)
    current = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    identity_affine = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    combined, residuals = affine_class_residual_transport(
        prototypes, identity_affine, old, current, targets
    )
    expected_current_means = F.normalize(current[::2], dim=1)

    assert bool((residuals.norm(dim=1) > 0.0).all())
    assert torch.allclose(combined, expected_current_means)


def test_settings_parse_component_ablation() -> None:
    config = {
        "experiment": {
            "learner": "test",
            "checkpoints": "checkpoints",
            "output": "result.json",
            "expected_checkpoint_method": "icarl",
        },
        "cmpt": {
            "transport": "affine_ridge",
            "component_ablation": {"enabled": True},
        },
    }
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    assert settings.component_ablation_enabled


def test_drift_similarity_correlation_detects_matched_geometry() -> None:
    prototypes = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        dim=1,
    )
    result = direction_similarity_correlation(
        prototypes,
        prototypes,
        permutations=100,
        seed=7,
    )
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["permutation_p_positive"] <= 0.05
