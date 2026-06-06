from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.density import gaussian_bump, gaussian_bump_integrals, gaussian_logpdf
from core.hierarchy_inference import hierarchical_node_probabilities


class TinyHierarchy:
    def __init__(self):
        self.id_node_list = ["root", "parent", "child_a", "child_b"]
        self.node_ancestors = {
            "root": [],
            "parent": [0],
            "child_a": [0, 1],
            "child_b": [0, 1],
        }
        self.parent2children = {
            "root": ["parent"],
            "parent": ["child_a", "child_b"],
        }


def build_density_payload():
    means = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    variances = torch.tensor(
        [
            [9.0, 9.0],
            [4.0, 1.0],
            [0.1, 0.1],
            [0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    return {
        "means": means,
        "variances": variances,
        "covariance_type": "diag",
        "counts": torch.tensor([100, 100, 50, 50], dtype=torch.long),
    }


def main():
    hierarchy = TinyHierarchy()
    density = build_density_payload()
    features = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    logpdf = gaussian_logpdf(
        features,
        density["means"],
        density["variances"],
        covariance_type="diag",
    )
    assert logpdf.shape == (3, 4)

    bumps = gaussian_bump(
        features,
        density["means"],
        density["variances"],
        covariance_type="diag",
        node_indices=[2, 3],
    )
    assert bumps.shape == (3, 2)
    assert torch.all(bumps > 0)
    assert torch.all(bumps <= 1.0)
    assert torch.isclose(bumps[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(bumps[2, 1], torch.tensor(1.0), atol=1e-6)

    integrals = gaussian_bump_integrals(
        1,
        [2, 3],
        density["means"],
        density["variances"],
        covariance_type="diag",
    )
    assert integrals.shape == (2,)
    assert torch.all(integrals > 0)
    assert torch.all(integrals <= 1.0)

    final_probs, debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "normalize_ood_pdf": True,
            "eps": 1e-12,
        },
    )
    assert final_probs.shape == (3, 4)
    assert torch.allclose(final_probs.sum(dim=1), torch.ones(3), atol=1e-6)

    parent_local = debug["local_info"]["parent"]
    local_sum = parent_local["child_probs"].sum(dim=1) + parent_local["ood_prob"]
    assert torch.allclose(local_sum, torch.ones(3), atol=1e-6)
    assert parent_local["ood_prob"][1] > parent_local["ood_prob"][0]
    assert parent_local["ood_prob"][1] > parent_local["ood_prob"][2]

    print("CGM smoke checks passed")


if __name__ == "__main__":
    main()
