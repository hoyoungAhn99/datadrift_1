from __future__ import annotations

import torch

from sacil.memory import (
    ExemplarMemory,
    herding_select,
    icarl_herding_select,
)


def test_herding_is_deterministic_unique_and_bounded() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    indices = [10, 11, 12, 13]
    first = herding_select(features, indices, 3)
    second = herding_select(features, indices, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first).issubset(indices)


def test_memory_roundtrip_preserves_class_indices() -> None:
    memory = ExemplarMemory(2)
    memory.set_class_indices(5, [10, 11])
    memory.set_class_indices(2, [3])
    restored = ExemplarMemory.from_state_dict(memory.state_dict())
    assert restored.exemplars_per_class == 2
    assert restored.class_ids == (2, 5)
    assert restored.indices_for_class(5) == (10, 11)
    assert restored.all_indices((5, 2)) == [10, 11, 3]


def test_icarl_herding_is_deterministic_unique_and_bounded() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    indices = [10, 11, 12, 13]
    first = icarl_herding_select(features, indices, 3)
    second = icarl_herding_select(features, indices, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first).issubset(indices)
