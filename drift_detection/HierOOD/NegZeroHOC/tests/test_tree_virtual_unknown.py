from types import SimpleNamespace

import torch

from negzerohoc.metric_terminal import build_metric_terminal_specs
from negzerohoc.tree_virtual_unknown import (
    augmented_unknown_distance_matrix,
    decoder_aligned_hierarchical_id_loss,
    leaf_unknown_distance_matrix,
    predict_tree_complement_terminals,
    tree_complement_terminal_scores,
    tree_ordinal_prompt_loss,
    virtual_sibling_shell_loss,
)


def toy_hierarchy():
    nodes = ["root", "a", "a1", "a2", "b", "b1", "b2"]
    index = {node: i for i, node in enumerate(nodes)}
    ancestors = {
        "root": [],
        "a": [index["root"]],
        "a1": [index["root"], index["a"]],
        "a2": [index["root"], index["a"]],
        "b": [index["root"]],
        "b1": [index["root"], index["b"]],
        "b2": [index["root"], index["b"]],
    }
    return SimpleNamespace(
        id_node_list=nodes,
        node_ancestors=ancestors,
        parent2children={
            "root": ["a", "b"],
            "a": ["a1", "a2"],
            "b": ["b1", "b2"],
        },
    )


def test_augmented_tree_distances_treat_unknown_as_virtual_child():
    hierarchy = toy_hierarchy()
    distances = augmented_unknown_distance_matrix(
        hierarchy,
        ["a"],
        ["a", "a1", "a2", "b", "b1"],
    )
    assert distances.tolist() == [[1.0, 2.0, 2.0, 3.0, 4.0]]
    leaf_distances = leaf_unknown_distance_matrix(
        hierarchy,
        ["a1", "b1"],
        ["a", "b"],
    )
    assert leaf_distances.tolist() == [[2.0, 4.0], [4.0, 2.0]]


def test_tree_ordinal_loss_prefers_tree_consistent_similarity_order():
    positive_nodes = torch.eye(3)
    distances = torch.tensor([[1.0, 2.0, 3.0]])
    ordered = torch.tensor([[[0.9, 0.4, 0.1]]])
    reversed_order = torch.tensor([[[0.1, 0.4, 0.9]]])
    ordered_loss, _ = tree_ordinal_prompt_loss(
        ordered,
        positive_nodes,
        distances,
        margin_per_step=0.05,
        temperature=0.05,
    )
    reversed_loss, _ = tree_ordinal_prompt_loss(
        reversed_order,
        positive_nodes,
        distances,
        margin_per_step=0.05,
        temperature=0.05,
    )
    assert ordered_loss < reversed_loss


def test_virtual_sibling_shell_penalizes_collapsed_unknown_bank():
    children = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    diverse = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    collapsed = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    diverse_loss, _ = virtual_sibling_shell_loss(
        {"a": diverse},
        {"a": children},
    )
    collapsed_loss, _ = virtual_sibling_shell_loss(
        {"a": collapsed},
        {"a": children},
    )
    assert diverse_loss < collapsed_loss


def decoder_fixture():
    hierarchy = toy_hierarchy()
    positive = {
        ("root", "a"): torch.tensor([1.0, 0.0, 0.0]),
        ("root", "b"): torch.tensor([0.0, 1.0, 0.0]),
        ("a", "a1"): torch.tensor([0.9, 0.0, 0.1]),
        ("a", "a2"): torch.tensor([0.9, 0.0, -0.1]),
        ("b", "b1"): torch.tensor([0.0, 0.9, 0.1]),
        ("b", "b2"): torch.tensor([0.0, 0.9, -0.1]),
    }
    unknown = {
        "a": torch.tensor([[0.6, 0.0, 0.8]]),
        "b": torch.tensor([[0.0, 0.6, 0.8]]),
    }
    specs = build_metric_terminal_specs(
        hierarchy,
        unknown_parents=["a", "b"],
    )
    return hierarchy, positive, unknown, specs


def test_tree_complement_decoder_can_select_leaf_and_parent_unknown():
    hierarchy, positive, unknown, specs = decoder_fixture()
    images = torch.tensor([
        [0.9, 0.0, 0.1],
        [0.6, 0.0, 0.8],
    ])
    output = predict_tree_complement_terminals(
        images,
        hierarchy,
        positive,
        specs,
        unknown,
        terminal_weight=1.0,
        complement_weight=0.5,
    )
    predicted = [
        hierarchy.id_node_list[int(index)]
        for index in output["preds"].tolist()
    ]
    assert predicted[0] == "a1"
    assert predicted[1] == "a"


def test_decoder_aligned_id_loss_backpropagates_to_unknown():
    hierarchy, positive, unknown, specs = decoder_fixture()
    unknown_parameter = torch.nn.Parameter(unknown["a"].clone())
    unknown_with_grad = {
        "a": unknown_parameter,
        "b": unknown["b"],
    }
    images = torch.tensor([[0.9, 0.0, 0.1]])
    scores = tree_complement_terminal_scores(
        images,
        hierarchy,
        positive,
        specs,
        unknown_with_grad,
        terminal_weight=1.0,
        complement_weight=0.5,
    )
    distances = leaf_unknown_distance_matrix(
        hierarchy,
        ["a1"],
        ["a", "b"],
    )
    loss, stats = decoder_aligned_hierarchical_id_loss(
        scores,
        ["a1"],
        distances,
    )
    loss.backward()
    assert unknown_parameter.grad is not None
    assert torch.isfinite(unknown_parameter.grad).all()
    assert stats["decoder_id_loss"] >= 0.0
