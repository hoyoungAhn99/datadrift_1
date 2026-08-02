from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sacil.engine.table1_trainer import (
    BalancedClassBatchSampler,
    StandaloneTable1Trainer,
)
from sacil.methods import casper_spectral_loss
from sacil.methods import (
    pycil_finetune_loss,
    pycil_icarl_kd_loss,
    unified_method_contract,
)
from sacil.models import (
    AFCMultiProxyClassifier,
    CREATEIncrementalNet,
    CSCCTIncrementalNet,
    FGPIncrementalNet,
    FGPResNet32,
    ScaleShiftConv2d,
)
from sacil.config import load_config_tree


def test_afc_classifier_matches_scaled_negative_cosine_distance() -> None:
    classifier = AFCMultiProxyClassifier(
        3, [2], proxies_per_class=1, distance_scale=3.0
    )
    features = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        classifier._weights[0].copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    assert torch.allclose(
        classifier(features), torch.tensor([[0.0, -18.0]]), atol=1e-6
    )


def test_fgp_backbone_and_incremental_classifier_shapes() -> None:
    backbone = FGPResNet32()
    features = backbone(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, 64)
    model = FGPIncrementalNet(4)
    old = model.classifier.weight.detach().clone()
    model.expand_classes(6)
    assert torch.equal(model.classifier.weight[:4], old)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 6)


def test_create_native_model_expands_without_changing_old_autoencoders() -> None:
    model = CREATEIncrementalNet(3, hidden_layers=(), latent_features=4)
    old = {
        key: value.detach().clone()
        for key, value in model.classifier.class_autoencoders[0].state_dict().items()
    }
    model.expand_classes(5)
    for key, value in old.items():
        assert torch.equal(
            model.classifier.class_autoencoders[0].state_dict()[key], value
        )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_cscct_expansion_builds_scale_shift_free_dual_branch() -> None:
    model = CSCCTIncrementalNet(4)
    images = torch.randn(2, 3, 32, 32)
    features = model.extract_features(images)
    imprint = F.normalize(torch.randn(2, 64), dim=1)
    model.expand_classes(imprint)
    assert model.second is not None
    assert any(isinstance(module, ScaleShiftConv2d) for module in model.first.modules())
    assert torch.allclose(features, model.extract_features(images), atol=1e-5)
    loss = model(images).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.fusion)


def test_casper_balanced_sampler_uses_exact_class_count() -> None:
    labels = [label for label in range(8) for _ in range(3)]
    sampler = BalancedClassBatchSampler(
        labels,
        batch_size=20,
        classes_per_batch=5,
        batches=4,
        seed=7,
    )
    for batch in sampler:
        assert len(batch) == 20
        assert len({labels[position] for position in batch}) == 5


def test_casper_partial_eigensolver_has_finite_gradient() -> None:
    features = torch.randn(64, 8, requires_grad=True)
    loss = casper_spectral_loss(
        features, num_classes=5, k=10, solver="partial"
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_standalone_runner_import_graph_does_not_reference_pycil() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "train_table1.py",
        root / "src" / "sacil" / "engine" / "table1_trainer.py",
    )
    imported: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
    assert not any("pycil" in module.lower() for module in imported)


def test_unified_reference_paths_are_metadata_only() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "train_table1.py",
        root / "src" / "sacil" / "engine" / "table1_trainer.py",
    )
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        assert "subprocess" not in names
        assert "importlib" not in names
    contract = unified_method_contract("create")
    assert contract.reference_only is True
    assert contract.implementation_module == "sacil.methods.create"
    assert contract.evaluation_classifier == "native_reconstruction_error"


def test_pycil_finetune_uses_only_the_new_head_slice() -> None:
    logits = torch.tensor(
        [[100.0, 90.0, 1.0, 2.0], [80.0, 70.0, 3.0, 1.0]],
        requires_grad=True,
    )
    targets = torch.tensor([3, 2])
    expected = F.cross_entropy(logits[:, 2:], targets - 2)
    actual = pycil_finetune_loss(logits, targets, known_classes=2)
    assert torch.allclose(actual, expected)


def test_unified_icarl_matches_pycil_softmax_kd_without_t_squared() -> None:
    current = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    reference = torch.tensor([[1.0, -1.0], [1.5, 0.5]])
    temperature = 2.0
    expected = -(
        F.softmax(reference / temperature, dim=1)
        * F.log_softmax(current / temperature, dim=1)
    ).sum() / current.shape[0]
    actual = pycil_icarl_kd_loss(
        current, reference, temperature=temperature
    )
    assert torch.allclose(actual, expected)


def test_standalone_runner_refuses_to_append_to_existing_run(tmp_path) -> None:
    trainer = StandaloneTable1Trainer.__new__(StandaloneTable1Trainer)
    trainer.run_dir = tmp_path
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir()
    (tmp_path / "sessions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new --run-name"):
        trainer.run()


def test_recursive_config_extends_preserves_method_and_debug_overrides(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    method = tmp_path / "method.yaml"
    validation = tmp_path / "validation.yaml"
    base.write_text("training:\n  base:\n    epochs: 200\n", encoding="utf-8")
    method.write_text(
        "extends: base.yaml\nmethod:\n  name: replay\n",
        encoding="utf-8",
    )
    validation.write_text(
        "extends: method.yaml\n"
        "training:\n  base:\n    epochs: 2\n"
        "debug:\n  max_sessions: 2\n",
        encoding="utf-8",
    )
    config = load_config_tree(validation)
    assert config["method"]["name"] == "replay"
    assert config["training"]["base"]["epochs"] == 2
    assert config["debug"]["max_sessions"] == 2
