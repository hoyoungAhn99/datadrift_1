from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from negzerohoc.checkpointing import (
    load_idea3_checkpoint,
    save_idea3_checkpoint,
)


def save_test_checkpoint(path: Path, *, training_state=None):
    return save_idea3_checkpoint(
        path,
        stage="test",
        dataset="test-dataset",
        clip_model="test-clip",
        hierarchy="test-hierarchy.json",
        id_split="test-split.csv",
        prompt_config={},
        positive_state_dict={"prompt": torch.tensor([1.0])},
        metrics={"score": 0.5},
        args={"epochs": 2},
        training_state=training_state,
    )


class AtomicCheckpointTest(unittest.TestCase):
    def test_round_trip_includes_resumable_training_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"
            training_state = {
                "epoch": 7,
                "optimizer_state_dict": {"state": {}, "param_groups": []},
            }

            save_test_checkpoint(path, training_state=training_state)
            payload = load_idea3_checkpoint(path)

            self.assertEqual(payload["training_state"]["epoch"], 7)
            self.assertFalse(path.with_name(".last.pt.tmp").exists())

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"
            path.write_bytes(b"previous-checkpoint")

            with mock.patch(
                "negzerohoc.checkpointing.torch.save",
                side_effect=RuntimeError("simulated power failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "power failure"):
                    save_test_checkpoint(path)

            self.assertEqual(path.read_bytes(), b"previous-checkpoint")
            self.assertFalse(path.with_name(".last.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
