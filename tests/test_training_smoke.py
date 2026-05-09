"""Smoke test for the minimal training entrypoint components."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import PairedImageDataset
from src.losses import BasicEnhancementLoss
from src.models import build_trainable_enhancer
from src.trainers import BasicTrainer, TrainerConfig


class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.data_root = cls.project_root / "data"
        cls.temp_ckpt_root = cls.project_root / "checkpoints" / "_smoke_test"

    def setUp(self) -> None:
        shutil.rmtree(self.temp_ckpt_root, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_ckpt_root, ignore_errors=True)

    def test_one_epoch_training_and_checkpoint(self) -> None:
        train_dataset = PairedImageDataset(self.data_root, split="train", image_size=(256, 256), patch_training=False)
        val_dataset = PairedImageDataset(self.data_root, split="val", image_size=(256, 256), patch_training=False)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

        model = build_trainable_enhancer(base_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = BasicEnhancementLoss()

        trainer = BasicTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainerConfig(
                device=torch.device("cpu"),
                amp=False,
                checkpoint_dir=self.temp_ckpt_root,
                checkpoint_name="smoke_latest.pth",
                val_every=1,
            ),
        )

        result = trainer.train(epochs=1)
        checkpoint_path = result["checkpoint"]
        self.assertTrue(checkpoint_path.exists())

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.assertIn("state_dict", checkpoint)
        self.assertIn("optimizer", checkpoint)


if __name__ == "__main__":
    unittest.main()
