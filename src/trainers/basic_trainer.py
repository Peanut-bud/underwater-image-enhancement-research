"""Minimal trainer for the stage-one trainable backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.io import ensure_dir


@dataclass
class TrainerConfig:
    device: torch.device
    amp: bool
    checkpoint_dir: Path
    checkpoint_name: str
    val_every: int = 1


class BasicTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        train_loader,
        val_loader,
        config: TrainerConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.use_amp = bool(config.amp and config.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self, epochs: int) -> Dict[str, Any]:
        history: Dict[str, Any] = {"train": [], "val": []}
        latest_train = {}
        latest_val = {}

        for epoch in range(1, epochs + 1):
            latest_train = self.train_one_epoch(epoch)
            history["train"].append(latest_train)

            if epoch % self.config.val_every == 0:
                latest_val = self.validate(epoch)
                history["val"].append(latest_val)

            self.save_checkpoint(epoch)

        return {
            "history": history,
            "last_train": latest_train,
            "last_val": latest_val,
            "checkpoint": self.config.checkpoint_dir / self.config.checkpoint_name,
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "edge": 0.0}
        batches = 0

        for batch in self.train_loader:
            inputs = batch["input"].to(self.config.device)
            targets = batch["target"].to(self.config.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.config.device.type, enabled=self.use_amp):
                preds = self.model(inputs)
                losses = self.criterion(preds, targets)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in running:
                running[key] += float(losses[key].detach().cpu())
            batches += 1

        metrics = {key: value / max(batches, 1) for key, value in running.items()}
        print(f"[TRAIN] epoch={epoch} total={metrics['total']:.6f} l1={metrics['l1']:.6f} ssim={metrics['ssim']:.6f} edge={metrics['edge']:.6f}")
        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "edge": 0.0}
        batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.config.device)
                targets = batch["target"].to(self.config.device)
                preds = self.model(inputs)
                losses = self.criterion(preds, targets)
                for key in running:
                    running[key] += float(losses[key].detach().cpu())
                batches += 1

        metrics = {key: value / max(batches, 1) for key, value in running.items()}
        print(f"[VAL]   epoch={epoch} total={metrics['total']:.6f} l1={metrics['l1']:.6f} ssim={metrics['ssim']:.6f} edge={metrics['edge']:.6f}")
        return metrics

    def save_checkpoint(self, epoch: int) -> Path:
        ensure_dir(self.config.checkpoint_dir)
        checkpoint_path = self.config.checkpoint_dir / self.config.checkpoint_name
        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path
