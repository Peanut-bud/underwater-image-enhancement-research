"""Trainer for stage-4 unsupervised adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.io import ensure_dir


@dataclass
class AdaptationTrainerConfig:
    device: torch.device
    amp: bool
    checkpoint_dir: Path
    checkpoint_name: str


class AdaptationTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, config: AdaptationTrainerConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.config = config
        self.use_amp = bool(config.amp and config.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self, epochs: int) -> Dict[str, Any]:
        history = []
        latest_metrics: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            latest_metrics = self.train_one_epoch(epoch)
            history.append(latest_metrics)
            self.save_checkpoint(epoch)
        return {
            "history": history,
            "last_train": latest_metrics,
            "checkpoint": self.config.checkpoint_dir / self.config.checkpoint_name,
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        batches = 0
        for batch in self.train_loader:
            inputs = batch["input"].to(self.config.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.config.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                losses = self.criterion(outputs, {"input": inputs})
            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())
            batches += 1

        metrics = {key: value / max(batches, 1) for key, value in running.items()}
        print(f"[ADAPT-TRAIN] epoch={epoch} total={metrics['total']:.6f}")
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
