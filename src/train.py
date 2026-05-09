"""Stage-one minimal training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import PairedImageDataset
from src.losses import BasicEnhancementLoss, BasicLossWeights
from src.models import build_trainable_enhancer
from src.trainers import BasicTrainer, TrainerConfig
from src.utils.config import load_config
from src.utils.io import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stage-one training smoke test.")
    parser.add_argument(
        "--config",
        default="configs/train_base.yaml",
        help="Path to the training config file.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    lowered = device_name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(lowered)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_project_path(args.config, project_root)
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]
    loss_cfg = config.get("loss", {})

    data_root = resolve_project_path(data_cfg.get("root", "./data"), project_root)
    image_size_cfg = data_cfg.get("image_size", [512, 512])
    image_size = (int(image_size_cfg[0]), int(image_size_cfg[1]))
    num_workers = int(data_cfg.get("num_workers", 0))
    patch_training = bool(data_cfg.get("patch_training", False))

    train_dataset = PairedImageDataset(data_root, split="train", image_size=image_size, patch_training=patch_training)
    val_dataset = PairedImageDataset(data_root, split="val", image_size=image_size, patch_training=False)

    batch_size = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    base_channels = int(model_cfg.get("base_channels", 32))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    model = build_trainable_enhancer(base_channels=base_channels).to(device)

    criterion = BasicEnhancementLoss(
        BasicLossWeights(
            l1=float(loss_cfg.get("l1", 1.0)),
            ssim=float(loss_cfg.get("ssim", 1.0)),
            edge=float(loss_cfg.get("edge", 0.5)),
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))

    trainer = BasicTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            device=device,
            amp=bool(train_cfg.get("amp", True)),
            checkpoint_dir=resolve_project_path(output_cfg.get("checkpoint_dir", "./checkpoints"), project_root),
            checkpoint_name=str(output_cfg.get("checkpoint_name", "stage1_trainable_latest.pth")),
            val_every=int(output_cfg.get("val_every", 1)),
        ),
    )

    epochs = int(train_cfg.get("epochs", 1))
    result = trainer.train(epochs=epochs)

    print("[INFO] Training complete.")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples  : {len(val_dataset)}")
    print(f"[INFO] Checkpoint   : {result['checkpoint']}")


if __name__ == "__main__":
    main()
