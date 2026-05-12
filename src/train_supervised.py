"""Stage-3 supervised training entrypoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.datasets import SyntheticQuadDataset
from src.losses import SupervisedLossStack, SupervisedLossWeights
from src.models import build_physical_guided_enhancer
from src.trainers import SupervisedTrainer, SupervisedTrainerConfig
from src.utils.config import load_config
from src.utils.io import resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-3 supervised training.")
    parser.add_argument("--config", default="configs/train/supervised_base.yaml", help="Training config path.")
    return parser.parse_args(argv)


def resolve_device(device_name: str) -> torch.device:
    lowered = device_name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(lowered)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    collated = {
        "input": default_collate([sample["input"] for sample in batch]),
        "target": default_collate([sample["target"] for sample in batch]),
        "transmission": default_collate([sample["transmission"] for sample in batch]),
        "airlight": default_collate([sample["airlight"] for sample in batch]),
    }
    collated["metadata"] = [sample.get("metadata") for sample in batch]
    collated["meta"] = [sample.get("meta") for sample in batch]
    return collated


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_project_path(args.config, project_root))

    set_seed(int(config.get("runtime", {}).get("seed", 0)))
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    output_cfg = config["output"]

    data_root = resolve_project_path(data_cfg["root"], project_root)
    image_size = tuple(int(value) for value in data_cfg.get("image_size", [256, 256]))
    train_dataset = SyntheticQuadDataset(data_root, split=str(data_cfg.get("train_split", "train")), image_size=image_size)
    val_dataset = SyntheticQuadDataset(data_root, split=str(data_cfg.get("val_split", "val")), image_size=image_size)

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=synthetic_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=synthetic_collate_fn,
    )

    device = resolve_device(str(train_cfg.get("device", "cpu")))
    model = build_physical_guided_enhancer(
        base_channels=int(model_cfg.get("base_channels", 32)),
        t_min=float(model_cfg.get("t_min", 0.05)),
        refinement_blocks=int(model_cfg.get("refinement_blocks", 2)),
    ).to(device)

    checkpoint = str(model_cfg.get("checkpoint", "")).strip()
    if checkpoint:
        checkpoint_path = resolve_project_path(checkpoint, project_root)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state.get("state_dict", state), strict=False)

    criterion = SupervisedLossStack(
        weights=SupervisedLossWeights(
            recon=float(loss_cfg.get("recon", 1.0)),
            ssim=float(loss_cfg.get("ssim", 0.5)),
            edge=float(loss_cfg.get("edge", 0.2)),
            transmission=float(loss_cfg.get("transmission", 0.5)),
            airlight=float(loss_cfg.get("airlight", 0.2)),
            perceptual=float(loss_cfg.get("perceptual", 0.1)),
        ),
        use_stub_perceptual=bool(loss_cfg.get("use_stub_perceptual", True)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))

    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=SupervisedTrainerConfig(
            device=device,
            amp=bool(train_cfg.get("amp", False)),
            checkpoint_dir=resolve_project_path(output_cfg.get("checkpoint_dir", "./checkpoints"), project_root),
            checkpoint_name=str(output_cfg.get("checkpoint_name", "physical_guided_supervised_latest.pth")),
            val_every=int(output_cfg.get("val_every", 1)),
        ),
    )

    result = trainer.train(epochs=int(train_cfg.get("epochs", 1)))
    print("[INFO] Supervised training complete.")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples  : {len(val_dataset)}")
    print(f"[INFO] Checkpoint   : {result['checkpoint']}")


if __name__ == "__main__":
    main()
