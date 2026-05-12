"""Stage-4 adaptation entrypoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import RealUnpairedDataset
from src.losses import AdaptationLossStack, AdaptationLossWeights
from src.models import build_physical_guided_enhancer
from src.trainers import AdaptationTrainer, AdaptationTrainerConfig
from src.utils.config import load_config
from src.utils.io import resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-4 unsupervised adaptation.")
    parser.add_argument("--config", default="configs/train/adapt_base.yaml", help="Adaptation config path.")
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


def freeze_refinement(model: torch.nn.Module) -> None:
    for parameter in model.refinement_net.parameters():
        parameter.requires_grad = False


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
    dataset = RealUnpairedDataset(data_root, split=str(data_cfg.get("train_split", "train")), image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
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

    if bool(model_cfg.get("freeze_refinement", True)):
        freeze_refinement(model)

    criterion = AdaptationLossStack(
        AdaptationLossWeights(
            color_constancy=float(loss_cfg.get("color_constancy", 1.0)),
            exposure_balance=float(loss_cfg.get("exposure_balance", 1.0)),
            transmission_smooth=float(loss_cfg.get("transmission_smooth", 0.5)),
            identity_regularization=float(loss_cfg.get("identity_regularization", 0.2)),
        )
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(train_cfg.get("learning_rate", 5e-4)))

    trainer = AdaptationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=loader,
        config=AdaptationTrainerConfig(
            device=device,
            amp=bool(train_cfg.get("amp", False)),
            checkpoint_dir=resolve_project_path(output_cfg.get("checkpoint_dir", "./checkpoints"), project_root),
            checkpoint_name=str(output_cfg.get("checkpoint_name", "physical_guided_adapt_latest.pth")),
        ),
    )

    result = trainer.train(epochs=int(train_cfg.get("epochs", 1)))
    print("[INFO] Adaptation training complete.")
    print(f"[INFO] Train samples: {len(dataset)}")
    print(f"[INFO] Checkpoint   : {result['checkpoint']}")


if __name__ == "__main__":
    main()
