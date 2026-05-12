"""Evaluation entrypoint for paired and no-reference protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets import RealUnpairedDataset, SyntheticQuadDataset
from src.models import build_physical_guided_enhancer
from src.utils.config import load_config
from src.utils.io import resolve_project_path, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired or no-reference evaluation.")
    parser.add_argument("--config", default="configs/eval/paired_base.yaml", help="Evaluation config path.")
    return parser.parse_args(argv)


def resolve_device(device_name: str) -> torch.device:
    lowered = device_name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(lowered)


def evaluate_paired(model, dataset, device: torch.device) -> dict[str, float]:
    mae_total = 0.0
    mse_total = 0.0
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["input"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            enhanced = model(inputs)["enhanced"]
            mae_total += float((enhanced - target).abs().mean().cpu())
            mse_total += float(((enhanced - target) ** 2).mean().cpu())
    count = max(len(dataset), 1)
    return {"mae": mae_total / count, "mse": mse_total / count}


def evaluate_noref(model, dataset, device: torch.device) -> dict[str, float]:
    brightness = 0.0
    contrast = 0.0
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["input"].unsqueeze(0).to(device)
            enhanced = model(inputs)["enhanced"].cpu()
            brightness += float(enhanced.mean())
            contrast += float(enhanced.std())
    count = max(len(dataset), 1)
    return {"brightness_mean": brightness / count, "contrast_std": contrast / count}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_project_path(args.config, project_root))

    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    runtime_cfg = config.get("runtime", {})
    mode = str(model_cfg.get("mode", "paired")).lower()

    device = resolve_device(str(runtime_cfg.get("device", "cpu")))
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
    model.eval()

    data_root = resolve_project_path(data_cfg["root"], project_root)
    image_size = tuple(int(value) for value in data_cfg.get("image_size", [256, 256]))
    split = str(data_cfg.get("split", "val"))
    if mode == "paired":
        dataset = SyntheticQuadDataset(data_root, split=split, image_size=image_size)
        report = evaluate_paired(model, dataset, device)
    else:
        dataset = RealUnpairedDataset(data_root, split=split, image_size=image_size)
        report = evaluate_noref(model, dataset, device)

    report_path = resolve_project_path(output_cfg["report_path"], project_root)
    write_json(report_path, report)
    print(f"[INFO] Evaluation mode: {mode}")
    print(f"[INFO] Samples        : {len(dataset)}")
    print(f"[INFO] Report         : {report_path}")


if __name__ == "__main__":
    main()
