"""Stage-1 synthesis planning entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.physics.synthesis import build_synthesis_summary
from src.utils.config import load_config
from src.utils.io import resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the stage-1 synthesis scaffold.")
    parser.add_argument("--config", default="configs/data/synth_build.yaml", help="Path to the synthesis config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_project_path(args.config, project_root))

    data_cfg = config["data"]
    clean_root = resolve_project_path(data_cfg["clean_source_root"], project_root)
    raw_root = resolve_project_path(data_cfg["raw_field_root"], project_root)
    synthetic_root = resolve_project_path(data_cfg["synthetic_root"], project_root)

    summary = build_synthesis_summary(clean_root, raw_root, synthetic_root)
    print("[INFO] Stage-1 synthesis scaffold validated.")
    print(f"[INFO] Clean images : {summary.clean_images}")
    print(f"[INFO] Raw images   : {summary.raw_images}")
    print(f"[INFO] Synthetic   : train={summary.synthetic_train} val={summary.synthetic_val} test={summary.synthetic_test}")


if __name__ == "__main__":
    main()
