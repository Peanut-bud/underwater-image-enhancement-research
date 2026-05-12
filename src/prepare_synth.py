"""Stage-1 synthesis entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.physics.synthesis import run_synthesis_pipeline
from src.utils.config import load_config
from src.utils.io import resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 synthetic quadruplet generation.")
    parser.add_argument("--config", default="configs/data/synth_build.yaml", help="Path to the synthesis config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_project_path(args.config, project_root))
    summary = run_synthesis_pipeline(config, project_root)
    print("[INFO] Stage-1 synthesis complete.")
    print(f"[INFO] Clean images : {summary.clean_images}")
    print(f"[INFO] Raw images   : {summary.raw_images}")
    print(f"[INFO] Generated   : total={summary.generated_total}")
    print(f"[INFO] Synthetic   : train={summary.synthetic_train} val={summary.synthetic_val} test={summary.synthetic_test}")
    if summary.manifest_path:
        print(f"[INFO] Manifest    : {summary.manifest_path}")


if __name__ == "__main__":
    main()
