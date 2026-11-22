"""DVC wrapper for the Hakaluki v0.4 dataset build stage.

DVC's per-parameter command templating is version- and platform-sensitive,
especially for a boolean flag. This wrapper removes that fragility: DVC invokes a
single, stable command, and the wrapper reads and validates ``params.yaml``
itself, then calls the builder's :func:`main` with explicit arguments.

Usage (from the repository root, as declared in dvc.yaml):

    python backend/scripts/dvc_build_dataset.py

The wrapper:
* locates ``params.yaml`` relative to this file (no working-directory guessing);
* validates the ``build`` section through the single validated configuration
  path (``app.ml.build_config.build_runtime_config``) — invalid values abort
  before any GEE request, directory creation, or file write;
* forwards every parameter explicitly to the builder;
* passes ``--rebuild-stale`` only when ``build.rebuild_stale`` is true.

This keeps DVC tracking all five parameters as stage dependencies while giving
deterministic, OS-independent expansion for both boolean values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from app.ml.build_config import build_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"


def load_build_params(params_path: Path) -> dict:
    """Load and return the validated ``build`` section of params.yaml."""
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")
    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("build"), dict):
        raise ValueError(f"{params_path} must contain a 'build' mapping")
    build = data["build"]
    # Validate through the single configuration path; raises on invalid input.
    build_runtime_config(
        sar_unit=build.get("sar_unit"),
        s2_cloud_max=build.get("s2_cloud_max"),
        pair_window_days=build.get("pair_window_days"),
        label_ignore_value=build.get("label_ignore_value"),
        rebuild_stale=build.get("rebuild_stale"),
    )
    return build


def build_argv(build: dict, *, out_dir: str, version: str, aoi: str) -> list[str]:
    """Expand the validated ``build`` mapping into builder CLI arguments.

    The boolean ``rebuild_stale`` is passed as ``--rebuild-stale`` only when
    true; a false value omits the flag entirely.
    """
    argv = [
        "--out-dir", out_dir,
        "--version", version,
        "--aoi", aoi,
        "--sar-unit", str(build["sar_unit"]),
        "--s2-cloud-max", str(build["s2_cloud_max"]),
        "--pair-window-days", str(build["pair_window_days"]),
        "--label-ignore-value", str(build["label_ignore_value"]),
    ]
    if bool(build["rebuild_stale"]):
        argv.append("--rebuild-stale")
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description="DVC wrapper for the Hakaluki v0.4 dataset build.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--out-dir", default="backend/data/hakaluki_v04_dataset")
    parser.add_argument("--version", default="0.4.0")
    parser.add_argument("--aoi", default="backend/data/hakaluki_aoi.geojson")
    args = parser.parse_args(argv)

    build = load_build_params(Path(args.params))
    builder_argv = build_argv(build, out_dir=args.out_dir, version=args.version, aoi=args.aoi)

    from scripts.build_hakaluki_v04_dataset import main as build_main

    build_main(builder_argv)


if __name__ == "__main__":
    main()
