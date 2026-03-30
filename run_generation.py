"""Dataset-only runner with support for partial YAML configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from settings import DatasetGenerationSettings, ImageTransformSettings
from run_experiment import (
    PROJECT_ROOT,
    apply_overrides,
    load_config,
    portable_path,
    prepare_dataset,
    resolve_path,
    write_yaml,
)


def run_generation_only(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Generate or reuse dataset artifacts without starting training."""
    apply_overrides(DatasetGenerationSettings, config.get("dataset_settings_overrides", {}), "dataset_settings_overrides")
    apply_overrides(ImageTransformSettings, config.get("image_transform_overrides", {}), "image_transform_overrides")

    dataset_info = prepare_dataset(config, PROJECT_ROOT)

    persistent_root = resolve_path(config["project"]["persistent_root"], PROJECT_ROOT)
    assert persistent_root is not None

    summary = {
        "mode": "dataset-only",
        "config_path": portable_path(config_path, PROJECT_ROOT),
        "dataset_root": portable_path(dataset_info["root"], persistent_root),
        "dataset_manifest_path": portable_path(dataset_info["manifest_path"], persistent_root),
        "dataset_fingerprint": dataset_info["fingerprint"],
        "dataset_reused": dataset_info["reused"],
        "yolo_dataset_path": dataset_info["manifest"].get("yolo_dataset_path"),
    }

    summary_path = dataset_info["root"] / "generation_summary.yaml"
    write_yaml(summary_path, summary)

    print("\n" + "=" * 60)
    print("Generazione dataset completata")
    print("=" * 60)
    print(f"Dataset: {summary['dataset_root']}")
    print(f"Manifest: {summary['dataset_manifest_path']}")
    print(f"Riutilizzato: {summary['dataset_reused']}")
    print(f"Summary: {portable_path(summary_path, persistent_root)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera o riusa il dataset senza avviare il training")
    parser.add_argument("--config", type=str, required=True, help="Percorso YAML (puo contenere anche solo sezioni dataset/project)")
    args = parser.parse_args()

    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config non trovata: {args.config}")

    config = load_config(config_path)
    run_generation_only(config, config_path)


if __name__ == "__main__":
    main()
