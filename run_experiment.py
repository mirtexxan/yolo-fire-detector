"""Config-driven local/cloud pipeline for dataset generation and YOLO training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

import yaml

from generator import generate_dataset
from settings import DatasetGenerationSettings, ImageTransformSettings, TrainingSettings
from train import train_model

PROJECT_ROOT = Path(__file__).resolve().parent


def collect_class_settings(cls: type, exclude: set[str] | None = None) -> dict[str, Any]:
    """Return uppercase class attributes as a regular dict."""
    excluded = exclude or set()
    return {
        name.lower(): getattr(cls, name)
        for name in dir(cls)
        if name.isupper() and name not in excluded
    }


def apply_overrides(cls: type, overrides: dict[str, Any], section_name: str) -> None:
    """Apply lower_snake_case config keys to UPPER_CASE settings classes."""
    for key, value in overrides.items():
        attr_name = key.upper()
        if not hasattr(cls, attr_name):
            raise KeyError(f"Chiave non supportata in {section_name}: {key}")
        setattr(cls, attr_name, value)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config() -> dict[str, Any]:
    """Default configuration for local or cloud experiments."""
    return {
        "project": {
            "environment": "local",
            "label": "baseline",
            "persistent_root": "artifacts/local",
        },
        "dataset": {
            "label": "synthetic-fire",
            "fire_image_paths": list(DatasetGenerationSettings.FIRE_IMAGE_PATHS),
            "num_images": DatasetGenerationSettings.NUM_IMAGES,
            "image_size": DatasetGenerationSettings.IMAGE_SIZE,
            "negative_ratio": DatasetGenerationSettings.NEGATIVE_RATIO,
            "train_split": DatasetGenerationSettings.TRAIN_SPLIT,
            "seed": 42,
            "force_regenerate": False,
        },
        "training": {
            "label": "yolov8-fire",
            "model_size": TrainingSettings.MODEL_SIZE,
            "weights": None,
            "device": TrainingSettings.DEVICE,
            "epochs": TrainingSettings.EPOCHS,
            "batch_size": TrainingSettings.BATCH_SIZE,
            "image_size": TrainingSettings.IMAGE_SIZE,
            "resume": "auto",
        },
        "dataset_settings_overrides": {},
        "image_transform_overrides": {},
        "training_overrides": {},
    }


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and merge the YAML config with defaults."""
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Il file di configurazione deve contenere una mappa YAML")
    return deep_merge(default_config(), loaded)


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    """Resolve relative paths against the repository root."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def stable_hash(payload: dict[str, Any]) -> str:
    """Create a deterministic short hash for configs and manifests."""
    content = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(content).hexdigest()[:10]


def count_files(root: Path, pattern: str) -> int:
    """Count files under a root with a glob pattern."""
    return sum(1 for item in root.glob(pattern) if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML metadata for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def dataset_manifest_path(dataset_root: Path) -> Path:
    """Return the dataset metadata path."""
    return dataset_root / "dataset_info.json"


def dataset_ready(dataset_root: Path, expected_fingerprint: str, expected_images: int) -> bool:
    """Check whether a dataset folder already matches the requested configuration."""
    manifest_path = dataset_manifest_path(dataset_root)
    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError:
        return False

    if manifest.get("fingerprint") != expected_fingerprint:
        return False

    train_images = count_files(dataset_root, "images/train/*.jpg")
    val_images = count_files(dataset_root, "images/val/*.jpg")
    train_labels = count_files(dataset_root, "labels/train/*.txt")
    val_labels = count_files(dataset_root, "labels/val/*.txt")
    total_images = train_images + val_images
    total_labels = train_labels + val_labels
    return total_images == expected_images and total_labels == expected_images and train_images > 0


def build_dataset_snapshot(config: dict[str, Any], fire_image_paths: list[Path]) -> dict[str, Any]:
    """Build the fingerprinted dataset snapshot."""
    dataset_cfg = config["dataset"]
    serialized_paths = [str(path) for path in fire_image_paths]
    return {
        "label": dataset_cfg["label"],
        "fire_image_paths": serialized_paths,
        "num_images": dataset_cfg["num_images"],
        "image_size": dataset_cfg["image_size"],
        "negative_ratio": dataset_cfg["negative_ratio"],
        "train_split": dataset_cfg["train_split"],
        "seed": dataset_cfg.get("seed"),
        "dataset_settings": collect_class_settings(
            DatasetGenerationSettings,
            exclude={"DATASET_ROOT", "FIRE_IMAGE_PATHS", "NUM_IMAGES", "IMAGE_SIZE", "NEGATIVE_RATIO", "TRAIN_SPLIT", "DEMO_MODE", "DEMO_WAIT_MS"},
        ),
        "image_transform": collect_class_settings(ImageTransformSettings),
    }


def prepare_dataset(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Generate or reuse a dataset folder keyed by configuration fingerprint."""
    project_cfg = config["project"]
    dataset_cfg = config["dataset"]
    persistent_root = resolve_path(project_cfg["persistent_root"], project_root)
    assert persistent_root is not None
    datasets_root = persistent_root / "datasets"

    configured_fire_paths = dataset_cfg.get("fire_image_paths")
    if not isinstance(configured_fire_paths, list) or not configured_fire_paths:
        raise ValueError("dataset.fire_image_paths deve essere una lista non vuota")

    fire_image_paths: list[Path] = []
    for raw_path in configured_fire_paths:
        resolved_path = resolve_path(str(raw_path), project_root)
        if resolved_path is None or not resolved_path.exists():
            raise FileNotFoundError(f"Fire image non trovata: {raw_path}")
        fire_image_paths.append(resolved_path)

    snapshot = build_dataset_snapshot(config, fire_image_paths)
    fingerprint = stable_hash(snapshot)
    dataset_folder_name = f"{dataset_cfg['label']}__{fingerprint}"
    dataset_root = datasets_root / dataset_folder_name
    manifest_path = dataset_manifest_path(dataset_root)

    reused = dataset_ready(dataset_root, fingerprint, dataset_cfg["num_images"]) and not dataset_cfg.get("force_regenerate", False)

    if reused:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(f"♻️ Dataset riutilizzato: {dataset_root}")
        return {
            "root": dataset_root,
            "fingerprint": fingerprint,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "reused": True,
        }

    stats = generate_dataset(
        dataset_root=str(dataset_root),
        fire_image_paths=[str(path) for path in fire_image_paths],
        num_images=dataset_cfg["num_images"],
        image_size=dataset_cfg["image_size"],
        negative_ratio=dataset_cfg["negative_ratio"],
        train_split=dataset_cfg["train_split"],
        demo_mode=False,
        seed=dataset_cfg.get("seed"),
        clean=True,
    )

    train_images = count_files(dataset_root, "images/train/*.jpg")
    val_images = count_files(dataset_root, "images/val/*.jpg")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": dataset_cfg["label"],
        "environment": project_cfg["environment"],
        "fingerprint": fingerprint,
        "root": str(dataset_root),
        "snapshot": snapshot,
        "stats": stats,
        "counts": {
            "train_images": train_images,
            "val_images": val_images,
            "total_images": train_images + val_images,
        },
    }
    write_json(manifest_path, manifest)
    print(f"🧾 Manifest dataset scritto in: {manifest_path}")
    return {
        "root": dataset_root,
        "fingerprint": fingerprint,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "reused": False,
    }


def build_run_label(config: dict[str, Any], dataset_fingerprint: str) -> str:
    """Compose a readable run label with enough entropy to distinguish experiments."""
    project_cfg = config["project"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    return "__".join(
        [
            project_cfg["environment"],
            project_cfg["label"],
            training_cfg["label"],
            f"yolov8{training_cfg['model_size']}",
            dataset_cfg["label"],
            dataset_fingerprint,
        ]
    )


def resolve_resume_policy(run_dir: Path, resume_policy: str) -> bool:
    """Determine whether training should resume from the last checkpoint."""
    last_checkpoint = run_dir / "weights" / "last.pt"
    if resume_policy == "always":
        return True
    if resume_policy == "never":
        return False
    return last_checkpoint.exists()


def register_model_artifacts(
    persistent_root: Path,
    run_label: str,
    export_dir: Path,
    dataset_manifest: dict[str, Any],
    resolved_config: dict[str, Any],
) -> dict[str, str]:
    """Copy final model and metadata into a persistent model registry."""
    registry_dir = persistent_root / "models"
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_path = export_dir / "best.pt"
    registry_model_path = registry_dir / f"{run_label}.pt"
    shutil.copy2(model_path, registry_model_path)

    metadata = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_path": str(registry_model_path),
        "export_dir": str(export_dir),
        "dataset": {
            "fingerprint": dataset_manifest["fingerprint"],
            "label": dataset_manifest["label"],
            "root": dataset_manifest["root"],
            "manifest_path": str(export_dir.parent / "dataset_info.json"),
        },
        "config": resolved_config,
    }
    registry_metadata_path = registry_dir / f"{run_label}.json"
    write_json(registry_metadata_path, metadata)

    latest_pointer_path = registry_dir / "latest.json"
    write_json(
        latest_pointer_path,
        {
            "run_label": run_label,
            "model_path": str(registry_model_path),
            "metadata_path": str(registry_metadata_path),
        },
    )

    return {
        "model_path": str(registry_model_path),
        "metadata_path": str(registry_metadata_path),
        "latest_path": str(latest_pointer_path),
    }


def run_pipeline(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Execute the full dataset-generation and training pipeline."""
    project_cfg = config["project"]
    training_cfg = config["training"]

    apply_overrides(DatasetGenerationSettings, config.get("dataset_settings_overrides", {}), "dataset_settings_overrides")
    apply_overrides(ImageTransformSettings, config.get("image_transform_overrides", {}), "image_transform_overrides")
    apply_overrides(TrainingSettings, config.get("training_overrides", {}), "training_overrides")

    dataset_info = prepare_dataset(config, PROJECT_ROOT)
    persistent_root = resolve_path(project_cfg["persistent_root"], PROJECT_ROOT)
    assert persistent_root is not None
    runs_root = persistent_root / "runs"

    run_label = build_run_label(config, dataset_info["fingerprint"])
    run_dir = runs_root / run_label
    export_dir = run_dir / "final_export"
    resume_enabled = resolve_resume_policy(run_dir, training_cfg["resume"])

    resolved_config = {
        "config_path": str(config_path),
        "project": config["project"],
        "dataset": config["dataset"],
        "training": config["training"],
        "dataset_settings_overrides": config.get("dataset_settings_overrides", {}),
        "image_transform_overrides": config.get("image_transform_overrides", {}),
        "training_overrides": config.get("training_overrides", {}),
        "resolved_dataset_root": str(dataset_info["root"]),
        "run_label": run_label,
        "resume_enabled": resume_enabled,
    }
    write_yaml(run_dir / "resolved_config.yaml", resolved_config)
    write_json(run_dir / "dataset_info.json", dataset_info["manifest"])

    extra_summary = {
        "run_label": run_label,
        "environment": project_cfg["environment"],
        "dataset_fingerprint": dataset_info["fingerprint"],
        "dataset_manifest_path": str(dataset_info["manifest_path"]),
        "dataset_reused": dataset_info["reused"],
        "fire_image_paths": dataset_info["manifest"]["snapshot"]["fire_image_paths"],
        "base_image_usage": dataset_info["manifest"].get("stats", {}).get("base_image_usage", {}),
        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
    }

    train_model(
        model_size=training_cfg["model_size"],
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        image_size=training_cfg["image_size"],
        device=training_cfg["device"],
        resume=resume_enabled,
        dataset_root=str(dataset_info["root"]),
        project_name=str(runs_root),
        experiment_name=run_label,
        weights=training_cfg.get("weights"),
        export_dir=str(export_dir),
        extra_summary=extra_summary,
    )

    registry_paths = register_model_artifacts(
        persistent_root=persistent_root,
        run_label=run_label,
        export_dir=export_dir,
        dataset_manifest=dataset_info["manifest"],
        resolved_config=resolved_config,
    )

    summary = {
        "dataset_root": str(dataset_info["root"]),
        "dataset_manifest_path": str(dataset_info["manifest_path"]),
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "run_label": run_label,
        "resume_enabled": resume_enabled,
        "registry": registry_paths,
    }
    write_json(run_dir / "pipeline_summary.json", summary)

    print("\n" + "=" * 60)
    print("Pipeline completata")
    print("=" * 60)
    print(f"Dataset: {summary['dataset_root']}")
    print(f"Run: {summary['run_dir']}")
    print(f"Export: {summary['export_dir']}")
    print(f"Registry model: {summary['registry']['model_path']}")

    return summary


def main() -> None:
    """CLI entrypoint for config-driven experiments."""
    parser = argparse.ArgumentParser(description="Pipeline locale/cloud guidata da file YAML")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file YAML di configurazione")
    args = parser.parse_args()

    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config non trovata: {args.config}")

    config = load_config(config_path)
    run_pipeline(config, config_path)


if __name__ == "__main__":
    main()
