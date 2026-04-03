"""Config-driven local/cloud pipeline for dataset generation and YOLO training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from config_utils import deep_merge, load_layered_config
from generator import generate_dataset
from settings import DatasetGenerationSettings, ImageTransformSettings, TrainingSettings
from train import create_dataset_yaml, train_model

PROJECT_ROOT = Path(__file__).resolve().parent


def slugify(value: str) -> str:
    """Convert free-form labels into stable path-safe slugs."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-") or "item"


def portable_path(path: Path, root: Path | None = None) -> str:
    """Serialize a path relative to a chosen root when possible."""
    resolved_path = path.resolve()
    if root is not None:
        try:
            return resolved_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved_path.as_posix()


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


def default_config() -> dict[str, Any]:
    """Default configuration for experiments."""
    return {
        "project": {
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
            "require_gpu": False,
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
    loaded = load_layered_config(config_path)
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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML metadata with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML metadata into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata YAML non valida: {path}")
    return payload


def dataset_manifest_path(dataset_root: Path) -> Path:
    """Return the dataset metadata path."""
    return dataset_root / "dataset_manifest.yaml"


def dataset_ready(dataset_root: Path, expected_fingerprint: str, expected_images: int) -> bool:
    """Check whether a dataset folder already matches the requested configuration."""
    manifest_path = dataset_manifest_path(dataset_root)
    if not manifest_path.exists():
        return False

    try:
        manifest = read_yaml(manifest_path)
    except (OSError, ValueError, yaml.YAMLError):
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
    serialized_paths = [portable_path(path, PROJECT_ROOT) for path in fire_image_paths]
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


def build_dataset_manifest(
    *,
    created_at: str,
    status: str,
    config: dict[str, Any],
    persistent_root: Path,
    dataset_root: Path,
    fingerprint: str,
    snapshot: dict[str, Any],
    stats: dict[str, Any],
    counts: dict[str, int],
    yolo_dataset_path: Path,
) -> dict[str, Any]:
    """Compose dataset manifest payloads for in-progress or completed generations."""
    project_cfg = config["project"]
    dataset_cfg = config["dataset"]
    manifest: dict[str, Any] = {
        "created_at": created_at,
        "status": status,
        "label": dataset_cfg["label"],
        "fingerprint": fingerprint,
        "root": portable_path(dataset_root, persistent_root),
        "yolo_dataset_path": portable_path(yolo_dataset_path, persistent_root),
        "snapshot": snapshot,
        "stats": stats,
        "counts": counts,
    }
    if project_cfg.get("environment"):
        manifest["environment"] = project_cfg["environment"]
    if status == "completed":
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    return manifest


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
    dataset_folder_name = f"{slugify(dataset_cfg['label'])}-{fingerprint}"
    dataset_root = datasets_root / dataset_folder_name
    manifest_path = dataset_manifest_path(dataset_root)

    reused = dataset_ready(dataset_root, fingerprint, dataset_cfg["num_images"]) and not dataset_cfg.get("force_regenerate", False)

    if reused:
        manifest = read_yaml(manifest_path)
        print(f"♻️ Dataset riutilizzato: {dataset_root}")
        return {
            "root": dataset_root,
            "fingerprint": fingerprint,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "reused": True,
        }

    dataset_root.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    yolo_dataset_path = Path(create_dataset_yaml(str(dataset_root)))
    initial_manifest = build_dataset_manifest(
        created_at=created_at,
        status="generating",
        config=config,
        persistent_root=persistent_root,
        dataset_root=dataset_root,
        fingerprint=fingerprint,
        snapshot=snapshot,
        stats={
            "dataset_root": portable_path(dataset_root, persistent_root),
            "num_images_target": dataset_cfg["num_images"],
            "image_size": dataset_cfg["image_size"],
            "negative_ratio": dataset_cfg["negative_ratio"],
            "train_split": dataset_cfg["train_split"],
            "seed": dataset_cfg.get("seed"),
            "fire_image_paths": [portable_path(path, PROJECT_ROOT) for path in fire_image_paths],
            "base_image_usage": {
                portable_path(path, PROJECT_ROOT): 0 for path in fire_image_paths
            },
        },
        counts={
            "train_images": 0,
            "val_images": 0,
            "total_images": 0,
        },
        yolo_dataset_path=yolo_dataset_path,
    )
    write_yaml(manifest_path, initial_manifest)
    print(f"🧾 Manifest dataset iniziale scritto in: {manifest_path}")

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
    stats["dataset_root"] = portable_path(dataset_root, persistent_root)
    stats["fire_image_paths"] = [portable_path(path, PROJECT_ROOT) for path in fire_image_paths]
    stats["base_image_usage"] = {
        portable_path(Path(path), PROJECT_ROOT): count
        for path, count in stats.get("base_image_usage", {}).items()
    }

    train_images = count_files(dataset_root, "images/train/*.jpg")
    val_images = count_files(dataset_root, "images/val/*.jpg")
    manifest = build_dataset_manifest(
        created_at=created_at,
        status="completed",
        config=config,
        persistent_root=persistent_root,
        dataset_root=dataset_root,
        fingerprint=fingerprint,
        snapshot=snapshot,
        stats=stats,
        counts={
            "train_images": train_images,
            "val_images": val_images,
            "total_images": train_images + val_images,
        },
        yolo_dataset_path=yolo_dataset_path,
    )
    write_yaml(manifest_path, manifest)
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
    training_slug = slugify(training_cfg["label"])
    model_slug = slugify(f"yolov8{training_cfg['model_size']}")
    raw_parts = [
        slugify(project_cfg["label"]),
        training_slug,
        model_slug,
        slugify(dataset_cfg["label"]),
        dataset_fingerprint,
    ]

    parts: list[str] = []
    seen_tokens: set[str] = set()
    for part in raw_parts:
        tokens = [token for token in part.split("-") if token]
        unique_tokens = [token for token in tokens if token not in seen_tokens]
        if unique_tokens:
            parts.append("-".join(unique_tokens))
            seen_tokens.update(unique_tokens)

    return "-".join(parts)


def resolve_resume_policy(run_dir: Path, resume_policy: str) -> bool:
    """Determine whether training should resume from the last checkpoint."""
    last_checkpoint = run_dir / "weights" / "last.pt"
    if resume_policy == "always":
        return True
    if resume_policy == "never":
        return False
    return last_checkpoint.exists()


def register_export_artifacts(
    persistent_root: Path,
    run_label: str,
    run_dir: Path,
    dataset_manifest: dict[str, Any],
    resolved_config: dict[str, Any],
) -> dict[str, str]:
    """Copy final model and metadata into a persistent export registry."""
    export_dir = persistent_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "weights" / "best.pt"
    export_model_path = export_dir / f"{run_label}.pt"
    shutil.copy2(model_path, export_model_path)

    metadata = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_path": portable_path(export_model_path, persistent_root),
        "run_dir": portable_path(run_dir, persistent_root),
        "training_run_path": portable_path(run_dir / "training_run.yaml", persistent_root),
        "resolved_config_path": portable_path(run_dir / "resolved_config.yaml", persistent_root),
        "dataset": {
            "fingerprint": dataset_manifest["fingerprint"],
            "label": dataset_manifest["label"],
            "root": dataset_manifest["root"],
            "manifest_path": f"{dataset_manifest['root']}/dataset_manifest.yaml",
        },
        "config": resolved_config,
    }
    export_metadata_path = export_dir / f"{run_label}.yaml"
    write_yaml(export_metadata_path, metadata)

    latest_pointer_path = export_dir / "latest.yaml"
    write_yaml(
        latest_pointer_path,
        {
            "run_label": run_label,
            "model_path": portable_path(export_model_path, persistent_root),
            "metadata_path": portable_path(export_metadata_path, persistent_root),
        },
    )

    return {
        "model_path": portable_path(export_model_path, persistent_root),
        "metadata_path": portable_path(export_metadata_path, persistent_root),
        "latest_path": portable_path(latest_pointer_path, persistent_root),
    }


def cleanup_completed_run(run_dir: Path) -> None:
    """Remove heavy checkpoint artifacts after a successful completed run."""
    shutil.rmtree(run_dir / "weights", ignore_errors=True)


def run_pipeline(config: dict[str, Any], config_path: Path, *, skip_training: bool = False) -> dict[str, Any]:
    """Execute the dataset-generation pipeline and optionally skip training."""
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
    resume_enabled = resolve_resume_policy(run_dir, training_cfg["resume"])
    require_gpu = bool(training_cfg.get("require_gpu", False))
    training_cfg["require_gpu"] = require_gpu

    resolved_config = {
        "config_path": portable_path(config_path, PROJECT_ROOT),
        "project": config["project"],
        "dataset": config["dataset"],
        "training": config["training"],
        "dataset_settings_overrides": config.get("dataset_settings_overrides", {}),
        "image_transform_overrides": config.get("image_transform_overrides", {}),
        "training_overrides": config.get("training_overrides", {}),
        "resolved_dataset_root": portable_path(dataset_info["root"], persistent_root),
        "run_label": run_label,
        "resume_enabled": resume_enabled,
        "skip_training": skip_training,
    }
    write_yaml(run_dir / "resolved_config.yaml", resolved_config)

    if skip_training:
        summary = {
            "mode": "dataset-only",
            "dataset_root": portable_path(dataset_info["root"], persistent_root),
            "dataset_manifest_path": portable_path(dataset_info["manifest_path"], persistent_root),
            "run_dir": portable_path(run_dir, persistent_root),
            "resolved_config_path": portable_path(run_dir / "resolved_config.yaml", persistent_root),
            "run_label": run_label,
            "resume_enabled": resume_enabled,
            "dataset_fingerprint": dataset_info["fingerprint"],
            "dataset_reused": dataset_info["reused"],
        }
        write_yaml(run_dir / "pipeline_summary.yaml", summary)

        print("\n" + "=" * 60)
        print("Pipeline completata (training saltato)")
        print("=" * 60)
        print(f"Dataset: {summary['dataset_root']}")
        print(f"Manifest: {summary['dataset_manifest_path']}")
        print(f"Run metadata: {summary['resolved_config_path']}")

        return summary

    extra_summary = {
        "run_label": run_label,
        "dataset_fingerprint": dataset_info["fingerprint"],
        "dataset_manifest_path": portable_path(dataset_info["manifest_path"], persistent_root),
        "dataset_reused": dataset_info["reused"],
        "fire_image_paths": dataset_info["manifest"]["snapshot"]["fire_image_paths"],
        "base_image_usage": dataset_info["manifest"].get("stats", {}).get("base_image_usage", {}),
        "resolved_config_path": portable_path(run_dir / "resolved_config.yaml", persistent_root),
    }
    if project_cfg.get("environment"):
        extra_summary["environment"] = project_cfg["environment"]

    train_model(
        model_size=training_cfg["model_size"],
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        image_size=training_cfg["image_size"],
        device=training_cfg["device"],
        require_gpu=require_gpu,
        resume=resume_enabled,
        dataset_root=str(dataset_info["root"]),
        project_name=str(runs_root),
        experiment_name=run_label,
        weights=training_cfg.get("weights"),
        extra_summary=extra_summary,
    )

    export_paths = register_export_artifacts(
        persistent_root=persistent_root,
        run_label=run_label,
        run_dir=run_dir,
        dataset_manifest=dataset_info["manifest"],
        resolved_config=resolved_config,
    )

    cleanup_completed_run(run_dir)

    summary = {
        "dataset_root": portable_path(dataset_info["root"], persistent_root),
        "dataset_manifest_path": portable_path(dataset_info["manifest_path"], persistent_root),
        "run_dir": portable_path(run_dir, persistent_root),
        "training_run_path": portable_path(run_dir / "training_run.yaml", persistent_root),
        "run_label": run_label,
        "resume_enabled": resume_enabled,
        "exports": export_paths,
    }
    write_yaml(run_dir / "pipeline_summary.yaml", summary)

    print("\n" + "=" * 60)
    print("Pipeline completata")
    print("=" * 60)
    print(f"Dataset: {summary['dataset_root']}")
    print(f"Run: {summary['run_dir']}")
    print(f"Training metadata: {summary['training_run_path']}")
    print(f"Export model: {summary['exports']['model_path']}")

    try:
        import google.colab  # type: ignore

        print("\n" + "=" * 60)
        print("GOOGLE COLAB - COMANDI DI DOWNLOAD")
        print("=" * 60)
        print("from google.colab import files")
        print(f"files.download('{summary['exports']['model_path']}')")
        print(f"files.download('{summary['training_run_path']}')")
        print("=" * 60)
    except ImportError:
        pass

    return summary


def main() -> None:
    """CLI entrypoint for config-driven experiments."""
    parser = argparse.ArgumentParser(description="Pipeline guidata da file YAML")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file YAML di configurazione")
    parser.add_argument("--skip-training", action="store_true", help="Genera o riusa il dataset senza avviare il training")
    args = parser.parse_args()

    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config non trovata: {args.config}")

    config = load_config(config_path)
    run_pipeline(config, config_path, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
