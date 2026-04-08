"""Config-driven local/cloud pipeline for dataset generation and YOLO training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import deep_merge, load_layered_config
from generator import generate_dataset
from settings import DatasetGenerationSettings, ImageTransformSettings, TrainingSettings
from train import create_dataset_yaml, train_model
from tools.dataset.collect_hard_negatives import collect_hard_negatives, _resolve_model as resolve_hn_model
from tools.dataset.dataset_report import generate_dataset_report
from tools.dataset.fetch_unsplash_backgrounds import fetch_backgrounds, load_unsplash_access_key


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
            "epochs": TrainingSettings.EPOCHS,
            "batch_size": TrainingSettings.BATCH_SIZE,
            "image_size": TrainingSettings.IMAGE_SIZE,
            "resume": "auto",
        },
        "dataset_settings_overrides": {},
        "image_transform_overrides": {
            "use_unsplash_backgrounds": False,
            "unsplash_background_prob": 0.65,
            "unsplash_background_dirs": [],
            "use_hard_negative_backgrounds": False,
            "hard_negative_background_prob": 0.65,
            "hard_negative_background_dirs": [],
        },
        "hard_negative_mining": {
            "enabled": False,
            "sources": [],
            "weights": "latest",
            "conf": 0.15,
            "stride": 5,
            "max_samples": 500,
            "filter_negatives_only": False,
            "output_collection": "auto",
        },
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
        report_path = dataset_root / "dataset_report.yaml"
        try:
            generate_dataset_report(dataset_root, output_yaml=report_path)
            print(f"📊 Dataset report aggiornato: {report_path}")
        except Exception as exc:  # pragma: no cover - report must not block pipeline
            print(f"⚠️ Dataset report non disponibile ({dataset_root}): {exc}")
        print(f"♻️ Dataset riutilizzato: {dataset_root}")
        return {
            "root": dataset_root,
            "fingerprint": fingerprint,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "report_path": report_path,
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
    report_path = dataset_root / "dataset_report.yaml"
    try:
        generate_dataset_report(dataset_root, output_yaml=report_path)
        print(f"📊 Dataset report scritto in: {report_path}")
    except Exception as exc:  # pragma: no cover - report must not block pipeline
        print(f"⚠️ Dataset report non disponibile ({dataset_root}): {exc}")
    return {
        "root": dataset_root,
        "fingerprint": fingerprint,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "report_path": report_path,
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


def _ensure_real_backgrounds() -> None:
    """Auto-fetch Unsplash backgrounds for any configured Unsplash dirs that are missing or empty."""
    use_unsplash = bool(getattr(ImageTransformSettings, "USE_UNSPLASH_BACKGROUNDS", False))
    if not use_unsplash:
        return

    dirs: list[str] = getattr(ImageTransformSettings, "UNSPLASH_BACKGROUND_DIRS", []) or []
    if not dirs:
        return

    _image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def _is_empty(p: Path) -> bool:
        if not p.exists():
            return True
        return not any(f.suffix.lower() in _image_exts for f in p.rglob("*") if f.is_file())

    missing_themes: list[str] = []
    output_root: Path | None = None

    for raw_dir in dirs:
        d = Path(raw_dir)
        if _is_empty(d):
            missing_themes.append(d.name)
            if output_root is None:
                output_root = d.parent  # e.g. .../unsplash/

    if not missing_themes:
        return

    access_key = load_unsplash_access_key(PROJECT_ROOT)
    if not access_key:
        print(
            "⚠️  Sfondi reali abilitati ma le cartelle sono vuote o mancanti: "
            + ", ".join(missing_themes)
            + "\n   Imposta UNSPLASH_ACCESS_KEY oppure crea .env.local/.env con la chiave, poi riprova;"
            + "\n   in alternativa scarica manualmente con:"
            + "\n   python tools/dataset/fetch_unsplash_backgrounds.py "
            + f"--themes \"{','.join(missing_themes)}\" --count 60"
        )
        return

    print(f"🌐 Scarico sfondi Unsplash per i temi mancanti: {', '.join(missing_themes)} ...")
    assert output_root is not None
    try:
        result = fetch_backgrounds(
            access_key=access_key,
            themes=missing_themes,
            output_root=output_root,
            total_per_theme=60,
            per_page=30,
            orientation="landscape",
            min_width=1200,
        )
        for theme, info in result.get("themes", {}).items():
            print(f"   {theme}: {info.get('downloaded', 0)}/{info.get('requested', 0)} immagini scaricate")
    except Exception as exc:
        print(f"⚠️  Errore durante il download degli sfondi Unsplash: {exc}")
        print("   Il dataset verrà generato con soli sfondi sintetici.")


def _normalize_unique_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in paths:
        text = str(item).strip().replace("\\", "/")
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_image_transform_overrides(config: dict[str, Any]) -> None:
    """Normalize image_transform_overrides to explicit Unsplash/Hard-Negative keys."""
    image_overrides = config.setdefault("image_transform_overrides", {})
    if not isinstance(image_overrides, dict):
        raise ValueError("image_transform_overrides deve essere una mappa")

    legacy_keys = {"use_real_backgrounds", "real_background_prob", "real_background_dirs"}
    present_legacy = sorted(key for key in legacy_keys if key in image_overrides)
    if present_legacy:
        raise KeyError(
            "Chiavi legacy non supportate in image_transform_overrides: "
            + ", ".join(present_legacy)
            + ". Usa use_unsplash_backgrounds/unsplash_background_* e use_hard_negative_backgrounds/hard_negative_background_*"
        )

    use_unsplash = bool(image_overrides.get("use_unsplash_backgrounds", False))
    unsplash_prob = image_overrides.get("unsplash_background_prob", 0.65)
    unsplash_dirs = image_overrides.get("unsplash_background_dirs", [])
    if not isinstance(unsplash_dirs, list):
        unsplash_dirs = []

    use_hn = bool(image_overrides.get("use_hard_negative_backgrounds", False))
    hn_prob = image_overrides.get("hard_negative_background_prob", 0.65)
    hn_dirs = image_overrides.get("hard_negative_background_dirs", [])
    if not isinstance(hn_dirs, list):
        hn_dirs = []

    image_overrides["use_unsplash_backgrounds"] = use_unsplash
    image_overrides["unsplash_background_prob"] = float(unsplash_prob)
    image_overrides["unsplash_background_dirs"] = _normalize_unique_paths([str(item) for item in unsplash_dirs])
    image_overrides["use_hard_negative_backgrounds"] = use_hn
    image_overrides["hard_negative_background_prob"] = float(hn_prob)
    image_overrides["hard_negative_background_dirs"] = _normalize_unique_paths([str(item) for item in hn_dirs])


def _run_hard_negative_mining(config: dict[str, Any]) -> list[str]:
    """Optionally collect hard negatives and return output dirs to be reused as backgrounds."""
    section = config.get("hard_negative_mining", {})
    if not isinstance(section, dict) or not bool(section.get("enabled", False)):
        return []

    raw_sources = section.get("sources", [])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("hard_negative_mining.enabled=true ma hard_negative_mining.sources e' vuoto")

    project_cfg = config.get("project", {})
    if not isinstance(project_cfg, dict):
        raise ValueError("Config progetto non valida")
    persistent_root = resolve_path(str(project_cfg.get("persistent_root", "artifacts/local")), PROJECT_ROOT)
    assert persistent_root is not None

    weights_arg = str(section.get("weights", "latest") or "latest")
    conf_threshold = float(section.get("conf", 0.15))
    stride = int(section.get("stride", 5))
    max_samples = int(section.get("max_samples", 500))
    filter_negatives_only = bool(section.get("filter_negatives_only", False))
    output_collection = str(section.get("output_collection", "auto") or "auto").strip()

    model_path = resolve_hn_model(weights_arg)
    output_dirs: list[str] = []

    print("🧪 Hard negative mining abilitato")
    for raw_source in raw_sources:
        source_text = str(raw_source).strip()
        if not source_text:
            continue
        source_path = resolve_path(source_text, PROJECT_ROOT)
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Sorgente hard negative non trovata: {source_text}")

        if output_collection and output_collection.lower() != "auto":
            collection_name = slugify(output_collection)
        else:
            collection_name = slugify(source_path.stem or source_path.name)

        output_dir = (persistent_root / "hard_negatives" / collection_name).resolve()
        collect_hard_negatives(
            source=source_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
            output_dir=output_dir,
            max_samples=max_samples,
            stride=stride,
            filter_negatives_only=filter_negatives_only,
            deduplicate=True,
        )
        output_dirs.append(output_dir.as_posix())

    return _normalize_unique_paths(output_dirs)


def _inject_hard_negative_background_dirs(config: dict[str, Any], mined_dirs: list[str]) -> None:
    if not mined_dirs:
        return

    image_overrides = config.setdefault("image_transform_overrides", {})
    if not isinstance(image_overrides, dict):
        raise ValueError("image_transform_overrides deve essere una mappa")

    existing = image_overrides.get("hard_negative_background_dirs", [])
    existing_dirs = [str(item) for item in existing] if isinstance(existing, list) else []
    merged_dirs = _normalize_unique_paths(existing_dirs + mined_dirs)

    image_overrides["hard_negative_background_dirs"] = merged_dirs
    image_overrides["use_hard_negative_backgrounds"] = True
    try:
        current_prob = float(image_overrides.get("hard_negative_background_prob", 0.0) or 0.0)
    except (TypeError, ValueError):
        current_prob = 0.0
    if current_prob <= 0.0:
        image_overrides["hard_negative_background_prob"] = 0.65


def run_pipeline(config: dict[str, Any], config_path: Path, *, skip_training: bool = False) -> dict[str, Any]:
    """Execute the dataset-generation pipeline and optionally skip training."""
    project_cfg = config["project"]
    training_cfg = config["training"]

    apply_overrides(DatasetGenerationSettings, config.get("dataset_settings_overrides", {}), "dataset_settings_overrides")
    apply_overrides(TrainingSettings, config.get("training_overrides", {}), "training_overrides")

    mined_dirs = _run_hard_negative_mining(config)
    _inject_hard_negative_background_dirs(config, mined_dirs)
    _normalize_image_transform_overrides(config)
    apply_overrides(ImageTransformSettings, config.get("image_transform_overrides", {}), "image_transform_overrides")

    _ensure_real_backgrounds()

    dataset_info = prepare_dataset(config, PROJECT_ROOT)
    persistent_root = resolve_path(project_cfg["persistent_root"], PROJECT_ROOT)
    assert persistent_root is not None
    runs_root = persistent_root / "runs"

    run_label = build_run_label(config, dataset_info["fingerprint"])
    run_dir = runs_root / run_label
    resume_enabled = resolve_resume_policy(run_dir, training_cfg["resume"])

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
            "dataset_report_path": portable_path(dataset_info["report_path"], persistent_root),
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
        "dataset_report_path": portable_path(dataset_info["report_path"], persistent_root),
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


_GENERATED_CONFIGS_DIR = PROJECT_ROOT / "configs" / "generated"


def _fuzzy_resolve_config(name: str) -> Path:
    """Find a config YAML by name (with or without .yaml) inside configs/generated/.

    Resolution order:
    1. Exact filename match (``name`` or ``name.yaml``)
    2. ``startswith`` match (first alphabetically)

    Raises FileNotFoundError if nothing matches.
    """
    stem = name.removesuffix(".yaml")
    candidates = sorted(_GENERATED_CONFIGS_DIR.glob("*.yaml"))

    # 1. exact match
    for candidate in candidates:
        if candidate.stem == stem or candidate.name == name:
            print(f"ℹ️  Config trovata in configs/generated/: {candidate.name}")
            return candidate

    # 2. startswith match
    for candidate in candidates:
        if candidate.stem.startswith(stem) or candidate.name.startswith(name):
            print(f"ℹ️  Config trovata per prefisso in configs/generated/: {candidate.name}")
            return candidate

    raise FileNotFoundError(
        f"Config non trovata: '{name}'. "
        f"Cerca in configs/generated/ i file disponibili: "
        f"{[c.name for c in candidates] or '(nessuno)'}"
    )


def main() -> None:
    """CLI entrypoint for config-driven experiments."""
    parser = argparse.ArgumentParser(description="Pipeline guidata da file YAML")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file YAML di configurazione")
    parser.add_argument("--skip-training", action="store_true", help="Genera o riusa il dataset senza avviare il training")
    args = parser.parse_args()

    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        config_path = _fuzzy_resolve_config(args.config)

    config = load_config(config_path)
    run_pipeline(config, config_path, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
