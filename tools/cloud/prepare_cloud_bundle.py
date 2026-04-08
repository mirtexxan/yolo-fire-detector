"""Creates a zip bundle of the project for Colab or similar notebook services."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import LATEST_CLOUD_CONFIG_RELATIVE, choose_default_cloud_launchable_config

BUNDLED_CLOUD_CONFIG_RELATIVE = Path("configs") / LATEST_CLOUD_CONFIG_RELATIVE


ROOT_LEVEL_EXCLUDES = {
    ".git",
    ".venv",
    ".vscode",
    "artifacts",
    "dataset",
    "detections",
    "fire_detector_runs",
    "runs",
    "dist",
    "env",
    "venv",
}

ANYWHERE_EXCLUDES = {
    ".ipynb_checkpoints",
    "__pycache__",
}

DEFAULT_INCLUDE_NAMES = {
    "README.md",
    "CLOUD_TRAINING.md",
    "TRAINING_PRESETS.md",
    "requirements.txt",
}

DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".ipynb",
    ".yaml",
    ".md",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def detect_default_launchable_config(project_root: Path) -> str | None:
    """Return the single cloud config that must be bundled."""
    return choose_default_cloud_launchable_config(project_root / "configs")


def should_skip(path: Path, include_dataset: bool, include_runs: bool) -> bool:
    parts = path.parts
    root_excludes = set(ROOT_LEVEL_EXCLUDES)
    if include_dataset:
        root_excludes.discard("dataset")
    if include_runs:
        root_excludes.discard("fire_detector_runs")
        root_excludes.discard("runs")

    if any(part in ANYWHERE_EXCLUDES for part in parts):
        return True

    return bool(parts) and parts[0] in root_excludes


def should_include(relative_path: Path) -> bool:
    if relative_path.parts[:2] == ("configs", "generated"):
        return relative_path == BUNDLED_CLOUD_CONFIG_RELATIVE
    if relative_path.parts[:1] == ("configs",):
        return False
    if relative_path.name in DEFAULT_INCLUDE_NAMES or relative_path.suffix.lower() in DEFAULT_INCLUDE_SUFFIXES:
        return True
    if relative_path.suffix.lower() == ".pt" and len(relative_path.parts) == 1:
        return True
    return "base_fire_images" in relative_path.parts and relative_path.suffix.lower() == ".png"


def strip_notebook_outputs(notebook_path: Path) -> bytes:
    """Return a UTF-8 notebook payload with code outputs removed."""
    payload = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    cells = payload.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict) and cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    return json.dumps(payload, indent=1, ensure_ascii=False).encode("utf-8")


def detect_latest_generated_dataset_dir(project_root: Path) -> Path | None:
    """Return the latest generated dataset directory under artifacts/local/datasets/."""
    datasets_root = project_root / "artifacts" / "local" / "datasets"
    if not datasets_root.exists() or not datasets_root.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for dataset_dir in datasets_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        manifest = dataset_dir / "dataset_manifest.yaml"
        if not manifest.exists() or not manifest.is_file():
            continue
        candidates.append((manifest.stat().st_mtime, dataset_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].name.lower()))
    return candidates[-1][1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_yaml_map(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config YAML non valida: {path}")
    return payload


def resolve_required_real_background_dirs(project_root: Path) -> list[Path]:
    """Resolve Unsplash background dirs from latest cloud config."""
    config_path = project_root / BUNDLED_CLOUD_CONFIG_RELATIVE
    config = _read_yaml_map(config_path)
    image_transform = config.get("image_transform_overrides", {})
    if not isinstance(image_transform, dict):
        return []
    legacy_keys = {"use_real_backgrounds", "real_background_prob", "real_background_dirs"}
    present_legacy = sorted(key for key in legacy_keys if key in image_transform)
    if present_legacy:
        raise ValueError(
            "Config cloud non compatibile: chiavi legacy in image_transform_overrides: "
            + ", ".join(present_legacy)
        )

    use_unsplash = bool(image_transform.get("use_unsplash_backgrounds", False))
    if not use_unsplash:
        return []

    raw_dirs = image_transform.get("unsplash_background_dirs", [])
    if not isinstance(raw_dirs, list):
        raise ValueError(
            "Campo non valido in latest.cloud.yaml: image_transform_overrides.unsplash_background_dirs deve essere una lista"
        )

    resolved_dirs: list[Path] = []
    for item in raw_dirs:
        text = str(item).strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved_dirs.append(candidate)

    return resolved_dirs


def _count_images(root: Path) -> int:
    return sum(1 for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def validate_required_real_background_dirs(project_root: Path, required_dirs: list[Path]) -> None:
    """Fail early when config references real background dirs that are missing/unusable."""
    if not required_dirs:
        return

    errors: list[str] = []
    for directory in required_dirs:
        if not _is_relative_to(directory, project_root):
            errors.append(
                "- path fuori repository non supportato nel bundle: "
                f"{directory.as_posix()}"
            )
            continue
        if not directory.exists() or not directory.is_dir():
            errors.append(f"- cartella mancante: {directory.relative_to(project_root).as_posix()}")
            continue
        if _count_images(directory) == 0:
            errors.append(f"- cartella vuota (nessuna immagine): {directory.relative_to(project_root).as_posix()}")

    if errors:
        raise FileNotFoundError(
            "real_background_dirs configurate ma non bundle-ready:\n" + "\n".join(errors)
        )


def resolve_required_hn_sources(project_root: Path) -> list[Path]:
    """Resolve hard_negative_mining.sources from latest cloud config."""
    config_path = project_root / BUNDLED_CLOUD_CONFIG_RELATIVE
    config = _read_yaml_map(config_path)
    hn = config.get("hard_negative_mining", {})
    if not isinstance(hn, dict) or not bool(hn.get("enabled", False)):
        return []

    raw_sources = hn.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("Campo non valido in latest.cloud.yaml: hard_negative_mining.sources deve essere una lista")

    resolved: list[Path] = []
    for item in raw_sources:
        text = str(item).strip()
        if not text:
            continue
        p = Path(text)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        resolved.append(p)
    return resolved


def validate_required_hn_sources(project_root: Path, required_sources: list[Path]) -> None:
    if not required_sources:
        return

    errors: list[str] = []
    for source in required_sources:
        if not _is_relative_to(source, project_root):
            errors.append(f"- sorgente HN fuori repository non supportata nel bundle: {source.as_posix()}")
            continue
        if not source.exists():
            errors.append(f"- sorgente HN mancante: {source.relative_to(project_root).as_posix()}")
            continue
        if source.is_dir() and _count_images(source) == 0:
            # A directory can still be a valid video folder; skip strict image check
            # when any non-directory file exists.
            has_files = any(item.is_file() for item in source.rglob("*"))
            if not has_files:
                errors.append(f"- sorgente HN directory vuota: {source.relative_to(project_root).as_posix()}")

    if errors:
        raise FileNotFoundError("hard_negative_mining.sources non bundle-ready:\n" + "\n".join(errors))


def create_bundle(
    project_root: Path,
    output_path: Path,
    include_dataset: bool,
    include_runs: bool,
    include_latest_generated_dataset: bool,
    strip_notebook_output: bool,
) -> None:
    files_added = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_launchable_config = detect_default_launchable_config(project_root)
    if default_launchable_config != LATEST_CLOUD_CONFIG_RELATIVE.as_posix():
        raise FileNotFoundError(
            "Config cloud mancante: genera prima configs/generated/latest.cloud.yaml con il configuratore usando un runtime cloud."
        )

    required_real_background_dirs = resolve_required_real_background_dirs(project_root)
    validate_required_real_background_dirs(project_root, required_real_background_dirs)
    required_real_background_counts: dict[Path, int] = {path: 0 for path in required_real_background_dirs}
    required_hn_sources = resolve_required_hn_sources(project_root)
    validate_required_hn_sources(project_root, required_hn_sources)
    required_hn_counts: dict[Path, int] = {path: 0 for path in required_hn_sources}

    latest_dataset_dir: Path | None = None
    if include_latest_generated_dataset:
        latest_dataset_dir = detect_latest_generated_dataset_dir(project_root)
        if latest_dataset_dir is None:
            raise FileNotFoundError(
                "Nessun dataset generato trovato in artifacts/local/datasets/. "
                "Genera prima un dataset locale con --skip-training."
            )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in project_root.rglob("*"):
            if candidate.is_dir():
                continue
            relative_path = candidate.relative_to(project_root)

            forced_real_background = None
            for required_dir in required_real_background_dirs:
                if _is_relative_to(candidate, required_dir):
                    forced_real_background = required_dir
                    break

            if forced_real_background is not None:
                archive.write(candidate, arcname=relative_path.as_posix())
                files_added += 1
                required_real_background_counts[forced_real_background] += 1
                continue

            forced_hn_source = None
            for source in required_hn_sources:
                if source.is_dir() and _is_relative_to(candidate, source):
                    forced_hn_source = source
                    break
                if source.is_file() and candidate == source:
                    forced_hn_source = source
                    break

            if forced_hn_source is not None:
                archive.write(candidate, arcname=relative_path.as_posix())
                files_added += 1
                required_hn_counts[forced_hn_source] += 1
                continue

            if latest_dataset_dir is not None and _is_relative_to(candidate, latest_dataset_dir):
                archive.write(candidate, arcname=relative_path.as_posix())
                files_added += 1
                continue
            if should_skip(relative_path, include_dataset, include_runs):
                continue
            if not should_include(relative_path):
                continue
            if strip_notebook_output and candidate.suffix.lower() == ".ipynb":
                archive.writestr(relative_path.as_posix(), strip_notebook_outputs(candidate))
            else:
                archive.write(candidate, arcname=relative_path.as_posix())
            files_added += 1

    missing_in_archive = [
        path for path, count in required_real_background_counts.items() if count == 0
    ]
    if missing_in_archive:
        listed = "\n".join(f"- {path.relative_to(project_root).as_posix()}" for path in missing_in_archive)
        raise RuntimeError(
            "Bundle incompleto: alcune real_background_dirs presenti in config non sono state incluse nello zip:\n"
            f"{listed}"
        )

    missing_hn_in_archive = [path for path, count in required_hn_counts.items() if count == 0]
    if missing_hn_in_archive:
        listed = "\n".join(f"- {path.relative_to(project_root).as_posix()}" for path in missing_hn_in_archive)
        raise RuntimeError(
            "Bundle incompleto: alcune hard_negative_mining.sources presenti in config non sono state incluse nello zip:\n"
            f"{listed}"
        )

    bundle_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundle creato: {output_path}")
    print(f"File inclusi: {files_added}")
    print(f"Dimensione: {bundle_size_mb:.2f} MB")
    if default_launchable_config is not None:
        print(f"Config cloud inclusa nel bundle: {default_launchable_config}")
    if latest_dataset_dir is not None:
        print(f"Dataset locale incluso nel bundle: {latest_dataset_dir.relative_to(project_root).as_posix()}")
    if required_real_background_counts:
        print("Cartelle unsplash_background_dirs incluse:")
        for directory, count in sorted(required_real_background_counts.items(), key=lambda pair: pair[0].as_posix()):
            rel = directory.relative_to(project_root).as_posix()
            print(f"- {rel}: {count} file")
    if required_hn_counts:
        print("Sorgenti hard_negative_mining incluse:")
        for source, count in sorted(required_hn_counts.items(), key=lambda pair: pair[0].as_posix()):
            rel = source.relative_to(project_root).as_posix()
            print(f"- {rel}: {count} file")
    print("Passo successivo: carica lo zip in Colab o in Google Drive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara uno zip del progetto per notebook cloud")
    parser.add_argument("--output", type=str, default="dist/yolo-fire-detector-cloud.zip")
    parser.add_argument("--include-dataset", action="store_true", help="Includi anche la cartella dataset se gia' generata")
    parser.add_argument(
        "--include-latest-generated-dataset",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Se true include nel bundle l'ultimo dataset in artifacts/local/datasets/",
    )
    parser.add_argument("--include-runs", action="store_true", help="Includi checkpoint e risultati precedenti")
    parser.add_argument(
        "--keep-notebook-outputs",
        action="store_true",
        help="Mantieni gli output nei notebook inclusi nello zip (default: output rimossi)",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_path = (project_root / args.output).resolve()
    create_bundle(
        project_root,
        output_path,
        args.include_dataset,
        args.include_runs,
        include_latest_generated_dataset=(args.include_latest_generated_dataset == "true"),
        strip_notebook_output=not args.keep_notebook_outputs,
    )


if __name__ == "__main__":
    main()