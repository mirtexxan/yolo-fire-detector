"""Creates a zip bundle of the project for Colab or similar notebook services."""

import argparse
from pathlib import Path
import zipfile


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    ".vscode",
    ".ipynb_checkpoints",
    "__pycache__",
    "artifacts",
    "dataset",
    "detections",
    "fire_detector_runs",
    "runs",
    "dist",
    "env",
    "venv",
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


def should_skip(path: Path, include_dataset: bool, include_runs: bool) -> bool:
    parts = set(path.parts)
    excludes = set(DEFAULT_EXCLUDES)
    if include_dataset:
        excludes.discard("dataset")
    if include_runs:
        excludes.discard("fire_detector_runs")
        excludes.discard("runs")
    return any(part in excludes for part in parts)


def should_include(relative_path: Path) -> bool:
    if relative_path.name in DEFAULT_INCLUDE_NAMES or relative_path.suffix.lower() in DEFAULT_INCLUDE_SUFFIXES:
        return True
    if relative_path.suffix.lower() == ".pt" and len(relative_path.parts) == 1:
        return True
    return "base_fire_images" in relative_path.parts and relative_path.suffix.lower() == ".png"


def create_bundle(project_root: Path, output_path: Path, include_dataset: bool, include_runs: bool) -> None:
    files_added = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in project_root.rglob("*"):
            if candidate.is_dir():
                continue
            relative_path = candidate.relative_to(project_root)
            if should_skip(relative_path, include_dataset, include_runs):
                continue
            if not should_include(relative_path):
                continue
            archive.write(candidate, arcname=str(relative_path))
            files_added += 1

    bundle_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundle creato: {output_path}")
    print(f"File inclusi: {files_added}")
    print(f"Dimensione: {bundle_size_mb:.2f} MB")
    print("Passo successivo: carica lo zip in Colab o in Google Drive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara uno zip del progetto per notebook cloud")
    parser.add_argument("--output", type=str, default="dist/yolo-fire-detector-cloud.zip")
    parser.add_argument("--include-dataset", action="store_true", help="Includi anche la cartella dataset se gia' generata")
    parser.add_argument("--include-runs", action="store_true", help="Includi checkpoint e risultati precedenti")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_path = (project_root / args.output).resolve()
    create_bundle(project_root, output_path, args.include_dataset, args.include_runs)


if __name__ == "__main__":
    main()
