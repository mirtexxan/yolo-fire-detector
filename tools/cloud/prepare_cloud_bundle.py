"""Creates a zip bundle of the project for Colab or similar notebook services."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import LATEST_CLOUD_CONFIG_RELATIVE, choose_default_cloud_launchable_config

BUNDLED_CLOUD_CONFIG_RELATIVE = Path("configs") / LATEST_CLOUD_CONFIG_RELATIVE


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

def detect_default_launchable_config(project_root: Path) -> str | None:
    """Return the single cloud config that must be bundled."""
    return choose_default_cloud_launchable_config(project_root / "configs")


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

    bundle_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundle creato: {output_path}")
    print(f"File inclusi: {files_added}")
    print(f"Dimensione: {bundle_size_mb:.2f} MB")
    if default_launchable_config is not None:
        print(f"Config cloud inclusa nel bundle: {default_launchable_config}")
    if latest_dataset_dir is not None:
        print(f"Dataset locale incluso nel bundle: {latest_dataset_dir.relative_to(project_root).as_posix()}")
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