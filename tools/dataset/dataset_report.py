"""Dataset quality report for YOLO-style datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any

import cv2
import yaml


@dataclass
class LabelStats:
    total_images: int = 0
    positive_images: int = 0
    negative_images: int = 0
    total_boxes: int = 0


def _iter_images(split_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(split_dir.glob(pattern))
    return sorted([path for path in files if path.is_file()])


def _parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return rows

    for line in raw.splitlines():
        chunks = line.strip().split()
        if len(chunks) != 5:
            continue
        try:
            class_id = int(float(chunks[0]))
            cx = float(chunks[1])
            cy = float(chunks[2])
            w = float(chunks[3])
            h = float(chunks[4])
        except ValueError:
            continue
        rows.append((class_id, cx, cy, w, h))
    return rows


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return sorted_vals[idx]

    return {
        "count": n,
        "mean": round(statistics.fmean(sorted_vals), 6),
        "median": round(statistics.median(sorted_vals), 6),
        "p05": round(pct(0.05), 6),
        "p95": round(pct(0.95), 6),
        "min": round(sorted_vals[0], 6),
        "max": round(sorted_vals[-1], 6),
    }


def generate_dataset_report(
    dataset_root: str | Path,
    *,
    output_yaml: str | Path | None = None,
    output_json: str | Path | None = None,
    max_image_samples: int = 400,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    labels_root = root / "labels"
    images_root = root / "images"

    if not root.exists():
        raise FileNotFoundError(f"Dataset root non trovata: {root}")

    split_payload: dict[str, Any] = {}
    class_counts: dict[int, int] = {}
    all_box_areas: list[float] = []
    all_box_widths: list[float] = []
    all_box_heights: list[float] = []
    all_box_cx: list[float] = []
    all_box_cy: list[float] = []
    sampled_brightness: list[float] = []

    for split in ("train", "val"):
        split_images_dir = images_root / split
        split_labels_dir = labels_root / split
        image_paths = _iter_images(split_images_dir)

        stats = LabelStats(total_images=len(image_paths))
        split_areas: list[float] = []

        for index, image_path in enumerate(image_paths):
            label_path = split_labels_dir / f"{image_path.stem}.txt"
            boxes = _parse_label_file(label_path)

            if boxes:
                stats.positive_images += 1
            else:
                stats.negative_images += 1

            stats.total_boxes += len(boxes)
            for class_id, cx, cy, w, h in boxes:
                area = max(0.0, w * h)
                split_areas.append(area)
                all_box_areas.append(area)
                all_box_widths.append(max(0.0, w))
                all_box_heights.append(max(0.0, h))
                all_box_cx.append(cx)
                all_box_cy.append(cy)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

            if index < max_image_samples:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sampled_brightness.append(float(gray.mean() / 255.0))

        split_payload[split] = {
            "images": stats.total_images,
            "positive_images": stats.positive_images,
            "negative_images": stats.negative_images,
            "total_boxes": stats.total_boxes,
            "positive_ratio": round((stats.positive_images / stats.total_images), 6) if stats.total_images else None,
            "negative_ratio": round((stats.negative_images / stats.total_images), 6) if stats.total_images else None,
            "bbox_area": _describe(split_areas),
        }

    report = {
        "dataset_root": root.as_posix(),
        "splits": split_payload,
        "global": {
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "bbox_area": _describe(all_box_areas),
            "bbox_width": _describe(all_box_widths),
            "bbox_height": _describe(all_box_heights),
            "bbox_center_x": _describe(all_box_cx),
            "bbox_center_y": _describe(all_box_cy),
            "image_brightness": _describe(sampled_brightness),
        },
    }

    yaml_path = Path(output_yaml) if output_yaml else (root / "dataset_report.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=False)

    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un report QA per dataset YOLO")
    parser.add_argument("--dataset-root", type=str, required=True, help="Cartella root del dataset")
    parser.add_argument("--output-yaml", type=str, default="", help="Path report YAML (default: <dataset_root>/dataset_report.yaml)")
    parser.add_argument("--output-json", type=str, default="", help="Path report JSON opzionale")
    parser.add_argument("--max-image-samples", type=int, default=400, help="Numero massimo immagini per statistiche brightness")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = generate_dataset_report(
        dataset_root=args.dataset_root,
        output_yaml=args.output_yaml or None,
        output_json=args.output_json or None,
        max_image_samples=args.max_image_samples,
    )
    print("Dataset report scritto con successo")
    print(f"dataset_root: {report['dataset_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
