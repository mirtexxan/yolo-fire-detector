"""Hard negative mining tool.

Runs YOLO inference on a video or image directory and saves every frame
that triggers a detection above ``--conf`` as a hard-negative image for
use in the next training round's hard_negative_background_dirs.

The saved frames can be added directly to
``image_transform_overrides.hard_negative_background_dirs`` in your YAML config:
the generator will use them as backgrounds both for negative samples and,
when compositing, for positive samples (fire marker on top of a confusing scene).

Usage:
    # From drone video (processes 1 frame every 5)
    python tools/dataset/collect_hard_negatives.py --source drone_footage.mp4

    # From a folder of ground images
    python tools/dataset/collect_hard_negatives.py --source path/to/images/ --conf 0.15

    # Custom output and limit
    python tools/dataset/collect_hard_negatives.py \\
        --source footage.mp4 \\
        --output artifacts/local/hard_negatives/drone-round1 \\
        --conf 0.20 --max-samples 300 --stride 8
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
import re

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _next_sample_index(output_dir: Path) -> int:
    """Return the next available numeric index for hn_<idx>_*.jpg files."""
    pattern = re.compile(r"^hn_(\d+)_")
    max_index = -1
    for item in output_dir.glob("hn_*_*.jpg"):
        match = pattern.match(item.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _image_sha1(frame: "cv2.typing.MatLike") -> str:
    """Compute a deterministic hash of pixel content for exact-duplicate filtering."""
    return hashlib.sha1(frame.tobytes()).hexdigest()


def _load_existing_hashes(output_dir: Path) -> set[str]:
    """Index existing hard negatives to support append-mode deduplication."""
    hashes: set[str] = set()
    if not output_dir.exists():
        return hashes

    for path in output_dir.glob("hn_*_*.jpg"):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        hashes.add(_image_sha1(frame))
    return hashes


def _resolve_model(model_arg: str) -> Path:
    """Resolve model path: 'latest'/'auto' -> latest.yaml pointer -> .pt file."""
    model_arg = model_arg.strip()
    if model_arg.lower() in {"", "latest", "auto", "default"}:
        candidates = [
            PROJECT_ROOT / "artifacts" / "local" / "exports" / "latest.yaml",
            *sorted(PROJECT_ROOT.glob("artifacts/**/exports/latest.yaml")),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh)
                if not isinstance(payload, dict):
                    continue
                model_path_str = payload.get("model_path", "")
                if not model_path_str:
                    continue
                resolved = Path(str(model_path_str))
                if not resolved.is_absolute():
                    resolved = (candidate.parent.parent / resolved).resolve()
                if resolved.exists():
                    return resolved
            except (OSError, yaml.YAMLError):
                continue
        raise FileNotFoundError(
            "Nessun modello trovato tramite latest.yaml.\n"
            "Specifica il percorso con --weights path/to/model.pt"
        )
    p = Path(model_arg)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Modello non trovato: {p}")
    return p


def _iter_image_paths(source: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(source.rglob(f"*{ext}"))
        paths.extend(source.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def _infer_labels_dir(images_dir: Path) -> Path | None:
    """Try to resolve the sibling labels/ folder from a standard YOLO images/ dir."""
    parts = images_dir.parts
    # Replace the last occurrence of 'images' in the path with 'labels'
    for i in reversed(range(len(parts))):
        if parts[i].lower() == "images":
            labels_dir = Path(*parts[:i], "labels", *parts[i + 1:])
            return labels_dir
    # Fallback: sibling 'labels' folder at the same level
    return images_dir.parent / "labels"


def _is_confirmed_negative(img_path: Path, labels_dir: Path) -> bool:
    """Return True if the image has no corresponding non-empty YOLO label file."""
    label_path = labels_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        return True  # no label file -> negative
    content = label_path.read_text(encoding="utf-8").strip()
    return content == ""  # empty file -> negative


def collect_hard_negatives(
    source: Path,
    model_path: Path,
    conf_threshold: float,
    output_dir: Path,
    max_samples: int,
    stride: int,
    filter_negatives_only: bool = False,
    deduplicate: bool = True,
) -> None:
    from ultralytics import YOLO  # deferred: heavy import

    print(f"Modello        : {model_path}")
    print(f"Sorgente       : {source}")
    print(f"Conf minima FP : {conf_threshold}")
    print(f"Output         : {output_dir}")
    print(f"Max campioni   : {max_samples}")
    if filter_negatives_only:
        print(f"Modalita'      : solo immagini senza label YOLO (filter-negatives-only)")
    print()

    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    start_index = _next_sample_index(output_dir)
    known_hashes = _load_existing_hashes(output_dir) if deduplicate else set()

    saved = 0
    frames_processed = 0
    skipped_positives = 0
    skipped_duplicates = 0
    total_detections = 0
    max_conf_seen = 0.0

    if source.is_dir() or _is_image_file(source):
        image_paths = [source] if _is_image_file(source) else _iter_image_paths(source)
        print(f"Immagini trovate: {len(image_paths)}")

        labels_dir: Path | None = None
        if filter_negatives_only:
            labels_probe = source.parent if _is_image_file(source) else source
            labels_dir = _infer_labels_dir(labels_probe)
            if labels_dir and labels_dir.exists():
                print(f"Labels dir      : {labels_dir}")
            else:
                print(f"⚠️  Labels dir non trovata ({labels_dir}) — il filtro non verrà applicato.")
                labels_dir = None

        for img_path in image_paths:
            if saved >= max_samples:
                break
            if filter_negatives_only and labels_dir is not None:
                if not _is_confirmed_negative(img_path, labels_dir):
                    skipped_positives += 1
                    continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frames_processed += 1
            results = model(frame, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                if deduplicate:
                    img_hash = _image_sha1(frame)
                    if img_hash in known_hashes:
                        skipped_duplicates += 1
                        continue
                    known_hashes.add(img_hash)
                confs = boxes.conf.cpu().numpy()
                total_detections += len(confs)
                max_conf_seen = max(max_conf_seen, float(confs.max()))
                out_path = output_dir / f"hn_{start_index + saved:05d}_{img_path.stem}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
        if skipped_positives:
            print(f"  {skipped_positives} immagini saltate (label positiva presente)")

    else:
        # video
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire il video: {source}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Frames totali  : {total_frames}, FPS: {fps:.1f}, stride: ogni {stride}")

        frame_idx = 0
        while saved < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            frames_processed += 1

            results = model(frame, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                if deduplicate:
                    img_hash = _image_sha1(frame)
                    if img_hash in known_hashes:
                        skipped_duplicates += 1
                        continue
                    known_hashes.add(img_hash)
                confs = boxes.conf.cpu().numpy()
                total_detections += len(confs)
                max_conf_seen = max(max_conf_seen, float(confs.max()))
                out_path = output_dir / f"hn_{start_index + saved:05d}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                if saved % 20 == 0:
                    print(f"  {saved} hard negative salvati...")

        cap.release()

    print()
    print("=" * 56)
    print(f"Frames analizzati      : {frames_processed}")
    if skipped_positives:
        print(f"Saltati (positivi)     : {skipped_positives}")
    if deduplicate:
        print(f"Saltati (duplicati)    : {skipped_duplicates}")
    print(f"Hard negative salvati  : {saved}")
    print(f"Detections FP trovate  : {total_detections}")
    if max_conf_seen > 0:
        print(f"Max confidenza FP      : {max_conf_seen:.3f}")
    print(f"Output                 : {output_dir}")

    if saved > 0:
        try:
            display_path = output_dir.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = output_dir.as_posix()
        print()
        print("Aggiungi a latest.local.yaml per usare nel prossimo round:")
        print("  image_transform_overrides:")
        print("    hard_negative_background_dirs:")
        print(f"    - {output_dir.as_posix()}")
        print("    # ... aggiungi gli altri domini esistenti sotto")
    else:
        print()
        print("Nessun hard negative trovato con questa soglia.")
        print("Prova ad abbassare --conf (es. 0.10) o usa un modello diverso.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Raccoglie hard negative frame da video o cartella immagini.\n"
            "I frame salvati possono essere aggiunti a hard_negative_background_dirs nel YAML config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video (mp4/avi/...), cartella immagini o singola immagine da analizzare",
    )
    parser.add_argument(
        "--weights",
        default="latest",
        help="Percorso modello .pt oppure 'latest' (default: latest)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Soglia confidenza minima per salvare il frame come FP (default: 0.15)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Cartella di output (default: artifacts/local/hard_negatives/<source-name>/)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Numero massimo di hard negative da salvare (default: 500)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Video: analizza 1 frame ogni N (default: 5)",
    )
    parser.add_argument(
        "--filter-negatives-only",
        action="store_true",
        default=False,
        help=(
            "Solo per cartelle con label YOLO (layout standard images/ + labels/). "
            "Salta automaticamente le immagini con label non vuota (positivi noti) "
            "e analizza solo le immagini senza label o con label vuota (negative confermate). "
            "Utile ad es. con la cartella val/ di un dataset reale gia' annotato."
        ),
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        default=False,
        help="Disattiva deduplica per contenuto (default: deduplica attiva)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERRORE: sorgente non trovata: {source}")
        sys.exit(1)

    try:
        model_path = _resolve_model(args.weights)
    except FileNotFoundError as exc:
        print(f"ERRORE: {exc}")
        sys.exit(1)

    if args.output.strip():
        output_dir = Path(args.output.strip())
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "local"
            / "hard_negatives"
            / source.stem
        )

    collect_hard_negatives(
        source=source,
        model_path=model_path,
        conf_threshold=args.conf,
        output_dir=output_dir,
        max_samples=args.max_samples,
        stride=args.stride,
        filter_negatives_only=args.filter_negatives_only,
        deduplicate=not args.no_deduplicate,
    )


if __name__ == "__main__":
    main()
