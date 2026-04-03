import argparse
import cv2
import glob
import os
from pathlib import Path
import random
import sys
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from settings import ViewerSettings

_DATASETS_DIR = PROJECT_ROOT / "artifacts" / "local" / "datasets"


def _fuzzy_resolve_dataset(name: str) -> Path:
    """Find a dataset folder by name (or prefix) inside artifacts/local/datasets/.

    Resolution order:
    1. Treat ``name`` as a direct path (absolute or relative to project root).
    2. Exact folder-name match inside artifacts/local/datasets/.
    3. ``startswith`` match (first alphabetically).

    Raises FileNotFoundError if nothing matches.
    """
    # 1. Direct path
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return candidate

    candidates = sorted(d for d in _DATASETS_DIR.iterdir() if d.is_dir())

    # 2. Exact name
    for d in candidates:
        if d.name == name:
            print(f"ℹ️  Dataset trovato in artifacts/local/datasets/: {d.name}")
            return d

    # 3. startswith
    for d in candidates:
        if d.name.startswith(name):
            print(f"ℹ️  Dataset trovato per prefisso in artifacts/local/datasets/: {d.name}")
            return d

    available = [d.name for d in candidates]
    raise FileNotFoundError(
        f"Dataset non trovato: '{name}'.\n"
        f"Dataset disponibili: {available or '(nessuno)'}"
    )


def load_sample_paths(dataset_root: str, split: str) -> list[str]:
    """
    Carica tutti i percorsi delle immagini nel dataset scelto.
    """
    pattern = os.path.join(dataset_root, "images", split, "*.jpg")
    return glob.glob(pattern)


def corresponding_label_path(img_path: str) -> str:
    """
    Converte il path immagine nel path label corrispondente.
    Esempio:
    dataset/images/train/img_00001.jpg
    -> dataset/labels/train/img_00001.txt
    """
    return img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")


def draw_yolo_bbox(image, label_path):
    """
    Disegna le bounding box lette dal file YOLO.
    Se il file è vuoto, l'immagine è negativa.
    """
    out = image.copy()
    h, w = out.shape[:2]

    if not os.path.exists(label_path):
        return out

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id, cx, cy, bw, bh = parts
        cx = float(cx)
        cy = float(cy)
        bw = float(bw)
        bh = float(bh)

        box_w = int(bw * w)
        box_h = int(bh * h)
        x = int(cx * w - box_w / 2)
        y = int(cy * h - box_h / 2)

        cv2.rectangle(out, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"class {class_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return out


def add_filename_title(image, filename):
    """
    Aggiunge una piccola barra superiore con il nome del file.
    """
    h, w = image.shape[:2]
    title_bar_h = 28

    canvas = cv2.copyMakeBorder(
        image,
        title_bar_h, 0, 0, 0,
        borderType=cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    cv2.putText(
        canvas,
        filename,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return canvas


def build_grid(images, cols=3):
    """
    Costruisce una griglia 3x3 (o più righe se necessario).
    """
    if not images:
        return None

    rows = []
    current_row = []

    for img in images:
        current_row.append(img)
        if len(current_row) == cols:
            rows.append(cv2.hconcat(current_row))
            current_row = []

    if current_row:
        while len(current_row) < cols:
            current_row.append(np.zeros_like(images[0]))
        rows.append(cv2.hconcat(current_row))

    grid = cv2.vconcat(rows)
    return grid


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Visualizza campioni di un dataset YOLO")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Cartella del dataset da visualizzare. Accetta: percorso assoluto, "
            "percorso relativo al root del progetto, nome esatto in artifacts/local/datasets/, "
            "oppure prefisso (startswith)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Split da visualizzare (default: usa ViewerSettings.SPLIT)",
    )
    args = parser.parse_args()

    if args.dataset is not None:
        ViewerSettings.DATASET_ROOT = _fuzzy_resolve_dataset(args.dataset).as_posix()
    if args.split is not None:
        ViewerSettings.SPLIT = args.split

    img_paths = load_sample_paths(
        ViewerSettings.DATASET_ROOT,
        ViewerSettings.SPLIT
    )

    if len(img_paths) == 0:
        print("Nessuna immagine trovata.")
        return

    sample_count = min(ViewerSettings.NUM_SAMPLES, len(img_paths))
    chosen = random.sample(img_paths, sample_count)

    thumbs = []

    for img_path in chosen:
        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = corresponding_label_path(img_path)
        img = draw_yolo_bbox(img, label_path)
        img = cv2.resize(img, (ViewerSettings.THUMB_SIZE, ViewerSettings.THUMB_SIZE))

        if ViewerSettings.DRAW_TITLE:
            img = add_filename_title(img, os.path.basename(img_path))

        thumbs.append(img)

    if not thumbs:
        print("Impossibile costruire il viewer: nessuna immagine valida.")
        return

    h, w = thumbs[0].shape[:2]
    while len(thumbs) % 3 != 0:
        thumbs.append(np.zeros((h, w, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(thumbs), 3):
        rows.append(cv2.hconcat(thumbs[i:i+3]))

    grid = cv2.vconcat(rows)

    cv2.imshow("Dataset Viewer", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()