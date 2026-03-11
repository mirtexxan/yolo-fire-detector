import cv2
import glob
import os
import random
import numpy as np

# ============================================================
# SETTINGS VIEWER
# ============================================================

SETTINGS = {
    "dataset_root": "dataset",
    "split": "train",       # "train" oppure "val"
    "num_samples": 9,       # quante immagini mostrare
    "thumb_size": 280,      # dimensione miniatura
    "draw_title": True,     # scrive il nome file sopra
}


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
        # riempi eventuali celle mancanti con immagini nere
        while len(current_row) < cols:
            current_row.append(np.zeros_like(images[0]))
        rows.append(cv2.hconcat(current_row))

    grid = cv2.vconcat(rows)
    return grid


def main():
    import numpy as np

    img_paths = load_sample_paths(
        SETTINGS["dataset_root"],
        SETTINGS["split"]
    )

    if len(img_paths) == 0:
        print("Nessuna immagine trovata.")
        return

    sample_count = min(SETTINGS["num_samples"], len(img_paths))
    chosen = random.sample(img_paths, sample_count)

    thumbs = []

    for img_path in chosen:
        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = corresponding_label_path(img_path)
        img = draw_yolo_bbox(img, label_path)
        img = cv2.resize(img, (SETTINGS["thumb_size"], SETTINGS["thumb_size"]))

        if SETTINGS["draw_title"]:
            img = add_filename_title(img, os.path.basename(img_path))

        thumbs.append(img)

    if not thumbs:
        print("Impossibile costruire il viewer: nessuna immagine valida.")
        return

    # Se il numero di sample non è multiplo di 3, riempiamo le celle mancanti
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