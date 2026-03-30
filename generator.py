"""Dataset generator for the YOLO Fire Detector project."""

import random
import shutil
from collections.abc import Sequence

import cv2
import numpy as np

from settings import ImageTransformSettings, DatasetGenerationSettings
from transformations import (
    generate_random_background,
    augment_fire,
    resize_fire_with_alpha,
    add_shadow,
    alpha_composite,
    add_occlusion_from_background,
    augment_background,
)
from utils import (
    make_output_folders,
    load_fire_image,
    yolo_label_from_bbox,
    save_sample,
    show_demo,
)


# ============================================================
# ================== DATASET GENERATION ======================
# ============================================================


def generate_negative_sample(image_size: int) -> tuple:
    """
    Genera un'immagine negativa: solo sfondo casuale, nessun fuoco.
    In YOLO, per una negativa basta label vuota.
    
    Returns:
        Tuple[np.ndarray, str]: (immagine, label vuota)
    """
    bg = generate_random_background(image_size)

    if ImageTransformSettings.AUGMENT_NEGATIVE_BACKGROUNDS:
        bg = augment_background(bg)

    return bg, ""


def generate_positive_sample(fire_rgba, image_size: int) -> tuple:
    """
    Genera un'immagine positiva:
    - sfondo sintetico
    - fuoco augmentato
    - dimensione casuale
    - posizione casuale
    - ombra
    - occlusione
    
    Args:
        fire_rgba: Immagine del fuoco con alpha channel
        image_size: Dimensione dell'immagine di output
    
    Returns:
        Tuple[np.ndarray, str, Tuple[int, int, int, int]]: (immagine, label YOLO, bbox)
    """
    bg = generate_random_background(image_size)

    fire_aug = augment_fire(fire_rgba)

    # scala del fuoco
    scale = random.uniform(
        DatasetGenerationSettings.FIRE_SCALE_MIN,
        DatasetGenerationSettings.FIRE_SCALE_MAX
    )
    fire_aug = resize_fire_with_alpha(fire_aug, scale)

    fh, fw = fire_aug.shape[:2]

    # sicurezza: se il fuoco fosse più grande della canvas, riduco
    if fw >= image_size or fh >= image_size:
        safe_scale = min((image_size - 2) / max(1, fw), (image_size - 2) / max(1, fh))
        fire_aug = resize_fire_with_alpha(fire_aug, safe_scale)
        fh, fw = fire_aug.shape[:2]

    x = random.randint(0, image_size - fw)
    y = random.randint(0, image_size - fh)

    # ombra prima del compositing
    bg = add_shadow(bg, x, y, fw, fh)

    # compositing con alpha
    composed = alpha_composite(bg, fire_aug, x, y)

    # eventuale occlusione
    composed = add_occlusion_from_background(composed, x, y, fw, fh)

    # ulteriore rumore leggero sull'immagine finale (opzionale)
    if random.random() < 0.15:
        composed = augment_background(composed)

    label = yolo_label_from_bbox(x, y, fw, fh, image_size)
    bbox = (x, y, fw, fh)

    return composed, label, bbox


def normalize_fire_image_paths(
    fire_image_paths: Sequence[str] | None,
) -> list[str]:
    """Normalizza i path delle immagini base del fuoco rimuovendo vuoti e duplicati."""
    normalized: list[str] = []

    if fire_image_paths:
        for candidate in fire_image_paths:
            clean = str(candidate).strip()
            if clean and clean not in normalized:
                normalized.append(clean)

    if not normalized:
        raise ValueError("Devi specificare almeno una immagine base del fuoco")

    return normalized


def generate_dataset(
    dataset_root: str = DatasetGenerationSettings.DATASET_ROOT,
    fire_image_paths: Sequence[str] = DatasetGenerationSettings.FIRE_IMAGE_PATHS,
    num_images: int = DatasetGenerationSettings.NUM_IMAGES,
    image_size: int = DatasetGenerationSettings.IMAGE_SIZE,
    negative_ratio: float = DatasetGenerationSettings.NEGATIVE_RATIO,
    train_split: float = DatasetGenerationSettings.TRAIN_SPLIT,
    demo_mode: bool = DatasetGenerationSettings.DEMO_MODE,
    demo_wait_ms: int = DatasetGenerationSettings.DEMO_WAIT_MS,
    seed: int | None = None,
    clean: bool = False,
) -> dict:
    """Genera un dataset sintetico con parametri espliciti."""

    if not 0.0 <= negative_ratio <= 1.0:
        raise ValueError("negative_ratio deve essere compreso tra 0.0 e 1.0")
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split deve essere compreso tra 0.0 e 1.0")
    if num_images <= 0:
        raise ValueError("num_images deve essere maggiore di zero")
    if image_size <= 0:
        raise ValueError("image_size deve essere maggiore di zero")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if clean:
        shutil.rmtree(dataset_root, ignore_errors=True)

    make_output_folders(dataset_root)

    resolved_fire_paths = normalize_fire_image_paths(fire_image_paths)
    fire_assets = {path: load_fire_image(path) for path in resolved_fire_paths}

    print("Avvio generazione dataset...")
    print(f"Immagini fuoco base: {resolved_fire_paths}")
    print(f"Cartella dataset: {dataset_root}")
    print(f"Numero immagini: {num_images}")
    print(f"Dimensione output: {image_size}x{image_size}")
    print(f"Negative ratio: {negative_ratio}")
    print(f"Train split: {train_split}")
    if seed is not None:
        print(f"Seed: {seed}")

    num_negative = 0
    num_positive = 0
    base_image_usage = {path: 0 for path in resolved_fire_paths}

    for i in range(num_images):

        is_negative = random.random() < negative_ratio

        if is_negative:
            num_negative += 1
            image, label = generate_negative_sample(image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=train_split
            )

            if demo_mode and i % 10 == 0:
                show_demo(image, bbox=None, wait_ms=demo_wait_ms)

        else:
            num_positive += 1
            selected_fire_path = random.choice(resolved_fire_paths)
            selected_fire = fire_assets[selected_fire_path]
            base_image_usage[selected_fire_path] += 1
            image, label, bbox = generate_positive_sample(selected_fire, image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=train_split
            )

            if demo_mode and i % 10 == 0:
                show_demo(image, bbox=bbox, wait_ms=demo_wait_ms)

        if (i + 1) % 100 == 0:
            print(f"Generate {i + 1}/{num_images} immagini")

    cv2.destroyAllWindows()
    print("Dataset generato correttamente.")
    return {
        "dataset_root": dataset_root,
        "num_images": num_images,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "image_size": image_size,
        "negative_ratio": negative_ratio,
        "train_split": train_split,
        "seed": seed,
        "fire_image_paths": resolved_fire_paths,
        "base_image_usage": base_image_usage,
    }
