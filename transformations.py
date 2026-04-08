"""
Image transformation and augmentation functions

Contains all functions for:
- Synthetic background generation (domain randomization)
- Fire augmentation (geometric and photometric)
- Compositing (alpha blending, shadows, occlusions)
"""

import cv2
import numpy as np
import random
from pathlib import Path

from settings import ImageTransformSettings


REAL_BG_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")

BACKGROUND_SOURCE_COUNTS = {
    "hard_negative": 0,
    "unsplash": 0,
    "synthetic": 0,
}
_BACKGROUND_CALL_DEPTH = 0


def reset_background_source_counters() -> None:
    for key in BACKGROUND_SOURCE_COUNTS:
        BACKGROUND_SOURCE_COUNTS[key] = 0


def get_background_source_counters() -> dict[str, int]:
    return dict(BACKGROUND_SOURCE_COUNTS)


def _iter_background_paths(roots: list[str]) -> list[Path]:
    if not isinstance(roots, list) or not roots:
        return []

    paths: list[Path] = []
    for root in roots:
        base = Path(str(root)).expanduser()
        if not base.exists():
            continue
        for pattern in REAL_BG_PATTERNS:
            paths.extend(base.rglob(pattern))

    return [path for path in paths if path.is_file()]


def _sample_background_from_dirs(size: int, roots: list[str]) -> np.ndarray | None:
    candidates = _iter_background_paths(roots)
    if not candidates:
        return None

    selected = random.choice(candidates)
    image = cv2.imread(str(selected), cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]
    if h < 2 or w < 2:
        return None

    # Crop quadrato casuale per mantenere varieta' di inquadrature.
    side = min(h, w)
    y = random.randint(0, h - side)
    x = random.randint(0, w - side)
    crop = image[y:y + side, x:x + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ============================================================
# ================== SYNTHETIC BACKGROUNDS ===================
# ============================================================


def random_color() -> tuple[int, int, int]:
    """Restituisce un colore BGR casuale."""
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def background_flat_color(size: int) -> np.ndarray:
    """Sfondo a colore uniforme."""
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    bg[:] = random_color()
    return bg


def background_noise(size: int) -> np.ndarray:
    """Sfondo a rumore casuale."""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def background_gradient(size: int) -> np.ndarray:
    """
    Sfondo con gradiente tra due colori.
    Può essere orizzontale, verticale o diagonale.
    """
    c1 = np.array(random_color(), dtype=np.float32)
    c2 = np.array(random_color(), dtype=np.float32)

    mode = random.choice(["horizontal", "vertical", "diagonal"])

    if mode == "horizontal":
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
        t = np.repeat(t, size, axis=0)
    elif mode == "vertical":
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        t = np.repeat(t, size, axis=1)
    else:
        coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        t = (xx + yy) * 0.5

    gradient = (1.0 - t[..., None]) * c1 + t[..., None] * c2
    return gradient.astype(np.uint8)


def background_blobs(size: int) -> np.ndarray:
    """
    Sfondo con 'macchie' colorate casuali, utile per simulare texture naturali.
    """
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    bg[:] = random_color()

    num_blobs = random.randint(20, 60)

    for _ in range(num_blobs):
        center = (
            random.randint(0, size - 1),
            random.randint(0, size - 1)
        )
        radius = random.randint(size // 30, size // 8)
        color = random_color()
        cv2.circle(bg, center, radius, color, -1)

    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=random.uniform(3, 10))
    return bg


def background_lines(size: int) -> np.ndarray:
    """
    Sfondo con linee casuali, utile per introdurre pattern più strutturati.
    """
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    bg[:] = random_color()

    num_lines = random.randint(20, 80)

    for _ in range(num_lines):
        x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
        x2, y2 = random.randint(0, size - 1), random.randint(0, size - 1)
        thickness = random.randint(1, 4)
        color = random_color()
        cv2.line(bg, (x1, y1), (x2, y2), color, thickness)

    if random.random() < 0.5:
        bg = cv2.GaussianBlur(bg, (5, 5), 0)

    return bg


def background_checker(size: int) -> np.ndarray:
    """
    Sfondo a scacchiera: pattern artificiale forte.
    Serve a evitare che la rete si affidi a contesti troppo regolari.
    """
    bg = np.zeros((size, size, 3), dtype=np.uint8)

    block = random.randint(20, 80)
    c1 = random_color()
    c2 = random_color()

    for y in range(0, size, block):
        for x in range(0, size, block):
            color = c1 if ((x // block) + (y // block)) % 2 == 0 else c2
            cv2.rectangle(bg, (x, y), (x + block, y + block), color, -1)

    if random.random() < 0.6:
        bg = cv2.GaussianBlur(bg, (5, 5), 0)

    return bg


def background_mixed(size: int) -> np.ndarray:
    """
    Sfondo combinato: miscela due sfondi casuali.
    Questo aumenta ulteriormente la variabilità visiva.
    """
    bg1 = generate_random_background(size)
    bg2 = generate_random_background(size)
    alpha = random.uniform(0.3, 0.7)
    mixed = cv2.addWeighted(bg1, alpha, bg2, 1 - alpha, 0)
    return mixed


def generate_random_background(size: int) -> np.ndarray:
    """
    Seleziona casualmente uno dei tipi di sfondo sintetico.
    """
    global _BACKGROUND_CALL_DEPTH
    _BACKGROUND_CALL_DEPTH += 1
    is_top_level = _BACKGROUND_CALL_DEPTH == 1
    try:
        use_unsplash = bool(getattr(ImageTransformSettings, "USE_UNSPLASH_BACKGROUNDS", False))
        unsplash_prob = float(getattr(ImageTransformSettings, "UNSPLASH_BACKGROUND_PROB", 0.65))
        unsplash_dirs = getattr(ImageTransformSettings, "UNSPLASH_BACKGROUND_DIRS", [])

        use_hn = bool(getattr(ImageTransformSettings, "USE_HARD_NEGATIVE_BACKGROUNDS", False))
        hn_prob = float(getattr(ImageTransformSettings, "HARD_NEGATIVE_BACKGROUND_PROB", 0.65))
        hn_dirs = getattr(ImageTransformSettings, "HARD_NEGATIVE_BACKGROUND_DIRS", [])

        if use_hn and random.random() < max(0.0, min(1.0, hn_prob)):
            sampled = _sample_background_from_dirs(size, hn_dirs)
            if sampled is not None:
                if is_top_level:
                    BACKGROUND_SOURCE_COUNTS["hard_negative"] += 1
                return sampled

        if use_unsplash and random.random() < max(0.0, min(1.0, unsplash_prob)):
            sampled = _sample_background_from_dirs(size, unsplash_dirs)
            if sampled is not None:
                if is_top_level:
                    BACKGROUND_SOURCE_COUNTS["unsplash"] += 1
                return sampled

        generators = [
            background_flat_color,
            background_noise,
            background_gradient,
            background_blobs,
            background_lines,
            background_checker,
        ]

        # ogni tanto crea uno sfondo misto più complesso
        if random.random() < 0.20:
            out = background_mixed(size)
            if is_top_level:
                BACKGROUND_SOURCE_COUNTS["synthetic"] += 1
            return out

        generator = random.choice(generators)
        out = generator(size)
        if is_top_level:
            BACKGROUND_SOURCE_COUNTS["synthetic"] += 1
        return out
    finally:
        _BACKGROUND_CALL_DEPTH -= 1


# ============================================================
# ================ FIRE AUGMENTATIONS: GEOMETRY ==============
# ============================================================


def rotate_image_keep_canvas(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Ruota l'immagine attorno al centro mantenendo la stessa dimensione del canvas.
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
    )
    return rotated


def perspective_warp_keep_canvas(image: np.ndarray, shift: int) -> np.ndarray:
    """
    Applica una distorsione prospettica casuale.
    È la simulazione più semplice della rotazione sui 3 assi / cambio di vista.
    """
    h, w = image.shape[:2]

    pts1 = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ], dtype=np.float32)

    pts2 = np.array([
        [random.randint(0, shift), random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h - 1 - random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), h - 1 - random.randint(0, shift)]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    warped = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
    )
    return warped


# ============================================================
# ============= FIRE AUGMENTATIONS: PHOTOMETRIC ==============
# ============================================================


def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """
    Modifica contrasto (alpha) e luminosità (beta).
    Funziona sui primi 3 canali; l'eventuale alpha viene preservato.
    """
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        a = image[:, :, 3]
        bgr_out = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
        return np.dstack([bgr_out, a])

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def color_shift_hsv(image: np.ndarray, hue_delta: int) -> np.ndarray:
    """
    Piccola variazione del colore in HSV.
    Anche se il fuoco dovrebbe restare 'simile', questa randomizzazione aiuta.
    """
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        a = image[:, :, 3]
    else:
        bgr = image
        a = None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_delta) % 180
    bgr_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if a is not None:
        return np.dstack([bgr_out, a])
    return bgr_out


# ============================================================
# =============== FIRE AUGMENTATIONS: BLUR & NOISE ===========
# ============================================================


def add_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aggiunge blur gaussiano."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        a = image[:, :, 3]
        bgr_out = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0)
        return np.dstack([bgr_out, a])

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Motion blur semplice con direzione casuale:
    - orizzontale
    - verticale
    - diagonale principale
    - diagonale secondaria
    """
    direction = random.choice(["h", "v", "d1", "d2"])

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    if direction == "h":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "v":
        kernel[:, kernel_size // 2] = 1.0
    elif direction == "d1":
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)

    kernel /= kernel.sum()

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        a = image[:, :, 3]
        bgr_out = cv2.filter2D(bgr, -1, kernel)
        return np.dstack([bgr_out, a])

    return cv2.filter2D(image, -1, kernel)


def add_noise(image: np.ndarray, noise_level: int) -> np.ndarray:
    """
    Aggiunge rumore uniforme.
    """
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        a = image[:, :, 3]
        noise = np.random.randint(0, noise_level + 1, bgr.shape, dtype=np.uint8)
        bgr_out = cv2.add(bgr, noise)
        return np.dstack([bgr_out, a])

    noise = np.random.randint(0, noise_level + 1, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def augment_fire(fire: np.ndarray) -> np.ndarray:
    """
    Applica una sequenza di augmentazioni al fuoco.
    L'ordine è pensato per introdurre variazione geometrica e fotometrica.
    """
    out = fire.copy()

    # Rotazione 2D
    angle = random.uniform(
        ImageTransformSettings.ROTATION_DEG_MIN,
        ImageTransformSettings.ROTATION_DEG_MAX
    )
    out = rotate_image_keep_canvas(out, angle)

    # Prospettiva: simula variazioni di viewpoint / pseudo-3D
    out = perspective_warp_keep_canvas(out, ImageTransformSettings.PERSPECTIVE_SHIFT)

    # Contrasto + luminosità
    alpha = random.uniform(
        ImageTransformSettings.CONTRAST_ALPHA_MIN,
        ImageTransformSettings.CONTRAST_ALPHA_MAX
    )
    beta = random.randint(
        ImageTransformSettings.BRIGHTNESS_BETA_MIN,
        ImageTransformSettings.BRIGHTNESS_BETA_MAX
    )
    out = adjust_brightness_contrast(out, alpha, beta)

    # Piccolo shift di colore
    if ImageTransformSettings.ENABLE_COLOR_SHIFT and random.random() < ImageTransformSettings.COLOR_SHIFT_PROB:
        hue_delta = random.randint(
            -ImageTransformSettings.COLOR_SHIFT_HUE_MAX,
            ImageTransformSettings.COLOR_SHIFT_HUE_MAX
        )
        out = color_shift_hsv(out, hue_delta)

    # Blur gaussiano
    if random.random() < ImageTransformSettings.GAUSSIAN_BLUR_PROB:
        k = random.choice(ImageTransformSettings.GAUSSIAN_BLUR_KERNEL_CHOICES)
        out = add_gaussian_blur(out, k)

    # Motion blur
    if random.random() < ImageTransformSettings.MOTION_BLUR_PROB:
        k = random.choice(ImageTransformSettings.MOTION_BLUR_KERNEL_CHOICES)
        out = add_motion_blur(out, k)

    # Rumore
    if random.random() < ImageTransformSettings.NOISE_PROB:
        noise_level = random.randint(
            ImageTransformSettings.NOISE_LEVEL_MIN,
            ImageTransformSettings.NOISE_LEVEL_MAX
        )
        out = add_noise(out, noise_level)

    return out


# ============================================================
# ==================== COMPOSITING FUNCTIONS =================
# ============================================================


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Se l'immagine ha alpha, separa canali BGR e maschera alpha.
    Se non ha alpha, crea alpha pieno.
    """
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)

    return bgr, alpha


def resize_fire_with_alpha(fire: np.ndarray, scale: float) -> np.ndarray:
    """
    Ridimensiona il fuoco mantenendo i canali.
    """
    h, w = fire.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(fire, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def add_shadow(background: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Disegna un'ombra morbida ellittica sotto il fuoco.
    È una semplificazione, ma aiuta il realismo.
    """
    if random.random() > ImageTransformSettings.SHADOW_PROB:
        return background

    shadow_alpha = random.uniform(
        ImageTransformSettings.SHADOW_ALPHA_MIN,
        ImageTransformSettings.SHADOW_ALPHA_MAX
    )

    shadow = background.copy()
    overlay = shadow.copy()

    center = (x + w // 2, y + h)
    axes = (max(3, w // 2), max(3, h // 5))

    cv2.ellipse(
        overlay,
        center,
        axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(0, 0, 0),
        thickness=-1
    )

    # blur dell'ombra per renderla meno netta
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=6, sigmaY=6)

    shadowed = cv2.addWeighted(
        overlay, shadow_alpha,
        shadow, 1 - shadow_alpha,
        0
    )
    return shadowed


def alpha_composite(background: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Sovrappone il foreground con alpha sul background in posizione (x, y).
    """
    out = background.copy()

    fg_bgr, fg_alpha = split_alpha(fg_rgba)
    h, w = fg_bgr.shape[:2]

    alpha = (fg_alpha.astype(np.float32) / 255.0)[:, :, None]

    roi = out[y:y + h, x:x + w].astype(np.float32)
    fg = fg_bgr.astype(np.float32)

    composed = alpha * fg + (1 - alpha) * roi
    out[y:y + h, x:x + w] = composed.astype(np.uint8)

    return out


def add_occlusion_from_background(
    image: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int
) -> np.ndarray:
    """
    Simula un' occlusione prendendo una patch dello sfondo
    e sovrapponendola parzialmente al fuoco.

    Questo è molto utile perché insegna al modello a riconoscere
    l'oggetto anche se è parzialmente nascosto.
    """
    if random.random() > ImageTransformSettings.OCCLUSION_PROB:
        return image

    out = image.copy()
    H, W = out.shape[:2]

    coverage = random.uniform(
        ImageTransformSettings.OCCLUSION_COVERAGE_MIN,
        ImageTransformSettings.OCCLUSION_COVERAGE_MAX
    )

    occ_w = max(3, int(bbox_w * coverage))
    occ_h = max(3, int(bbox_h * coverage))

    # posizione dell'occlusione: deve cadere in parte sopra il bbox del fuoco
    occ_x = random.randint(
        max(0, bbox_x),
        min(W - occ_w, bbox_x + max(0, bbox_w - occ_w))
    )
    occ_y = random.randint(
        max(0, bbox_y),
        min(H - occ_h, bbox_y + max(0, bbox_h - occ_h))
    )

    # patch presa da una zona casuale dell'immagine
    src_x = random.randint(0, W - occ_w)
    src_y = random.randint(0, H - occ_h)

    patch = out[src_y:src_y + occ_h, src_x:src_x + occ_w].copy()

    # leggera sfumatura per evitare bordi troppo netti
    if occ_w >= 3 and occ_h >= 3:
        patch = cv2.GaussianBlur(patch, (3, 3), 0)

    out[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = patch
    return out


# ============================================================
# ================== BACKGROUND AUGMENTATION =================
# ============================================================


def augment_background(background: np.ndarray) -> np.ndarray:
    """
    Applica lievi trasformazioni allo sfondo negativo, per variarlo ulteriormente.
    """
    out = background.copy()

    # luminosità/contrasto leggero
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # piccolo blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # rumore
    if random.random() < 0.3:
        noise_level = random.randint(5, 25)
        noise = np.random.randint(0, noise_level + 1, out.shape, dtype=np.uint8)
        out = cv2.add(out, noise)

    return out
