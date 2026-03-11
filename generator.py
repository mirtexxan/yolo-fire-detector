import cv2
import numpy as np
import os
import random
import math

# ============================================================
# SETTINGS PRINCIPALI
# ============================================================
# Qui puoi personalizzare il comportamento del generatore.
# ============================================================

SETTINGS = {
    # ---------- PATH ----------
    # Percorso configurabile dell'immagine del fuoco.
    # Esempio Windows:
    # r"C:\Users\Nome\Desktop\fire.png"
    "fire_path": r"fire.png",

    # Cartella di output dataset YOLO
    "dataset_root": "dataset",

    # ---------- DATASET ----------
    "num_images": 2000,
    "train_split": 0.8,          # 80% train, 20% val
    "negative_ratio": 0.35,      # percentuale immagini senza fuoco

    # ---------- DIMENSIONI ----------
    "image_size": 640,           # dimensione finale immagini quadrate

    # ---------- FUOCO: DIMENSIONE RELATIVA ----------
    # Percentuale rispetto alla dimensione originale del PNG.
    # Valori bassi = fuoco lontano, valori alti = fuoco vicino.
    "fire_scale_min": 0.05,
    "fire_scale_max": 0.50,

    # ---------- ROTAZIONI / GEOMETRIA ----------
    "rotation_deg_min": -180,
    "rotation_deg_max": 180,
    "perspective_shift": 120,    # quanto può essere forte la distorsione prospettica

    # ---------- LUMINOSITÀ / CONTRASTO ----------
    "brightness_beta_min": -50,
    "brightness_beta_max": 50,
    "contrast_alpha_min": 0.60,
    "contrast_alpha_max": 1.40,

    # ---------- COLORE ----------
    "enable_color_shift": True,
    "color_shift_prob": 0.35,
    "color_shift_hue_max": 20,

    # ---------- BLUR / RUMORE ----------
    "motion_blur_prob": 0.40,
    "motion_blur_kernel_choices": [5, 7, 9, 11],
    "gaussian_blur_prob": 0.20,
    "gaussian_blur_kernel_choices": [3, 5],
    "noise_prob": 0.25,
    "noise_level_min": 5,
    "noise_level_max": 30,

    # ---------- OMBRA ----------
    "shadow_prob": 0.50,
    "shadow_alpha_min": 0.20,
    "shadow_alpha_max": 0.55,

    # ---------- OCCLUSIONE ----------
    "occlusion_prob": 0.30,
    "occlusion_coverage_min": 0.10,   # percentuale minima di copertura del bbox
    "occlusion_coverage_max": 0.35,   # percentuale massima di copertura del bbox

    # ---------- NEGATIVI ----------
    # Se True, aggiunge anche disturbi sugli sfondi negativi per renderli vari.
    "augment_negative_backgrounds": True,

    # ---------- DEBUG / DEMO ----------
    "demo_mode": False,          # True = mostra qualche preview durante generazione
    "demo_wait_ms": 150,
}


# ============================================================
# FUNZIONI DI SUPPORTO: CARTELLE
# ============================================================

def make_output_folders(dataset_root: str) -> None:
    """
    Crea la struttura classica YOLO:
    dataset/
        images/train
        images/val
        labels/train
        labels/val
    """
    folders = [
        os.path.join(dataset_root, "images", "train"),
        os.path.join(dataset_root, "images", "val"),
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# ============================================================
# CARICAMENTO FUOCO
# ============================================================

def load_fire_image(path: str) -> np.ndarray:
    """
    Carica l'immagine PNG del fuoco.
    Se presente canale alpha, viene usato poi in compositing.
    """
    fire = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if fire is None:
        raise FileNotFoundError(
            f"Impossibile caricare l'immagine del fuoco dal path: {path}"
        )

    # Ammessi:
    # - PNG BGRA (4 canali)
    # - immagine BGR (3 canali)
    if len(fire.shape) != 3 or fire.shape[2] not in [3, 4]:
        raise ValueError(
            "L'immagine del fuoco deve avere 3 canali (BGR) o 4 canali (BGRA)."
        )

    return fire


# ============================================================
# SFONDI SINTETICI: DOMAIN RANDOMIZATION
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

    bg = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            if mode == "horizontal":
                t = x / max(1, size - 1)
            elif mode == "vertical":
                t = y / max(1, size - 1)
            else:
                t = (x + y) / max(1, 2 * size - 2)

            color = (1 - t) * c1 + t * c2
            bg[y, x] = color.astype(np.uint8)

    return bg


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
        return background_mixed(size)

    generator = random.choice(generators)
    return generator(size)


# ============================================================
# AUGMENTAZIONI DEL FUOCO
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

    pts1 = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    pts2 = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h - 1 - random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), h - 1 - random.randint(0, shift)]
    ])

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
        SETTINGS["rotation_deg_min"],
        SETTINGS["rotation_deg_max"]
    )
    out = rotate_image_keep_canvas(out, angle)

    # Prospettiva: simula variazioni di viewpoint / pseudo-3D
    out = perspective_warp_keep_canvas(out, SETTINGS["perspective_shift"])

    # Contrasto + luminosità
    alpha = random.uniform(
        SETTINGS["contrast_alpha_min"],
        SETTINGS["contrast_alpha_max"]
    )
    beta = random.randint(
        SETTINGS["brightness_beta_min"],
        SETTINGS["brightness_beta_max"]
    )
    out = adjust_brightness_contrast(out, alpha, beta)

    # Piccolo shift di colore
    if SETTINGS["enable_color_shift"] and random.random() < SETTINGS["color_shift_prob"]:
        hue_delta = random.randint(
            -SETTINGS["color_shift_hue_max"],
            SETTINGS["color_shift_hue_max"]
        )
        out = color_shift_hsv(out, hue_delta)

    # Blur gaussiano
    if random.random() < SETTINGS["gaussian_blur_prob"]:
        k = random.choice(SETTINGS["gaussian_blur_kernel_choices"])
        out = add_gaussian_blur(out, k)

    # Motion blur
    if random.random() < SETTINGS["motion_blur_prob"]:
        k = random.choice(SETTINGS["motion_blur_kernel_choices"])
        out = add_motion_blur(out, k)

    # Rumore
    if random.random() < SETTINGS["noise_prob"]:
        noise_level = random.randint(
            SETTINGS["noise_level_min"],
            SETTINGS["noise_level_max"]
        )
        out = add_noise(out, noise_level)

    return out


# ============================================================
# COMPOSITING
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
    if random.random() > SETTINGS["shadow_prob"]:
        return background

    shadow_alpha = random.uniform(
        SETTINGS["shadow_alpha_min"],
        SETTINGS["shadow_alpha_max"]
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
    if random.random() > SETTINGS["occlusion_prob"]:
        return image

    out = image.copy()
    H, W = out.shape[:2]

    coverage = random.uniform(
        SETTINGS["occlusion_coverage_min"],
        SETTINGS["occlusion_coverage_max"]
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
# AUGMENTAZIONI ANCHE SUGLI SFONDI NEGATIVI
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


# ============================================================
# YOLO LABEL
# ============================================================

def yolo_label_from_bbox(x: int, y: int, w: int, h: int, image_size: int, class_id: int = 0) -> str:
    """
    Converte bbox pixel -> formato YOLO:
    class_id center_x center_y width height
    con coordinate normalizzate in [0,1].
    """
    cx = (x + w / 2) / image_size
    cy = (y + h / 2) / image_size
    bw = w / image_size
    bh = h / image_size
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ============================================================
# SALVATAGGIO
# ============================================================

def save_sample(image: np.ndarray, label_text: str, dataset_root: str, index: int, train_split: float) -> None:
    """
    Salva immagine e label nel sottoinsieme train oppure val.
    """
    split = "train" if random.random() < train_split else "val"

    img_path = os.path.join(dataset_root, "images", split, f"img_{index:05d}.jpg")
    txt_path = os.path.join(dataset_root, "labels", split, f"img_{index:05d}.txt")

    cv2.imwrite(img_path, image)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(label_text)


# ============================================================
# PREVIEW DEMO
# ============================================================

def show_demo(image: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> None:
    """
    Preview veloce durante la generazione.
    Se bbox è presente, la disegna in verde.
    """
    preview = image.copy()

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # riduco per sicurezza la visualizzazione
    preview_small = cv2.resize(preview, (700, 700))
    cv2.imshow("Generator preview", preview_small)
    cv2.waitKey(SETTINGS["demo_wait_ms"])


# ============================================================
# GENERAZIONE DI UN SINGOLO ESEMPIO
# ============================================================

def generate_negative_sample(image_size: int) -> tuple[np.ndarray, str]:
    """
    Genera un'immagine negativa: solo sfondo casuale, nessun fuoco.
    In YOLO, per una negativa basta label vuota.
    """
    bg = generate_random_background(image_size)

    if SETTINGS["augment_negative_backgrounds"]:
        bg = augment_background(bg)

    return bg, ""


def generate_positive_sample(fire_rgba: np.ndarray, image_size: int) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    """
    Genera un'immagine positiva:
    - sfondo sintetico
    - fuoco augmentato
    - dimensione casuale
    - posizione casuale
    - ombra
    - occlusione
    """
    bg = generate_random_background(image_size)

    fire_aug = augment_fire(fire_rgba)

    # scala del fuoco
    scale = random.uniform(
        SETTINGS["fire_scale_min"],
        SETTINGS["fire_scale_max"]
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    dataset_root = SETTINGS["dataset_root"]
    image_size = SETTINGS["image_size"]

    make_output_folders(dataset_root)

    fire = load_fire_image(SETTINGS["fire_path"])

    print("Avvio generazione dataset...")
    print(f"Immagine fuoco: {SETTINGS['fire_path']}")
    print(f"Cartella dataset: {dataset_root}")
    print(f"Numero immagini: {SETTINGS['num_images']}")
    print(f"Dimensione output: {image_size}x{image_size}")
    print(f"Negative ratio: {SETTINGS['negative_ratio']}")

    for i in range(SETTINGS["num_images"]):

        is_negative = random.random() < SETTINGS["negative_ratio"]

        if is_negative:
            image, label = generate_negative_sample(image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=SETTINGS["train_split"]
            )

            if SETTINGS["demo_mode"] and i % 10 == 0:
                show_demo(image, bbox=None)

        else:
            image, label, bbox = generate_positive_sample(fire, image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=SETTINGS["train_split"]
            )

            if SETTINGS["demo_mode"] and i % 10 == 0:
                show_demo(image, bbox=bbox)

        if (i + 1) % 100 == 0:
            print(f"Generate {i + 1}/{SETTINGS['num_images']} immagini")

    cv2.destroyAllWindows()
    print("Dataset generato correttamente.")


if __name__ == "__main__":
    main()