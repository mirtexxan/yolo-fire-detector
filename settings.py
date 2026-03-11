"""
Configuration settings for YOLO Fire Detector Dataset Generator

Contains four main configuration classes:
- ImageTransformSettings: Parameters for image augmentation and transformations
- DatasetGenerationSettings: Parameters for dataset generation (paths, sizes, splits)
- ViewerSettings: Parameters for dataset visualization
- TrainingSettings: Parameters for YOLO model training
"""


class ImageTransformSettings:
    """Impostazioni per le trasformazioni geometriche e fotometriche."""
    
    # Rotazioni / Geometria
    ROTATION_DEG_MIN = -180
    ROTATION_DEG_MAX = 180
    PERSPECTIVE_SHIFT = 120  # distorsione prospettica
    
    # Luminosità / Contrasto
    BRIGHTNESS_BETA_MIN = -50
    BRIGHTNESS_BETA_MAX = 50
    CONTRAST_ALPHA_MIN = 0.60
    CONTRAST_ALPHA_MAX = 1.40
    
    # Colore
    ENABLE_COLOR_SHIFT = True
    COLOR_SHIFT_PROB = 0.35
    COLOR_SHIFT_HUE_MAX = 20
    
    # Blur
    MOTION_BLUR_PROB = 0.40
    MOTION_BLUR_KERNEL_CHOICES = [5, 7, 9, 11]
    GAUSSIAN_BLUR_PROB = 0.20
    GAUSSIAN_BLUR_KERNEL_CHOICES = [3, 5]
    
    # Rumore
    NOISE_PROB = 0.25
    NOISE_LEVEL_MIN = 5
    NOISE_LEVEL_MAX = 30
    
    # Ombra
    SHADOW_PROB = 0.50
    SHADOW_ALPHA_MIN = 0.20
    SHADOW_ALPHA_MAX = 0.55
    
    # Occlusione
    OCCLUSION_PROB = 0.30
    OCCLUSION_COVERAGE_MIN = 0.10
    OCCLUSION_COVERAGE_MAX = 0.35
    
    # Augmentazione sfondi negativi
    AUGMENT_NEGATIVE_BACKGROUNDS = True


class DatasetGenerationSettings:
    """Impostazioni per la generazione del dataset."""
    
    # Path
    FIRE_IMAGE_PATH = r"fire.png"
    DATASET_ROOT = "dataset"
    
    # Dataset
    NUM_IMAGES = 2000
    TRAIN_SPLIT = 0.8          # 80% train, 20% val
    NEGATIVE_RATIO = 0.35      # percentuale immagini senza fuoco
    
    # Dimensioni
    IMAGE_SIZE = 640
    
    # Dimensione fuoco
    FIRE_SCALE_MIN = 0.05
    FIRE_SCALE_MAX = 0.50
    
    # Demo
    DEMO_MODE = False
    DEMO_WAIT_MS = 150


class ViewerSettings:
    """Impostazioni per il visualizzatore del dataset."""
    
    # Dataset
    DATASET_ROOT = "dataset"
    SPLIT = "train"  # "train" oppure "val"
    
    # Visualizzazione
    NUM_SAMPLES = 9       # quante immagini mostrare
    THUMB_SIZE = 280      # dimensione miniatura
    DRAW_TITLE = True     # scrive il nome file sopra


class TrainingSettings:
    """Impostazioni per l'addestramento del modello YOLO."""
    
    # === MODELLO ===
    MODEL_SIZE = "n"      # Dimensione modello: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
    DEVICE = "cpu"        # Device per training: 'cpu' o numero GPU (0, 1, 2, ...)
    
    # === TRAINING ===
    EPOCHS = 100          # Numero totale di epoche di training
    BATCH_SIZE = 16       # Batch size per step di training
    IMAGE_SIZE = 640      # Dimensione delle immagini di input (pixels)
    PATIENCE = 10         # Early stopping patience (epoche senza miglioramento)
    
    # === OTTIMIZZAZIONE ===
    LEARNING_RATE_INIT = 0.01     # Learning rate iniziale
    LEARNING_RATE_FINAL = 0.01    # Learning rate finale (decade linearmente)
    MOMENTUM = 0.937               # Momentum per SGD optimizer
    WEIGHT_DECAY = 0.0005         # L2 regularization weight decay
    
    # === AUGMENTAZIONE ===
    # Rotazioni casuali delle immagini
    ROTATION_DEGREES = 15          # Gradi massimi di rotazione (±15°)
    
    # Traslazioni casuali
    TRANSLATE = 0.1                # Frazione massima di traslazione (10% dell'immagine)
    
    # Scaling casuale
    SCALE = 0.5                    # Fattore di scala (0.5 = ±50% dimensione)
    
    # Flip verticale/orizzontale
    FLIP_VERTICAL = 0.5            # Probabilità di flip verticale (0.0-1.0)
    FLIP_HORIZONTAL = 0.5          # Probabilità di flip orizzontale (0.0-1.0)
    
    # Mosaic augmentation
    MOSAIC = 1.0                   # Probabilità di mosaic augmentation (0.0-1.0)
    
    # === PRECISIONE MISTA ===
    MIXED_PRECISION = True         # Usa Automatic Mixed Precision (AMP) per velocità
    
    # === OUTPUT ===
    PROJECT_NAME = "fire_detector_runs"  # Nome del progetto per salvare i risultati
    EXPERIMENT_NAME = "train"            # Nome dell'esperimento specifico
    OVERWRITE_EXISTING = True            # Sovrascrivi esperimenti esistenti
    VERBOSE = True                       # Output verboso durante il training
