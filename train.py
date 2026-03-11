"""
Training script for YOLO Fire Detector

Trains a YOLOv8 model to detect fire using the generated synthetic dataset.

Usage:
    python train.py [options]
    
Configuration:
    All training parameters are centralized in TrainingSettings class.
    Command line arguments can override the default settings.
    
    Model sizes: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
    Default model: YOLOv8n
    Default epochs: 100
    Default image size: 640
    Default batch size: 16
"""

from ultralytics import YOLO
import os
from settings import DatasetGenerationSettings, TrainingSettings


def create_dataset_yaml() -> str:
    """
    Crea il file data.yaml necessario per il training YOLO.
    
    Returns:
        str: percorso del file data.yaml
    """
    dataset_root = DatasetGenerationSettings.DATASET_ROOT
    yaml_path = os.path.join(dataset_root, "data.yaml")
    
    # Conteggia immagini per validazione
    images_train = os.path.join(dataset_root, "images", "train")
    images_val = os.path.join(dataset_root, "images", "val")
    
    # Crea il contenuto del file YAML
    yaml_content = f"""path: {os.path.abspath(dataset_root)}
train: images/train
val: images/val

nc: 1  # number of classes
names: ['fire']  # class names
"""
    
    # Scrivi il file
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Dataset YAML creato: {yaml_path}")
    return yaml_path


def train_model(
    model_size: str = TrainingSettings.MODEL_SIZE,
    epochs: int = TrainingSettings.EPOCHS,
    batch_size: int = TrainingSettings.BATCH_SIZE,
    image_size: int = TrainingSettings.IMAGE_SIZE,
    device: str = TrainingSettings.DEVICE,
) -> None:
    """
    Addestra il modello YOLO sulla detection del fuoco.
    
    Args:
        model_size: Dimensione del modello ('n', 's', 'm', 'l') - default da TrainingSettings
        epochs: Numero di epoche di training - default da TrainingSettings
        batch_size: Batch size per il training - default da TrainingSettings
        image_size: Dimensione delle immagini di input - default da TrainingSettings
        device: Device per training ('cpu' o numero GPU) - default da TrainingSettings
    """
    
    # Verifica che il dataset esista
    dataset_root = DatasetGenerationSettings.DATASET_ROOT
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"Dataset non trovato in {dataset_root}\n"
            f"Esegui prima: python generator.py"
        )
    
    images_train = os.path.join(dataset_root, "images", "train")
    if not os.path.exists(images_train) or len(os.listdir(images_train)) == 0:
        raise FileNotFoundError(
            f"Nessuna immagine di training trovata in {images_train}\n"
            f"Esegui prima: python generator.py"
        )
    
    # Crea il file data.yaml
    yaml_path = create_dataset_yaml()
    
    # Carica il modello
    print("\n" + "="*60)
    print(f"Caricamento modello YOLOv8{model_size}...")
    print("="*60)
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Training
    print("\n" + "="*60)
    print("Inizio training...")
    print("="*60)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=TrainingSettings.PATIENCE,
        device=device,
        project=TrainingSettings.PROJECT_NAME,
        name=TrainingSettings.EXPERIMENT_NAME,
        exist_ok=TrainingSettings.OVERWRITE_EXISTING,
        verbose=TrainingSettings.VERBOSE,
        # Hyperparameters
        lr0=TrainingSettings.LEARNING_RATE_INIT,
        lrf=TrainingSettings.LEARNING_RATE_FINAL,
        momentum=TrainingSettings.MOMENTUM,
        weight_decay=TrainingSettings.WEIGHT_DECAY,
        # Augmentation
        degrees=TrainingSettings.ROTATION_DEGREES,
        translate=TrainingSettings.TRANSLATE,
        scale=TrainingSettings.SCALE,
        flipud=TrainingSettings.FLIP_VERTICAL,
        fliplr=TrainingSettings.FLIP_HORIZONTAL,
        mosaic=TrainingSettings.MOSAIC,
        # Mixed Precision
        amp=TrainingSettings.MIXED_PRECISION,
    )
    
    print("\n" + "="*60)
    print("Training completato!")
    print("="*60)
    print(f"Modello salvato in: fire_detector_runs/train/weights/best.pt")
    print(f"Risultati disponibili in: fire_detector_runs/train/")


def validate_model(model_path: str = "fire_detector_runs/train/weights/best.pt") -> None:
    """
    Valida il modello addestrato.
    
    Args:
        model_path: Percorso del modello da validare
    """
    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}")
        return
    
    print("\n" + "="*60)
    print("Validazione del modello...")
    print("="*60)
    
    model = YOLO(model_path)
    
    # Crea data.yaml se non esiste
    if not os.path.exists(os.path.join(DatasetGenerationSettings.DATASET_ROOT, "data.yaml")):
        yaml_path = create_dataset_yaml()
    else:
        yaml_path = os.path.join(DatasetGenerationSettings.DATASET_ROOT, "data.yaml")
    
    metrics = model.val(data=yaml_path)
    
    print("\n" + "="*60)
    print("Metriche di validazione:")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training YOLO Fire Detector")
    parser.add_argument(
        "--model",
        type=str,
        default=TrainingSettings.MODEL_SIZE,
        choices=["n", "s", "m", "l"],
        help=f"Dimensione del modello YOLOv8 (default: {TrainingSettings.MODEL_SIZE})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingSettings.EPOCHS,
        help=f"Numero di epoche (default: {TrainingSettings.EPOCHS})"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=TrainingSettings.BATCH_SIZE,
        help=f"Batch size (default: {TrainingSettings.BATCH_SIZE})"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TrainingSettings.IMAGE_SIZE,
        help=f"Image size (default: {TrainingSettings.IMAGE_SIZE})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=TrainingSettings.DEVICE,
        help=f"Device per training: 'cpu' o numero GPU (default: {TrainingSettings.DEVICE})"
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Solo validazione, non fare training"
    )
    
    args = parser.parse_args()
    
    if args.val_only:
        validate_model()
    else:
        train_model(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
        )
