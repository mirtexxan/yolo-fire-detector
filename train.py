"""
Training script for YOLO Fire Detector

Trains a YOLOv8 model to detect fire using the generated synthetic dataset.

Usage:
    python train.py
    
Configuration:
    - Model: YOLOv8n (nano), YOLOv8s (small), YOLOv8m (medium), YOLOv8l (large)
    - Epochs: 100
    - Image size: 640
    - Batch size: 16
    - Dataset: dataset/ (created by generator.py)
"""

from ultralytics import YOLO
import os
from settings import DatasetGenerationSettings


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
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    patience: int = 10,
    device: int = 0,
) -> None:
    """
    Addestra il modello YOLO sulla detection del fuoco.
    
    Args:
        model_size: Dimensione del modello ('n', 's', 'm', 'l')
        epochs: Numero di epoche di training
        batch_size: Batch size per il training
        image_size: Dimensione delle immagini di input
        patience: Patience per early stopping
        device: GPU device id (0 per la prima GPU, -1 per CPU)
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
        patience=patience,
        device=device,
        project="fire_detector_runs",
        name="train",
        exist_ok=True,
        verbose=True,
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Mixed Precision
        amp=True,
        # Early Stopping
        patience=patience,
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
        default="n",
        choices=["n", "s", "m", "l"],
        help="Dimensione del modello YOLOv8 (default: n)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero di epoche (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device id (default: 0)"
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
