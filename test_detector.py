#!/usr/bin/env python3
"""
Script di test rapido per il detector di fuoco

Questo script testa il modello YOLO addestrato sulle immagini di validation
per verificare che funzioni correttamente.

Usage:
    python test_detector.py
"""

import os
import sys

def main():
    print("🔥 Fire Detector - Test Script")
    print("=" * 40)

    # Verifica che il modello esista (YOLOv8 lo crea in runs/detect/)
    model_paths = [
        "runs/detect/fire_detector_runs/train/weights/best.pt",
        "fire_detector_runs/train/weights/best.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Modello trovato: {path}")
            break
    
    if model_path is None:
        print(f"❌ Modello non trovato in nessuna di queste locazioni:")
        for p in model_paths:
            print(f"   - {p}")
        print("\nEsegui prima il training: python train.py")
        return

    # Verifica che ci siano immagini di validation
    val_images_path = "dataset/images/val"
    if not os.path.exists(val_images_path):
        print(f"❌ Immagini di validation non trovate: {val_images_path}")
        print("Genera prima il dataset: python generator.py")
        return

    # Conta le immagini
    image_count = len([f for f in os.listdir(val_images_path) if f.endswith('.jpg')])
    print(f"✅ Trovate {image_count} immagini di validation")

    # Comando per testare
    print("\n🚀 Comando per testare il detector:")
    print(f"python detect.py --source {val_images_path}")
    print("\n📋 Controlli durante il test:")
    print("  ← → (frecce) o 'a'/'d': Naviga tra le immagini")
    print("  's' : Salva immagine con detections")
    print("  'q' : Esci")

    # Esegui il test automaticamente
    print("\n🔍 Avvio test automatico...")
    os.system(f"python detect.py --source {val_images_path} --conf 0.3")

if __name__ == "__main__":
    main()