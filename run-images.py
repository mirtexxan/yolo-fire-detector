#!/usr/bin/env python3
"""
Script per testare il detector su immagini statiche

Testa il modello su:
- Immagini di validation (default)
- Qualsiasi cartella di immagini

Usage:
    python run-images.py                    # Validation images
    python run-images.py <cartella>         # Cartella personalizzata
"""

import os
import sys

def main():
    print("\n🔥 Fire Detector - Image Detection")
    print("=" * 50)
    
    # Cartella di default
    folder = "dataset/images/val/"
    
    # Se passato un argomento, usalo come cartella
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    
    print(f"\n📋 Test su immagini: {folder}")
    print("\nControlli:")
    print("  ← → (frecce) o 'a'/'d': Naviga tra le immagini")
    print("  's': Salva immagine corrente")
    print("  'q' o ESC: Esci")
    print("\n" + "=" * 50 + "\n")
    
    # Lancia detect.py sulle immagini
    os.system(f'python detect.py --source "{folder}" --conf 0.3')

if __name__ == "__main__":
    main()
