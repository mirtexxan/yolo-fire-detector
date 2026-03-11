#!/usr/bin/env python3
"""
Script semplice per lanciare il detector in modalità webcam

Usage:
    python run_webcam.py
"""

import os
import sys

def main():
    print("🔥 Fire Detector - Webcam Mode")
    print("=" * 50)
    print("\n🎥 Avvio detection da webcam...")
    print("\nControlli:")
    print("  'q' o ESC: Esci")
    print("  's': Salva frame corrente")
    print("\n" + "=" * 50 + "\n")
    
    # Lancia detect.py in modalità webcam
    os.system("python detect.py --source 0")

if __name__ == "__main__":
    main()
