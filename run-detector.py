#!/usr/bin/env python3
"""
Script interattivo per scegliere la sorgente del detector

Usage:
    python run_detector.py
"""

import os
import sys

def main():
    print("\n🔥 Fire Detector - Source Selection")
    print("=" * 50)
    print("\nScegli la sorgente:")
    print("  1. Webcam")
    print("  2. Immagini di validation (dataset/images/val/)")
    print("  3. File video (specifica il path)")
    print("  4. RTMP Stream (specifica l'URL)")
    print("  5. Cartella di immagini (specifica il path)")
    print("\n0. Esci")
    print("=" * 50)
    
    choice = input("\nScelta (0-5): ").strip()
    
    if choice == "1":
        print("\n🎥 Avvio detection da webcam...")
        os.system("python detect.py --source 0")
    
    elif choice == "2":
        print("\n📋 Test su immagini di validation...")
        os.system("python detect.py --source dataset/images/val/ --conf 0.3")
    
    elif choice == "3":
        video_path = input("\nSpecifica il path del file video: ").strip()
        if os.path.isfile(video_path):
            os.system(f'python detect.py --source "{video_path}"')
        else:
            print(f"❌ File non trovato: {video_path}")
    
    elif choice == "4":
        rtmp_url = input("\nSpecifica l'URL RTMP/RTSP: ").strip()
        if rtmp_url.startswith(("rtmp://", "rtsp://")):
            os.system(f'python detect.py --source "{rtmp_url}"')
        else:
            print("❌ URL non valido. Deve iniziare con rtmp:// o rtsp://")
    
    elif choice == "5":
        folder_path = input("\nSpecifica il path della cartella: ").strip()
        if os.path.isdir(folder_path):
            os.system(f'python detect.py --source "{folder_path}"')
        else:
            print(f"❌ Cartella non trovata: {folder_path}")
    
    elif choice == "0":
        print("\n👋 Uscita...")
    
    else:
        print("\n❌ Scelta non valida")

if __name__ == "__main__":
    main()
