#!/usr/bin/env python3
"""
Launcher per il detector in modalità webcam.

Il programma chiede ripetutamente all'utente quale sorgente usare:
  - inserisci l'ID della webcam (es. 0, 1, 2, ...)
  - digita 'q' o 'exit' per uscire

Dopo che la finestra di rilevamento viene chiusa con 'q' o ESC, il menu ricompare
così da poter cambiare fonte senza riavviare lo script.

Usi avanzati:
    python run-webcam.py 0   # avvia direttamente con camera 0 (opzionale)

"""

import os
import sys


def prompt_camera():
    """Mostra il menu e ritorna un ID di camera o None per uscire."""
    print("\n" + "='*40")
    print("Seleziona la sorgente webcam:")
    print("  - digita un numero (es. 0) per scegliere una camera")
    print("  - digita 'q' oppure 'exit' per terminare")
    sel = input("Fonte> ").strip().lower()
    if sel == "":
        # default al 0 se l'utente preme Invio
        return 0
    if sel in ['q', 'quit', 'exit']:
        return None
    try:
        return int(sel)
    except ValueError:
        print(f"❌ Input non valido: '{sel}'")
        return prompt_camera()


def main():
    print("🔥 Fire Detector - Webcam Mode")
    print("=" * 50)

    # se è stato passato un argomento lo usiamo e poi apriamo prompt
    if len(sys.argv) > 1:
        try:
            cam = int(sys.argv[1])
            os.system(f"python detect.py --source {cam}")
        except ValueError:
            pass

    while True:
        cam_id = prompt_camera()
        if cam_id is None:
            print("Arrivederci!")
            break

        print(f"\n🎥 Avvio detection da webcam {cam_id}...")
        print("(premi 'q' o ESC nella finestra per tornare al menu)")
        os.system(f"python detect.py --source {cam_id}")


if __name__ == "__main__":
    main()
