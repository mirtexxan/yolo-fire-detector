# Guida per Training su Server Gratuiti

## Opzione 1: Google Colab (Raccomandato)

### Passi per eseguire il training:

1. **Apri Colab**: Vai su [colab.research.google.com](https://colab.research.google.com)

2. **Carica il codice**:
   - Crea un nuovo notebook
   - Carica i file del progetto (generator.py, train.py, settings.py, etc.)
   - Oppure clona il repo se è su GitHub

3. **Installa dipendenze**:
   ```python
   !pip install ultralytics opencv-python numpy
   ```

4. **Carica il dataset** (se necessario):
   ```python
   # Se hai il dataset locale, caricalo su Colab
   from google.colab import files
   uploaded = files.upload()  # carica dataset.zip

   # Estrai
   !unzip dataset.zip
   ```

5. **Esegui il training**:
   ```python
   # Genera dataset se necessario
   !python generator.py

   # Avvia training
   !python train.py --device 0  # usa GPU
   ```

6. **Scarica risultati**:
   ```python
   from google.colab import files
   files.download('runs/detect/fire_detector_runs/train/final_export/best.pt')
   files.download('runs/detect/fire_detector_runs/train/final_export/training_settings.txt')
   ```

### Limiti Colab:
- Sessione massima 12 ore
- GPU gratuita limitata (circa 2-3 ore al giorno)
- Storage temporaneo (perde dati alla chiusura)

## Opzione 2: Amazon SageMaker Studio Lab

### Passi:

1. **Registrati**: Vai su [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)

2. **Crea progetto**:
   - Carica i file Python
   - Usa il terminale integrato per installare dipendenze

3. **Training**:
   ```bash
   pip install ultralytics
   python train.py --device 0
   ```

### Limiti:
- 8 ore CPU al giorno
- 4 ore GPU al giorno
- Storage 15GB

## Suggerimenti Generali

### Per entrambi:
- **Dataset**: Se grande, considera di generarlo sul server invece di caricarlo
- **Checkpoints**: Usa `--resume` per continuare training interrotto
- **GPU**: Verifica disponibilità con `!nvidia-smi` su Colab
- **Salvataggio**: Scarica regolarmente i checkpoint per non perdere lavoro

### Ottimizzazioni:
- Riduci batch size se GPU limitata
- Usa modello 'n' (nano) per training più veloce
- Monitora RAM (Colab ha 12-25GB)

## Costi
- **Colab**: Completamente gratuito
- **SageMaker Studio Lab**: Gratuito (beta)

Entrambi sono ideali per scopi didattici!