# 🔥 YOLO Fire Detector

Sistema completo per il rilevamento di incendi utilizzando YOLOv8 e dataset sintetici.

## 🚀 Quick Start

```bash
# Installa le dipendenze
pip install -r requirements.txt

# 1. Genera il dataset sintetico
python generator.py

# 2. Addestra il modello YOLO
python train.py

# 3. Testa il modello sulle immagini di validation
python detect.py --source dataset/images/val/

# 4. Usa il detector in tempo reale
python detect.py  # Webcam
python detect.py --source rtmp://stream.url  # RTMP stream
python detect.py --source video.mp4  # Video file
```

## 📋 Funzionalità

### Generazione Dataset (`generator.py`)
- Dataset sintetico con immagini di fuoco generate proceduralmente
- Background casuali da cartelle locali
- Augmentazioni geometriche e fotometriche
- Bilanciamento automatico train/validation

### Training (`train.py`)
- Training YOLOv8 con configurazione ottimizzata
- Supporto CPU/GPU
- Early stopping e mixed precision
- Logging completo dei risultati

### Detection (`detect.py`)
- **Webcam**: Rilevamento in tempo reale da webcam
- **RTMP/RTSP Streams**: Streaming video remoto
- **Video Files**: Analisi di file video locali
- **Static Images**: Test su cartelle di immagini per validation

## 🎮 Controlli Detection

### Modalità Webcam/Stream/Video:
- `q` o `ESC`: Esci
- `s`: Salva frame corrente

### Modalità Test Immagini:
- `← →` (frecce) o `a`/`d`: Naviga tra le immagini
- `s`: Salva immagine corrente
- `q` o ESC: Esci

**Nota:** Per migliore supporto dei tasti, installa `pynput`:
```bash
pip install pynput
```
Con pynput, le arrow keys funzionano da qualsiasi applicazione (anche se la finestra non è in focus).

## ⚙️ Configurazione

Tutte le impostazioni sono centralizzate in `settings.py`:

- `ImageTransformSettings`: Parametri di augmentazione
- `DatasetGenerationSettings`: Configurazione dataset
- `ViewerSettings`: Impostazioni visualizzatore
- `TrainingSettings`: Parametri di training

## 📊 Risultati

I risultati del training vengono salvati in `runs/detect/fire_detector_runs/train/`:
- Modelli: `weights/best.pt`, `weights/last.pt`
- Log di training e metriche
- Il modello migliore viene automaticamente rilevato e usato da `detect.py`

## 🔧 Troubleshooting

### Errore CUDA
Se ottieni errori CUDA, il sistema usa automaticamente CPU. Per forzare GPU:
```bash
python train.py --device 0
python detect.py --device 0
```

### Dataset vuoto
```bash
# Rigenera il dataset
python generator.py
```

### Modello non trovato
```bash
# Riaddestra il modello
python train.py
```

## 📈 Performance

- **Modello**: YOLOv8n (nano) - bilanciamento velocità/accuratezza
- **Dataset**: ~2000 immagini sintetiche
- **Training**: ~100 epoche, ~30-60 minuti su CPU
- **Inference**: ~50-100 FPS su CPU, più veloce su GPU
