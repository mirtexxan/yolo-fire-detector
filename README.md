# YOLO Fire Detector

YOLO Fire Detector e' un progetto per il rilevamento di incendi costruito attorno a tre idee semplici:

- generare un dataset sintetico riproducibile a partire da immagini base in `base_fire_images/`
- addestrare YOLOv8 con una pipeline configurabile da file YAML
- usare il modello addestrato in locale o in cloud mantenendo tracciabilita' di dataset, run e modello finale

Il repository e' pensato sia come progetto dimostrativo sia come base concreta per iterare su dataset sintetici, training e inferenza video.

## Cosa fa il progetto

Il flusso completo e' questo:

1. una config YAML descrive dataset, training e destinazione degli output
2. la pipeline costruisce o riusa un dataset sintetico coerente con quei parametri
3. YOLOv8 viene addestrato o ripreso da checkpoint se la run esiste gia'
4. il modello migliore viene esportato e registrato con i metadati del dataset usato
5. `detect.py` e gli script `run-*` usano il modello piu' recente oppure un peso esplicito passato da CLI

## Entry point supportati

Per training e generazione esiste un solo entrypoint supportato:

```bash
python run_experiment.py --config configs/local.default.yaml
```

Per preparare il bundle cloud:

```bash
python prepare_cloud_bundle.py
```

Per l'inferenza:

```bash
python run-detector.py
python run-webcam.py
python run-images.py
python detect.py --source 0
```

`generator.py` e `train.py` restano moduli interni richiamati dalla pipeline, ma non sono piu' il percorso d'uso consigliato.

## Quick Start

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 2. Lancia una smoke run locale

```bash
python run_experiment.py --config configs/local.smoke.yaml
```

Questa run serve a verificare rapidamente che:

- generazione dataset
- training YOLO
- export finale
- registry del modello

funzionino end-to-end.

### 3. Lancia una run locale standard

```bash
python run_experiment.py --config configs/local.default.yaml
```

### 4. Avvia il detector

```bash
python run-detector.py
```

## Architettura

### 1. Config layer

Le configurazioni vivono in `configs/`:

- `configs/local.default.yaml`: run locale standard
- `configs/local.smoke.yaml`: test rapido locale
- `configs/cloud.default.yaml`: base per Colab/Kaggle

La config controlla:

- root persistente degli output
- lista esplicita delle immagini base (`dataset.fire_image_paths`)
- parametri del dataset
- parametri del training
- override avanzati su trasformazioni, dataset e training

### 2. Orchestrazione

`run_experiment.py` e' l'orchestratore centrale. Le sue responsabilita' sono:

- leggere la config YAML
- calcolare il fingerprint del dataset richiesto
- riusare un dataset compatibile se gia' presente
- generare il dataset se manca o se i parametri sono cambiati
- costruire una label leggibile per la run
- decidere se fare resume dal checkpoint esistente
- lanciare il training
- esportare il modello finale e registrarlo con i metadati

### 3. Dataset generation

`generator.py` genera immagini positive e negative con:

- background sintetici randomizzati
- trasformazioni geometriche e fotometriche sul fuoco
- selezione casuale tra le immagini presenti in `dataset.fire_image_paths`
- split train/val automatico

Ogni dataset viene salvato in una cartella separata identificata da label e fingerprint, per esempio:

```text
artifacts/local/datasets/synthetic-fire-default__a1b2c3d4e5/
```

Dentro la cartella c'e' anche `dataset_info.json`, che descrive:

- parametri usati
- fingerprint
- conteggi train/val
- statistiche del dataset generato

### 4. Training

`train.py` contiene le utility di training usate dalla pipeline. La run YOLO viene salvata in una cartella con nome leggibile, per esempio:

```text
artifacts/local/runs/local__baseline-local__yolov8n-local__yolov8n__synthetic-fire-default__a1b2c3d4e5/
```

Dentro la run trovi:

- checkpoint YOLO `weights/best.pt` e `weights/last.pt`
- file grafici e metriche generate da YOLO
- `resolved_config.yaml`
- `dataset_info.json`
- `pipeline_summary.json`
- `final_export/`

### 5. Model registry

Alla fine della pipeline il modello finale viene registrato in:

```text
artifacts/local/models/
```

In particolare:

- `models/<run_label>.pt`: modello finale esportato
- `models/<run_label>.json`: metadati di dataset e run
- `models/latest.json`: puntatore al modello piu' recente

`detect.py`, se non riceve `--weights`, prova a leggere proprio questo registry.

## Struttura del repository

I file piu' importanti sono:

- `run_experiment.py`: entrypoint unico per generation + training
- `prepare_cloud_bundle.py`: crea lo zip per Colab/Kaggle
- `cloud_train.ipynb`: notebook cloud persistente
- `detect.py`: inferenza da webcam, stream, video o immagini
- `run-detector.py`: menu interattivo per detection
- `generator.py`: modulo di generazione dataset
- `train.py`: modulo di training YOLO
- `transformations.py`: trasformazioni visuali e augmentazioni
- `utils.py`: utility di salvataggio e I/O
- `settings.py`: default statici delle classi di configurazione

## Come usare il repo in locale

### Workflow consigliato

Per lavorare bene in locale il percorso normale e':

```bash
python run_experiment.py --config configs/local.default.yaml
```

Poi:

```bash
python run-detector.py
```

oppure:

```bash
python detect.py --source 0
```

### Smoke test

Per verificare che tutto sia ancora integro dopo modifiche al codice:

```bash
python run_experiment.py --config configs/local.smoke.yaml
```

### Cambiare parametri

Il modo corretto per modificare una run non e' cambiare il codice, ma:

1. duplicare una config YAML
2. modificare label, dataset e training
3. rilanciare `run_experiment.py`

Per usare il default:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire.png
```

Per usare una sola immagine diversa:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire2.png
```

Per usare un sottoinsieme scelto da te:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire2.png
    - base_fire_images/fire.png
```

Se la lista contiene una sola immagine, viene usata solo quella. Se contiene piu' immagini, la pipeline sceglie casualmente tra quelle presenti. Se non specifichi niente, il default resta una lista con `base_fire_images/fire.png`.

Questo mantiene separati:

- dataset diversi
- run diverse
- modelli finali diversi

## Come usare il modello

### Modalita' interattiva

```bash
python run-detector.py
```

### Webcam

```bash
python run-webcam.py
python detect.py --source 0
```

### Cartella immagini

```bash
python run-images.py
python detect.py --source path/to/images
```

### Video o stream

```bash
python detect.py --source video.mp4
python detect.py --source rtmp://server/app/stream
python detect.py --source rtsp://server/stream
```

### Peso personalizzato

```bash
python detect.py --weights path/to/model.pt --source 0
```

## Output e artefatti

Nel contesto di questo progetto, gli artefatti sono tutti i file generati dalla pipeline e non scritti a mano nel repository. Per esempio:

- dataset generati
- manifest JSON del dataset
- checkpoint di training
- export finali del modello
- registry dei modelli
- zip preparato per il cloud

Per questo vengono ignorati da git in `.gitignore`.

## Launch profiles in VS Code

Sono presenti due configurazioni in `.vscode/launch.json`:

- `Local Pipeline (YAML)`
- `Create Cloud Bundle`

## Troubleshooting rapido

### Non trova il modello

Esegui una run locale:

```bash
python run_experiment.py --config configs/local.default.yaml
```

### Vuoi usare la GPU

Aggiorna la config YAML o il campo `device` nella config che stai usando.

### Il dataset va rigenerato

Imposta nella config:

```yaml
dataset:
  force_regenerate: true
```

### Ti serve il cloud

Usa la guida dedicata in `CLOUD_TRAINING.md`.

## Note finali

Questo repository oggi ha una separazione netta:

- `README.md`: presentazione progetto e guida utente generale
- `CLOUD_TRAINING.md`: procedura operativa per Colab e, in fondo, Kaggle

Se vuoi estendere il progetto, il punto giusto da cui partire e' quasi sempre `configs/` piu' `run_experiment.py`.
