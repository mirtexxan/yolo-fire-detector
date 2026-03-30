# YOLO Fire Detector

Repository per generare dataset sintetici di fuoco, addestrare YOLOv8 con config YAML e usare il modello finale in inferenza locale o cloud.

## Uso rapido

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 2. Verifica che la pipeline funzioni

```bash
python run_experiment.py --config configs/local.smoke.yaml
```

Questa smoke run controlla end-to-end:

- generazione dataset
- training
- export finale
- registrazione del modello corrente

### 3. Esegui una run locale normale

```bash
python run_experiment.py --config configs/local.default.yaml
```

### 4. Avvia il detector

```bash
python detect.py --source webcam
```

Se non passi `--weights`, il detector usa automaticamente il modello puntato da `artifacts/local/exports/latest.yaml`.

## Workflow locale consigliato

Il percorso normale e':

1. lancia una config YAML con `run_experiment.py`
2. verifica che sotto `artifacts/local/exports/` sia comparso il nuovo `.pt`
3. usa `detect.py` senza passare i pesi, oppure passa un peso esplicito se vuoi testare un modello specifico

Comando tipico:

```bash
python run_experiment.py --config configs/local.default.yaml
python detect.py --source webcam
```

## Guida pratica a detect

`detect.py` e' l'entry point principale per l'inferenza.

### Sorgenti supportate

#### Selettore webcam testuale

```bash
python detect.py
python detect.py --source webcam
```

Comportamento:

- fa una scansione delle webcam disponibili
- mostra un elenco testuale con gli ID trovati
- mette `*` sulle camere che hanno restituito un frame non nero
- chiede quale camera aprire

#### Webcam diretta per ID

```bash
python detect.py --source 0
python detect.py --source 1
```

Usa questa modalita' se sai gia' quale camera vuoi aprire.

#### Cartella immagini

```bash
python detect.py --source path/to/images
```

Comportamento:

- analizza tutte le immagini della cartella
- permette navigazione manuale tra le immagini
- e' utile per controlli veloci su validation set o batch di test

#### Video locale

```bash
python detect.py --source video.mp4
```

#### Stream RTMP o RTSP

```bash
python detect.py --source rtmp://server/app/stream
python detect.py --source rtsp://server/stream
```

### Modello usato

Default:

```bash
python detect.py --source webcam
```

In questo caso il detector usa l'ultimo export registrato.

Peso esplicito:

```bash
python detect.py --weights path/to/model.pt --source webcam
```

Usa `--weights` quando vuoi testare un modello preciso e non il corrente.

### Parametri importanti

#### `--conf`

```bash
python detect.py --source webcam --conf 0.3
python detect.py --source webcam --conf 0.7
```

Significato:

- `0.3`: detector piu' permissivo, vede piu' box ma aumenta il rumore
- `0.5`: valore intermedio ragionevole per partire
- `0.7`: detector piu' severo, meno falsi positivi ma rischi di perdere detections deboli

#### `--device`

```bash
python detect.py --source webcam --device cpu
python detect.py --source webcam --device 0
```

Significato:

- `cpu`: inferenza sul processore
- `0`: inferenza sulla GPU 0

Se hai GPU CUDA disponibile, `--device 0` e' in genere molto piu' veloce.

### Controlli durante l'uso

Per webcam, video e stream:

- `q` o `Esc`: esce
- `s`: salva il frame corrente in `detections/`

Per cartella immagini:

- `a` o freccia sinistra: immagine precedente
- `d` o freccia destra: immagine successiva
- `s`: salva l'immagine annotata
- `q` o `Esc`: esce

## Configurazioni YAML

Le configurazioni vivono in `configs/`.

File principali:

- `configs/local.smoke.yaml`: test rapido locale
- `configs/local.default.yaml`: run locale standard
- `configs/cloud.default.yaml`: base per Colab
- `configs/cloud.*.yaml`: preset cloud gia' pronti

Le config controllano:

- root persistente degli output
- immagini base del fuoco in `dataset.fire_image_paths`
- parametri del dataset sintetico
- parametri del training
- override avanzati di dataset, trasformazioni e training

### Cambiare immagini base

Una sola immagine:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire.png
```

Una immagine diversa:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire2.png
```

Un sottoinsieme esplicito:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire.png
    - base_fire_images/fire2.png
```

Se la lista contiene piu' immagini, la generazione sceglie casualmente tra quelle specificate.

## Output della pipeline

La root locale di default e' `artifacts/local/`.

### Dataset

```text
artifacts/local/datasets/<dataset-label-slug>-<fingerprint>/
```

Dentro trovi:

- `yolo_dataset.yaml`: file usato da Ultralytics per training/validation
- `dataset_manifest.yaml`: metadata del dataset generato
- `images/`
- `labels/`

Convenzione di naming:

- `<dataset-label-slug>-<fingerprint>`
- il fingerprint cambia quando cambiano i parametri che definiscono davvero il dataset

### Run

```text
artifacts/local/runs/<run-label>/
```

Dentro trovi:

- `resolved_config.yaml`
- `pipeline_summary.yaml`
- `training_run.yaml`
- plot e diagnostica YOLO

Convenzione di naming della run:

- `<environment>-<project-label>-<training-label-ridotta>-<dataset-label-ridotta>-<fingerprint>`
- i token duplicati vengono eliminati automaticamente
- il fingerprint finale e' quello del dataset usato

### Export finale

```text
artifacts/local/exports/
```

Dentro trovi:

- `exports/<run_label>.pt`: modello finale per inferenza
- `exports/<run_label>.yaml`: metadata sintetici dell'export
- `exports/latest.yaml`: puntatore stabile all'export corrente

`detect.py` legge `latest.yaml` quando non passi `--weights`.

## Launch profiles in VS Code

In `/.vscode/launch.json` ci sono profili pronti per:

- pipeline locale
- bundle cloud
- detect da webcam con latest export
- detect su validation images
- detect da video file
- detect da stream
- detect da cartella immagini

## Troubleshooting rapido

### Il detector non trova il modello

Esegui una run locale prima:

```bash
python run_experiment.py --config configs/local.default.yaml
```

### Vuoi rigenerare il dataset anche se esiste gia'

Imposta:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi usare la GPU

In training: imposta `training.device` nella config.

In detect:

```bash
python detect.py --source webcam --device 0
```

### Ti serve il cloud

Usa [CLOUD_TRAINING.md](CLOUD_TRAINING.md).

## File principali

- `run_experiment.py`: pipeline generation + training
- `detect.py`: inferenza da webcam, stream, video o immagini
- `prepare_cloud_bundle.py`: zip per Colab
- `cloud_train.ipynb`: notebook cloud persistente
- `TRAINING_PRESETS.md`: panoramica preset cloud
