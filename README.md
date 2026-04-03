# YOLO Fire Detector

YOLO Fire Detector e' un progetto per creare dataset sintetici di immagini con fuoco, addestrare un modello YOLOv8 in locale o in cloud e usare il modello finale in inferenza (detect) con diversi tipi di fonti (webcam, video, cartella immagini, stream)

L'entrypoint principale per generazione dataset, training ed export e' sempre:

```bash
python run_experiment.py --config <config.yaml>
```

Non e' necessario eseguire separatamente `generator.py` o `train.py` per il flusso standard.

## Prima di tutto: il percorso piu' semplice

Se stai aprendo il progetto per la prima volta, il percorso consigliato e' questo:

1. installa le dipendenze Python
2. esegui uno smoke test locale
3. avvia il detector con l'ultimo modello disponibile
4. se vuoi usare il cloud, crea il bundle e apri il notebook

Comandi minimi:

```bash
pip install -r requirements.txt
python run_experiment.py --config configs/presets/smoke.yaml
python detect.py --source webcam
```

Per il flusso cloud:

```bash
python tools/cloud/cloud_configurator.py
python tools/cloud/prepare_cloud_bundle.py
```

Poi apri `cloud_train.ipynb` in Google Colab, seleziona subito un runtime GPU, carica `yolo-fire-detector-cloud.zip` e segui le celle in ordine.

## Cosa produce il progetto

Una esecuzione completa della pipeline fa queste operazioni:

1. legge una configurazione YAML
2. genera oppure riusa un dataset YOLO sotto una cartella persistente
3. addestra un modello YOLOv8
4. esporta il modello finale in formato `.pt`
5. aggiorna un puntatore `latest.yaml` per il riuso in `detect.py`

Gli output locali, per impostazione predefinita, finiscono sotto `artifacts/local/`.

## Requisiti minimi

Prima di eseguire il progetto, verifica questi punti:

1. il terminale deve essere aperto nella root del repository
2. Python deve essere disponibile nel sistema
3. il file `requirements.txt` deve essere installato
4. il file `yolov8n.pt` deve essere presente nella root del progetto

Esempio con ambiente virtuale:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS o Linux:

```bash
source .venv/bin/activate
```

Installazione dipendenze:

```bash
pip install -r requirements.txt
```

## Struttura pratica del repository

Queste sono le cartelle e i file che conviene conoscere subito:

- `run_experiment.py`: pipeline completa per dataset, training ed export
- `detect.py`: inferenza locale da webcam, cartella immagini, video o stream RTSP/RTMP
- `cloud_train.ipynb`: notebook cloud per Google Colab
- `configs/`: configurazioni YAML
- `base_fire_images/`: immagini di fuoco usate dal generatore sintetico
- `artifacts/`: output persistenti locali, cloud o importati
- `tools/`: utility aggiuntive, compreso configuratore cloud e model registry

## Come funzionano le configurazioni

Le configurazioni si trovano in `configs/` e sono organizzate in tre gruppi:

- `configs/presets/`: preset di esperimento pronti all'uso
- `configs/runtime/`: override per l'ambiente di esecuzione
- `configs/generated/`: configurazioni finali generate dal configuratore

I due file piu' importanti generati automaticamente sono:

- `configs/generated/latest.local.yaml`: configurazione locale piu' recente
- `configs/generated/latest.cloud.yaml`: configurazione cloud piu' recente

La GUI `tools/cloud/cloud_configurator.py` parte da un preset, applica un runtime override obbligatorio e salva una configurazione finale in `configs/generated/`.

## Primo uso in locale

### 1. Esegui uno smoke test

Il test piu' semplice per verificare che il progetto funzioni e' questo:

```bash
python run_experiment.py --config configs/presets/smoke.yaml
```

Questo comando esegue una run ridotta utile per controllare che generazione, training ed export siano operativi.

### 2. Esegui una run locale standard

Per una prova locale piu' realistica:

```bash
python run_experiment.py --config configs/presets/baseline-lite.yaml
```

Se preferisci preparare una configurazione guidata con interfaccia grafica:

```bash
python tools/cloud/cloud_configurator.py
```

Dopo il salvataggio, puoi eseguire la configurazione finale generata con:

```bash
python run_experiment.py --config configs/generated/latest.local.yaml
```

### 3. Esegui solo la generazione dataset

Se vuoi fermarti prima del training:

```bash
python run_experiment.py --config configs/generated/latest.local.yaml --skip-training
```

## Dove trovare gli output locali

Dopo una run locale completata, la struttura piu' importante e' questa:

```text
artifacts/local/
  datasets/
  runs/
  exports/
```

### Dataset

Percorso tipico:

```text
artifacts/local/datasets/<dataset-label>-<fingerprint>/
```

File principali:

- `dataset_manifest.yaml`
- `yolo_dataset.yaml`
- `images/train/`
- `images/val/`
- `labels/train/`
- `labels/val/`

### Run di training

Percorso tipico:

```text
artifacts/local/runs/<run-label>/
```

File principali:

- `resolved_config.yaml`
- `training_run.yaml`
- `pipeline_summary.yaml`

### Export del modello

Percorso:

```text
artifacts/local/exports/
```

File principali:

- `<run-label>.pt`
- `<run-label>.yaml`
- `latest.yaml`

`latest.yaml` permette a `detect.py` di trovare automaticamente il modello piu' recente.

## Come usare detect.py

`detect.py` usa di default il modello piu' recente disponibile. Se non passi `--weights`, il valore usato e' equivalente a `latest`.

### Webcam con selezione guidata

```bash
python detect.py
python detect.py --source webcam
```

Questa modalita' cerca le webcam disponibili e permette di scegliere quella corretta.

### Webcam con indice esplicito

```bash
python detect.py --source 0
python detect.py --source 1
```

### Cartella immagini

```bash
python detect.py --source path/to/images
```

### File video

```bash
python detect.py --source video.mp4
```

### Stream RTSP o RTMP

```bash
python detect.py --source rtsp://server/stream
python detect.py --source rtmp://server/app/stream
```

### Modello esplicito

```bash
python detect.py --source webcam --weights artifacts/local/exports/<run-label>.pt
```

### Parametri piu' utili

Confidenza:

```bash
python detect.py --source webcam --conf 0.5
```

Device:

```bash
python detect.py --source webcam --device cpu
python detect.py --source webcam --device 0
```

Valori pratici:

- `--conf 0.5`: punto di partenza consigliato
- `--conf 0.3`: piu' sensibile, ma con maggiore rischio di falsi positivi
- `--conf 0.7`: piu' severo, ma puo' perdere fuochi deboli
- `--device cpu`: usa il processore
- `--device 0`: usa la GPU CUDA 0

### Tasti durante l'inferenza

Per webcam, stream e video:

- `q` o `Esc`: chiude la sessione
- `s`: salva il frame corrente in `detections/`

Per cartelle di immagini:

- `a` o freccia sinistra: immagine precedente
- `d` o freccia destra: immagine successiva
- `s`: salva l'immagine annotata
- `q` o `Esc`: chiude la sessione

## Come scegliere il preset iniziale

Per la maggior parte dei casi, il preset piu' semplice da cui partire e':

```text
configs/presets/balanced-mini-fires.yaml
```

Se vuoi un controllo rapido della pipeline usa invece:

```text
configs/presets/smoke.yaml
```

Per una panoramica completa dei preset disponibili, consulta `TRAINING_PRESETS.md`.

## Uso cloud in sintesi

Il flusso cloud corretto e' questo:

1. apri `tools/cloud/cloud_configurator.py`
2. scegli un preset e un runtime cloud
3. salva la configurazione finale
4. esegui `python tools/cloud/prepare_cloud_bundle.py`
5. carica `dist/yolo-fire-detector-cloud.zip` su Colab o Google Drive
6. apri `cloud_train.ipynb`
7. esegui le celle nell'ordine proposto

Il notebook usa `configs/generated/latest.cloud.yaml` gia' inclusa nel bundle. Per la procedura completa, consulta `CLOUD_TRAINING.md`.

## Bundle cloud

Questo comando crea lo zip pronto per il notebook:

```bash
python tools/cloud/prepare_cloud_bundle.py
```

Output atteso:

```text
dist/yolo-fire-detector-cloud.zip
```

Il bundle include automaticamente codice Python, notebook, file YAML, documentazione Markdown, immagini PNG sotto `base_fire_images/` e i pesi `.pt` presenti nella root del progetto.

## Troubleshooting essenziale

### `detect.py` non trova un modello

Esegui prima una run completa, ad esempio:

```bash
python run_experiment.py --config configs/presets/default.yaml
```

Poi verifica che esista:

```text
artifacts/local/exports/latest.yaml
```

### Vuoi rigenerare il dataset anche se esiste gia'

Nella configurazione YAML:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi evitare il resume del training

Nella configurazione YAML:

```yaml
training:
  resume: never
```

### Vuoi usare la GPU per il training

Nella configurazione YAML:

```yaml
training:
  device: "0"
```

### Vuoi usare la GPU per l'inferenza

```bash
python detect.py --source webcam --device 0
```

## Documenti utili

- `CLOUD_TRAINING.md`: guida completa al flusso cloud
- `TRAINING_PRESETS.md`: scelta ragionata dei preset
- `tools/README.md`: utility accessorie, inclusi configuratore, dataset viewer e model registry
