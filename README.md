# YOLO Fire Detector

Repository per generare dataset sintetici di fuoco, addestrare YOLOv8 tramite file YAML e usare il modello finale in inferenza locale o cloud.

Autore: Mirto Musci

## Cosa fa il progetto

La pipeline completa e' questa:

1. prende una o piu' immagini base del fuoco con alpha
2. genera un dataset sintetico YOLO in `datasets/`
3. allena un modello YOLOv8 in `runs/`
4. registra l'export finale in `exports/`
5. permette di usare quel modello con `detect.py`

L'entrypoint corretto per training e generazione non e' `generator.py` o `train.py` separatamente, ma:

```bash
python run_experiment.py --config <config.yaml>
```

## Avvio rapido per studenti

Se vuoi solo capire come usare il progetto senza perderti:

### Uso locale minimo

```bash
pip install -r requirements.txt
python run_experiment.py --config configs/local.smoke.yaml
python detect.py --source webcam
```

### Uso cloud consigliato

```bash
python tools/cloud/prepare_cloud_bundle.py
```

Poi apri `cloud_train.ipynb` in Colab e usa:

```python
USE_READY_CONFIG_AS_IS = True
READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'
```

## Prerequisiti locali

### 1. Apri il terminale nella root del progetto

Devi trovarti nella cartella che contiene file come:

- `run_experiment.py`
- `detect.py`
- `requirements.txt`
- `configs/`

### 2. Installa le dipendenze Python

```bash
pip install -r requirements.txt
```

Il progetto usa `opencv-python`, non `opencv-python-headless`.

### 3. Verifica che il peso base esista

Nella root del repository deve esserci:

```text
yolov8n.pt
```

Serve come base per i preset che usano modello `n`.

## Come usare il progetto in locale

## Flusso locale corretto

Il flusso locale standard e' questo:

1. esegui una config YAML con `run_experiment.py`
2. lascia che la pipeline generi dataset, training ed export
3. verifica i file creati in `artifacts/local/`
4. usa `detect.py` sul modello appena esportato

## Passo 1: smoke test end-to-end

Se vuoi verificare che tutto funzioni prima di una run vera:

```bash
python run_experiment.py --config configs/local.smoke.yaml
```

Questa run fa un controllo completo ma piccolo:

- genera un dataset ridotto
- allena 1 epoca
- scrive i metadata di run
- crea l'export finale
- aggiorna `artifacts/local/exports/latest.yaml`

## Passo 2: run locale normale

Per una run locale standard:

```bash
python run_experiment.py --config configs/local.default.yaml
```

Questa config usa per default:

- ambiente `local`
- output in `artifacts/local`
- dataset sintetico da 100 immagini
- YOLOv8n
- training su CPU

Se vuoi piu' velocita' e hai CUDA, cambia `training.device` nella config in `0`.

## Passo 3: controlla gli output

Dopo una run riuscita, la struttura importante e' questa:

```text
artifacts/local/
  datasets/
  runs/
  exports/
```

### Dataset locale

Percorso:

```text
artifacts/local/datasets/<dataset-label>-<fingerprint>/
```

File importanti:

- `dataset_manifest.yaml`
- `yolo_dataset.yaml`
- `images/train/`
- `images/val/`
- `labels/train/`
- `labels/val/`

### Run locale

Percorso:

```text
artifacts/local/runs/<run_label>/
```

File importanti:

- `resolved_config.yaml`
- `training_run.yaml`
- `pipeline_summary.yaml`

### Export locale

Percorso:

```text
artifacts/local/exports/
```

File importanti:

- `<run_label>.pt`
- `<run_label>.yaml`
- `latest.yaml`

`latest.yaml` e' il puntatore che `detect.py` usa automaticamente quando non passi `--weights`.

## Passo 4: avvia il detector

Per usare l'ultimo export locale:

```bash
python detect.py --source webcam
```

Per usare un peso specifico:

```bash
python detect.py --weights artifacts/local/exports/<run_label>.pt --source webcam
```

## Come usare `detect.py`

`detect.py` supporta piu' sorgenti.

### Webcam con selezione guidata

```bash
python detect.py
python detect.py --source webcam
```

Comportamento:

- scansiona le webcam disponibili
- mostra gli ID trovati
- evidenzia le camere che restituiscono un frame valido
- chiede quale aprire

### Webcam diretta per ID

```bash
python detect.py --source 0
python detect.py --source 1
```

### Cartella immagini

```bash
python detect.py --source path/to/images
```

### Video locale

```bash
python detect.py --source video.mp4
```

### Stream RTMP o RTSP

```bash
python detect.py --source rtmp://server/app/stream
python detect.py --source rtsp://server/stream
```

## Parametri utili di `detect.py`

### Confidenza

```bash
python detect.py --source webcam --conf 0.3
python detect.py --source webcam --conf 0.7
```

Interpretazione pratica:

- `0.3`: piu' sensibile, ma aumenta i falsi positivi
- `0.5`: buon punto di partenza
- `0.7`: piu' severo, ma rischia di perdere fuochi deboli

### Device

```bash
python detect.py --source webcam --device cpu
python detect.py --source webcam --device 0
```

Interpretazione pratica:

- `cpu`: usa il processore
- `0`: usa la GPU CUDA 0

## Tasti durante l'inferenza

Per webcam, video e stream:

- `q` o `Esc`: esce
- `s`: salva il frame corrente in `detections/`

Per cartella immagini:

- `a` o freccia sinistra: immagine precedente
- `d` o freccia destra: immagine successiva
- `s`: salva l'immagine annotata
- `q` o `Esc`: esce

## Come cambiare le immagini base del fuoco

Le immagini base si impostano nella chiave YAML `dataset.fire_image_paths`.

Esempio con i due asset mini trasparenti:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire1-mini_nobg.png
    - base_fire_images/fire2-mini_nobg.png
```

Se la lista contiene piu' immagini, la generazione sceglie casualmente tra quelle specificate.

## Config YAML principali

Le configurazioni stanno in `configs/`.

### Locali

- `configs/local.smoke.yaml`: test end-to-end rapido
- `configs/local.default.yaml`: baseline locale semplice
- `configs/local.smoke.yaml`: utile per controlli di installazione o debug

### Cloud

- `configs/cloud.balanced-mini-fires.yaml`: preset consigliato per studenti, usa `fire1-mini_nobg` e `fire2-mini_nobg`
- `configs/cloud.default.yaml`: baseline cloud semplice
- `configs/cloud.quick-screen.yaml`: run rapida per scremare idee
- `configs/cloud.recall.yaml`: spinge la recall
- `configs/cloud.robust.yaml`: spinge la robustezza
- `configs/cloud.small-fire.yaml`: ottimizzato per fuochi piccoli
- `configs/cloud.hard-negatives.yaml`: utile contro i falsi positivi
- `configs/cloud.capacity.yaml`: piu' costoso, punta alla massima capacita'
- `configs/cloud.fast-debug.yaml`: debug veloce della pipeline

## Preset cloud consigliato

Il preset consigliato per partire e':

```text
configs/cloud.balanced-mini-fires.yaml
```

Perche' e' un buon compromesso:

- usa due asset trasparenti gia' pronti
- mantiene YOLOv8n, quindi training e inferenza restano leggeri
- aumenta il dataset rispetto alla baseline base
- introduce un po' piu' di variabilita' sintetica senza diventare costoso come i preset piu' pesanti

## Come usare il progetto in cloud

Per il flusso cloud dettagliato usa [CLOUD_TRAINING.md](CLOUD_TRAINING.md).

Sintesi operativa:

1. esegui `python tools/cloud/prepare_cloud_bundle.py`
2. ottieni `dist/yolo-fire-detector-cloud.zip`
3. carica quello zip in Colab o Drive
4. apri `cloud_train.ipynb`
5. imposta `USE_READY_CONFIG_AS_IS = True`
6. imposta `READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'`
7. esegui le celle in ordine

## Bundle cloud

Lo script:

```bash
python tools/cloud/prepare_cloud_bundle.py
```

crea uno zip pronto per Colab in `dist/yolo-fire-detector-cloud.zip`.

Il bundle include automaticamente:

- file Python
- notebook
- file YAML
- documentazione Markdown
- PNG dentro `base_fire_images/`
- pesi `.pt` in root come `yolov8n.pt`

## Troubleshooting rapido

### `detect.py` non trova il modello

Esegui prima almeno una run completa:

```bash
python run_experiment.py --config configs/local.default.yaml
```

Poi controlla che esista:

```text
artifacts/local/exports/latest.yaml
```

### Vuoi rigenerare il dataset anche se esiste gia'

Nella config:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi disabilitare il resume del training

Nella config:

```yaml
training:
  resume: never
```

### Vuoi usare la GPU

In training, imposta nella config:

```yaml
training:
  device: "0"
```

In detect:

```bash
python detect.py --source webcam --device 0
```

## File principali

- `run_experiment.py`: entrypoint corretto per dataset + training + export
- `detect.py`: inferenza locale su webcam, immagini, video e stream
- `cloud_train.ipynb`: notebook cloud per Colab
- `tools/cloud/prepare_cloud_bundle.py`: crea lo zip da caricare nel cloud
- `configs/`: preset locali e cloud
- `base_fire_images/`: immagini base del fuoco usate dal generatore

- `run_experiment.py`: pipeline generation + training
- `detect.py`: inferenza da webcam, stream, video o immagini
- `tools/cloud/prepare_cloud_bundle.py`: zip per Colab
- `cloud_train.ipynb`: notebook cloud persistente
- `TRAINING_PRESETS.md`: panoramica preset cloud
