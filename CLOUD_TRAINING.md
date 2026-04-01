# Cloud Training Guide

Guida operativa verificata contro il codice attuale del repository e contro `cloud_train.ipynb`.

## Obiettivo pratico

Se vuoi il flusso piu' semplice e consigliato:

1. prepara lo zip cloud in locale
2. carica `dist/yolo-fire-detector-cloud.zip` su Colab o Drive
3. apri `cloud_train.ipynb`
4. imposta `USE_READY_CONFIG_AS_IS = True`
5. imposta `READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'`
6. esegui tutte le celle in ordine

Questo preset usa i due asset trasparenti:

- `base_fire_images/fire1-mini_nobg.png`
- `base_fire_images/fire2-mini_nobg.png`

ed e' pensato come compromesso tra tempo di training, costo GPU e qualita' finale.

## Passo 1: prepara il bundle in locale

Dal terminale aperto nella root del repository:

```bash
python tools/cloud/prepare_cloud_bundle.py
```

Output atteso:

```text
dist/yolo-fire-detector-cloud.zip
```

Questo e' il nome che il notebook cerca automaticamente.

## Passo 2: carica lo zip

Puoi mettere `yolo-fire-detector-cloud.zip` in uno di questi posti:

- `/content`
- Google Drive

Non serve copiare a mano file dentro la cartella persistente finale: ci pensa il notebook.

## Passo 3: apri il notebook cloud

Apri `cloud_train.ipynb` in Colab e fai subito queste due cose:

1. `Runtime -> Change runtime type`
2. seleziona `GPU`

Poi esegui le celle dall'alto verso il basso, senza saltarne una a meta' la prima volta.

## Passo 4: cosa fanno le prime celle

Le celle iniziali del notebook fanno automaticamente questo:

1. montano Google Drive se l'ambiente e' Colab
2. definiscono la root persistente, di default `/content/drive/MyDrive/yolo-fire-detector`
3. cercano `yolo-fire-detector-cloud.zip`
4. copiano lo zip in `<PERSISTENT_ROOT>/inputs/`
5. estraggono il repository in `<PERSISTENT_ROOT>/repo/`

Quindi il file management manuale serve solo per caricare lo zip all'inizio.

## Passo 5: scegli la config runtime

Il notebook costruisce sempre una config finale in:

```text
repo/configs/cloud.runtime.yaml
```

Hai due modalita'.

### Modalita' A: usa un preset pronto senza modificarlo a mano

Usa questa modalita' se sei studente e vuoi partire senza complicarti il flusso.

Imposta nella cella di configurazione:

```python
USE_READY_CONFIG_AS_IS = True
READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'
```

Effetto reale:

- il notebook legge `configs/cloud.balanced-mini-fires.yaml`
- aggiorna solo `project.persistent_root`
- salva il risultato come `configs/cloud.runtime.yaml`

### Modalita' B: parti da una base e cambi i parametri principali

Usa questa modalita' se vuoi fare esperimenti mantenendo una base ragionevole.

Imposta per esempio:

```python
USE_READY_CONFIG_AS_IS = False
BASE_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'
```

Poi modifica i campi esposti nella cella, tipicamente:

- `DATASET_LABEL`
- `TRAINING_LABEL`
- `NUM_IMAGES`
- `IMAGE_SIZE`
- `EPOCHS`
- `BATCH_SIZE`
- `MODEL_SIZE`
- `DEVICE`

Se ti serve una modifica piu' fine, usa `MANUAL_CONFIG_PATCH` nella cella successiva.

## Passo 6: lancia il training

La cella finale del notebook esegue:

```bash
python run_experiment.py --config configs/cloud.runtime.yaml
```

Non devi lanciare separatamente `generator.py` o `train.py`.

## Dove finiscono davvero gli output

La pipeline scrive sotto `project.persistent_root`, che in cloud di default e':

```text
/content/drive/MyDrive/yolo-fire-detector
```

### Dataset

Percorso:

```text
datasets/<dataset-label>-<fingerprint>/
```

File importanti:

- `dataset_manifest.yaml`
- `yolo_dataset.yaml`
- `images/train/`
- `images/val/`
- `labels/train/`
- `labels/val/`

### Run di training

Percorso:

```text
runs/<run_label>/
```

File importanti:

- `resolved_config.yaml`
- `training_run.yaml`
- `pipeline_summary.yaml`
- eventuali file YOLO di diagnostica

Durante il training possono esserci anche `weights/best.pt` e `weights/last.pt`.
Se la run si chiude correttamente, i checkpoint pesanti vengono ripuliti dalla pipeline.

### Export finali per inferenza

Percorso:

```text
exports/
```

File importanti:

- `exports/<run_label>.pt`
- `exports/<run_label>.yaml`
- `exports/latest.yaml`

`detect.py` usa proprio `exports/latest.yaml` per trovare il modello di default.

## Come riportare il modello dal cloud al PC locale

Il modo piu' semplice e' scaricare almeno questi file dalla cartella `exports/`:

- `latest.yaml`
- il file `.pt` indicato dentro `latest.yaml`
- opzionalmente il metadata `.yaml` della stessa run

Se vuoi usare il detector locale senza cambiare codice, la struttura piu' comoda sul PC e':

```text
artifacts/local/exports/
```

con dentro i file esportati dal cloud.

## Resume e riuso dataset

Se riapri una sessione Colab, il flusso corretto e' rieseguire il notebook.

La pipeline gestisce automaticamente:

- riuso del dataset se il fingerprint coincide
- resume del training se la run e' compatibile e `weights/last.pt` esiste ancora

## Troubleshooting minimo

### Il notebook non trova lo zip

Metti `yolo-fire-detector-cloud.zip` in:

- `/content`
- oppure Google Drive

Poi riesegui la cella di sync.

### Vuoi forzare l'aggiornamento del repository estratto

Imposta:

```python
FORCE_PROJECT_REFRESH = True
```

### Vuoi rigenerare il dataset anche se esiste gia'

In `cloud.runtime.yaml` o in `MANUAL_CONFIG_PATCH`:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi evitare il resume del training

In `cloud.runtime.yaml` o in `MANUAL_CONFIG_PATCH`:

```yaml
training:
  resume: never
```

## Checklist finale

1. esegui `python tools/cloud/prepare_cloud_bundle.py`
2. verifica che esista `dist/yolo-fire-detector-cloud.zip`
3. carica lo zip su Colab o Drive
4. apri `cloud_train.ipynb`
5. imposta `USE_READY_CONFIG_AS_IS = True`
6. imposta `READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'`
7. esegui le celle in ordine
8. controlla i risultati sotto `datasets/`, `runs/` ed `exports/`
9. copia in locale gli export che ti servono per `detect.py`
