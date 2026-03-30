# Cloud Training Guide

Guida operativa breve per usare `cloud_train.ipynb` con gli output attuali del progetto.

## Passo 1: prepara il bundle in locale

Da VS Code locale:

```bash
python prepare_cloud_bundle.py
```

Output atteso:

```text
dist/yolo-fire-detector-cloud.zip
```

Questo e' il nome che il notebook cerca automaticamente.

## Passo 2: carica lo zip dove ti e' piu' comodo

Puoi mettere `yolo-fire-detector-cloud.zip` in uno di questi posti:

- `/content`
- Google Drive

Non serve piu' spostarlo a mano nella cartella persistente finale.

## Passo 3: apri `cloud_train.ipynb`

In Colab:

1. apri il notebook
2. abilita la GPU in Runtime -> Change runtime type
3. esegui le celle in ordine

## Passo 4: lascia lavorare il notebook

Le prime celle fanno automaticamente questo:

1. montano Drive se sei in Colab
2. creano la root persistente se non esiste
3. cercano `yolo-fire-detector-cloud.zip` nel contesto locale e in Drive
4. spostano lo zip in:

```text
<PERSISTENT_ROOT>/inputs/
```

5. estraggono il repository in:

```text
<PERSISTENT_ROOT>/repo/
```

Quindi non devi fare file management manuale.

## Passo 5: scegli la config runtime

Nel notebook hai due modalita'.

### Modalita' A: usa una config gia' pronta senza toccare i parametri

Imposta:

- `USE_READY_CONFIG_AS_IS = True`
- `READY_CONFIG_NAME = 'cloud.recall.yaml'`

In questo caso il notebook prende quella config cosi' com'e' e cambia solo `project.persistent_root`.

### Modalita' B: parti da un preset e poi sovrascrivi i campi principali

Imposta:

- `USE_READY_CONFIG_AS_IS = False`
- `BASE_CONFIG_NAME = 'cloud.default.yaml'`

e poi eventualmente modifichi:

- `BASE_CONFIG_NAME`
- `DATASET_LABEL`
- `TRAINING_LABEL`
- `NUM_IMAGES`
- `IMAGE_SIZE`
- `EPOCHS`
- `BATCH_SIZE`
- `MODEL_SIZE`
- `DEVICE`

Il notebook genera poi:

```text
repo/configs/cloud.runtime.yaml
```

## Passo 6: lancia il training

La cella finale esegue:

```bash
python run_experiment.py --config configs/cloud.runtime.yaml
```

Non devi lanciare `generator.py` o `train.py` separatamente.

## Dove finiscono gli output

In Colab, di default, tutto finisce sotto:

```text
/content/drive/MyDrive/yolo-fire-detector
```

### `datasets/`

Ogni dataset sta in:

```text
datasets/<dataset-label>-<fingerprint>/
```

Dentro trovi:

- `yolo_dataset.yaml`
- `dataset_manifest.yaml`
- `images/`
- `labels/`

### `runs/`

Ogni run sta in:

```text
runs/<run_label>/
```

Dentro trovi almeno:

- `resolved_config.yaml`
- `pipeline_summary.yaml`
- `training_run.yaml`
- metriche e diagnostica YOLO

Durante il training possono esserci i checkpoint in `weights/`.
Quando la run si chiude con successo, i checkpoint di resume vengono rimossi automaticamente.

### `exports/`

Qui trovi quello che serve per l'inferenza finale:

- `exports/<run_label>.pt`
- `exports/<run_label>.yaml`
- `exports/latest.yaml`

`latest.yaml` punta all'export corrente.

## Come salvare gli export sul PC locale

L'ultima cella del notebook:

1. legge `exports/latest.yaml`
2. comprime tutta la cartella `exports/` in:

```text
<PERSISTENT_ROOT>/downloads/yolo-fire-detector-exports.zip
```

3. stampa il comando Colab per scaricare quello zip sul PC locale

Quello zip e' pensato per essere portato poi in locale e usato con `detect.py`.

## Cosa succede se riapri una sessione

Il flusso corretto e' semplicemente rieseguire il notebook.

La pipeline gestisce automaticamente:

- riuso del dataset se il fingerprint coincide
- resume del training se la run e' compatibile e c'e' un checkpoint valido

## Troubleshooting minimo

### Il notebook non trova lo zip

Metti `yolo-fire-detector-cloud.zip` in uno di questi posti e riesegui la cella di sync:

- `/content`
- Google Drive

### Vuoi forzare l'aggiornamento del codice estratto

Nella cella di sync imposta:

```python
FORCE_PROJECT_REFRESH = True
```

### Vuoi rigenerare il dataset

Nella config runtime imposta:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi evitare il resume

Nella config runtime imposta:

```yaml
training:
  resume: never
```

## Checklist finale

1. `python prepare_cloud_bundle.py`
2. carichi `dist/yolo-fire-detector-cloud.zip`
3. apri `cloud_train.ipynb`
4. scegli `USE_READY_CONFIG_AS_IS = True` se vuoi lanciare una config pronta senza modifiche manuali
5. esegui le celle in ordine
6. trovi i risultati in `datasets/`, `runs/`, `exports/`
7. scarichi `downloads/yolo-fire-detector-exports.zip` sul PC locale
