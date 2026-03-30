# Cloud Training Guide

Questo documento descrive come usare il repository in cloud nello stato attuale del progetto.

La logica corretta oggi e' questa:

- l'entrypoint di training e generazione e' `run_experiment.py`
- il notebook cloud supportato e' `cloud_train.ipynb`
- la configurazione cloud di base e' `configs/cloud.default.yaml`
- il bundle da caricare in cloud si prepara con `prepare_cloud_bundle.py`

La piattaforma consigliata e' Google Colab, perche' il notebook attuale e' pensato prima di tutto per Drive persistente. In fondo trovi anche la variante Kaggle.

## Obiettivo del workflow cloud

Il workflow cloud non deve limitarsi a lanciare un training una tantum. Deve anche:

- riusare un dataset se i parametri richiesti sono identici
- salvare sempre checkpoint, export e metadati nello stesso punto persistente
- ripartire in resume se trova `weights/last.pt`
- separare in cartelle diverse dataset e run con parametri diversi

Questo e' esattamente quello che fa la pipeline attuale.

## File da conoscere

- `prepare_cloud_bundle.py`
- `cloud_train.ipynb`
- `run_experiment.py`
- `configs/cloud.default.yaml`

## Passo 1: prepara il bundle dal repository locale

Da VS Code locale:

```bash
python prepare_cloud_bundle.py
```

Questo crea:

```text
dist/yolo-fire-detector-cloud.zip
```

Quello zip contiene il codice necessario per il training cloud, incluso il notebook e i file di configurazione.

## Passo 2: apri Colab e abilita la GPU

In Colab:

1. apri un nuovo notebook oppure carica `cloud_train.ipynb`
2. vai su Runtime -> Change runtime type
3. seleziona GPU

## Passo 3: carica o rendi disponibile il bundle

Hai due opzioni pratiche:

### Opzione A: upload diretto in sessione

Carica `yolo-fire-detector-cloud.zip` in `/content`.

### Opzione B: salvalo in Drive

Carica `yolo-fire-detector-cloud.zip` in una cartella di Google Drive e poi copialo nella root usata dal notebook.

Il notebook cerca prima di tutto il file in una root persistente, quindi questa opzione e' la piu' comoda se prevedi piu' sessioni.

## Passo 4: esegui il notebook `cloud_train.ipynb`

Il notebook attuale fa questo, in ordine:

1. monta Google Drive su Colab
2. usa come root persistente:

```text
/content/drive/MyDrive/yolo-fire-detector
```

3. prepara le cartelle persistenti:

```text
repo/
inputs/
datasets/
runs/
models/
```

4. cerca `yolo-fire-detector-cloud.zip`
5. se necessario aggiorna il codice dentro `repo/`
6. installa le dipendenze da `requirements.txt`
7. costruisce `configs/cloud.runtime.yaml`
8. lancia:

```bash
python run_experiment.py --config configs/cloud.runtime.yaml
```

## Perche' Drive e' importante

Se non salvi su Drive, Colab perde tutto quando la sessione finisce.

Con la root persistente attuale invece conservi:

- dataset generati
- manifest JSON del dataset
- checkpoint YOLO
- config risolte
- export finali
- registry dei modelli

Questo e' il motivo per cui il notebook e' costruito intorno a una root fissa e non a una cartella temporanea in `/content`.

## Come funziona il riuso del dataset

La pipeline legge i parametri della config e costruisce un fingerprint del dataset richiesto usando, tra le altre cose:

- label dataset
- numero di immagini
- dimensione immagine
- negative ratio
- train split
- seed
- impostazioni di trasformazione
- impostazioni dataset rilevanti
- lista esplicita delle immagini base (`dataset.fire_image_paths`, es. `base_fire_images/fire.png`)

Il dataset viene scritto in una cartella come questa:

```text
datasets/<dataset_label>__<fingerprint>/
```

E dentro trovi anche:

```text
dataset_info.json
```

Se un dataset compatibile esiste gia', la pipeline non lo rigenera.

## Come funziona il resume del training

La run ha una label leggibile che incorpora:

- ambiente
- label progetto
- label training
- modello YOLO
- label dataset
- fingerprint dataset

Per esempio:

```text
cloud__drive-persistent__yolov8n-cloud__yolov8n__synthetic-fire-cloud__abc123def4
```

La run viene salvata in:

```text
runs/<run_label>/
```

Se nella run esiste:

```text
weights/last.pt
```

e la config ha:

```yaml
training:
  resume: auto
```

allora la pipeline riparte automaticamente da quel checkpoint.

## Come cambiare i parametri

Il modo corretto non e' modificare a mano il codice Python del notebook. Il modo corretto e':

1. partire da `configs/cloud.default.yaml`
2. nel notebook scrivere una `cloud.runtime.yaml`
3. cambiare li' i parametri di dataset o training
4. lanciare la pipeline

Per scegliere una sola immagine:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire2.png
```

Per scegliere un sottoinsieme di immagini:

```yaml
dataset:
  fire_image_paths:
    - base_fire_images/fire2.png
    - base_fire_images/fire.png
```

Se la lista contiene una sola immagine, viene usata solo quella. Se contiene piu' immagini, la pipeline sceglie casualmente tra quelle presenti. Se non specifichi niente, il default resta una lista con `base_fire_images/fire.png`.

In pratica il notebook gia' fa questa parte per te.

I parametri che normalmente cambi sono:

- `dataset.label`
- `dataset.num_images`
- `dataset.image_size`
- `dataset.force_regenerate`
- `training.label`
- `training.model_size`
- `training.epochs`
- `training.batch_size`
- `training.device`
- `training.resume`

## Output finali in Colab

Dopo il training troverai tutto sotto:

```text
/content/drive/MyDrive/yolo-fire-detector
```

Le cartelle principali sono:

### `datasets/`

Contiene i dataset sintetici generati, uno per fingerprint.

### `runs/`

Contiene le run YOLO complete, quindi:

- checkpoint
- metriche
- immagini di training/validation generate da YOLO
- `resolved_config.yaml`
- `pipeline_summary.json`
- `final_export/`

### `models/`

Contiene il registry persistente dei modelli finali:

- `models/<run_label>.pt`
- `models/<run_label>.json`
- `models/latest.json`

## Flusso operativo consigliato su Colab

Questo e' il percorso concreto piu' robusto:

1. esegui `python prepare_cloud_bundle.py` in locale
2. carica lo zip in Colab o in Drive
3. apri `cloud_train.ipynb`
4. lascia la root persistente su Drive
5. imposta i parametri nella cella di config runtime
6. esegui le celle in ordine
7. se la sessione cade, riapri il notebook e rilancia: dataset e checkpoint verranno riusati quando compatibili

## Troubleshooting Colab

### Il notebook non trova lo zip

Metti `yolo-fire-detector-cloud.zip` in uno dei percorsi attesi dal notebook oppure aggiorna la cella che definisce `ZIP_CANDIDATES`.

### Vuoi forzare l'aggiornamento del codice

Imposta `FORCE_PROJECT_REFRESH = True` nella cella di sync del progetto.

### Vuoi rigenerare il dataset

Nella cella che costruisce `cloud.runtime.yaml`, imposta:

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

### Vuoi cambiare cartella persistente

Aggiorna `PERSISTENT_ROOT` nella prima cella del notebook.

## Kaggle

Il notebook attuale supporta anche Kaggle, ma Colab resta la piattaforma primaria perche' il flusso con Google Drive e' piu' naturale.

Su Kaggle la root persistente prevista e':

```text
/kaggle/working/yolo-fire-detector
```

Il principio resta identico:

- estrai il progetto nella root prevista
- installa le dipendenze
- genera `cloud.runtime.yaml`
- lancia `run_experiment.py`
- conserva dataset, run e modelli dentro la stessa root di lavoro

La differenza pratica e' che su Kaggle dovrai di solito preparare prima il file zip come input del notebook.

## In sintesi

Lo stato attuale del repository in cloud e' questo:

- un solo entrypoint: `run_experiment.py`
- un solo notebook cloud supportato: `cloud_train.ipynb`
- una root persistente stabile
- caching del dataset via fingerprint
- resume automatico della run
- registry persistente dei modelli finali

Se devi usare il repo in cloud, questo e' il percorso corretto.
