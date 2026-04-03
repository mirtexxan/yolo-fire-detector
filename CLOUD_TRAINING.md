# Cloud Training Guide

Questa guida spiega il flusso corretto per eseguire il progetto in Google Colab usando il notebook `cloud_train.ipynb`.

## Percorso consigliato

Se vuoi arrivare rapidamente a un training cloud funzionante, segui questo ordine:

1. genera in locale `configs/generated/latest.cloud.yaml`
2. esegui `python tools/cloud/prepare_cloud_bundle.py`
3. carica `cloud_train.ipynb` su colab o riaprilo
4. seleziona un runtime con GPU
5. carica `yolo-fire-detector-cloud.zip` in Colab o Google Drive
6. esegui tutte le celle in ordine

Il notebook usa direttamente `configs/generated/latest.cloud.yaml` contenuta nel bundle. Non richiede una selezione manuale della configurazione all'avvio e non richiede di preparare a mano i percorsi persistenti.

## Prima di iniziare

In locale devi avere questi elementi pronti:

- il repository aggiornato
- le dipendenze Python installate
- configuratore disponibile: `tools/cloud/cloud_configurator.py`

I preset (`configs/presets/dataset/` e `configs/presets/training/`) sono opzionali: puoi partire anche da configurazione manuale in tab `Generale`.

## Passo 1: genera la configurazione cloud

Apri il configuratore:

```bash
python tools/cloud/cloud_configurator.py
```

Operazioni consigliate:

1. imposta `Ambiente = Cloud`
2. verifica etichette, immagini di fuoco e parametri principali
3. applica preset dataset/training se servono (opzionale)
4. salva la configurazione finale

Il configuratore aggiorna questi file:

- `configs/generated/latest.cloud.yaml`
- `configs/generated/latest.cloud.meta.yaml`

Se selezioni ambiente `Local`, verra' aggiornato invece `latest.local.yaml`. Per il notebook cloud serve sempre `latest.cloud.yaml`.

## Passo 2: crea il bundle cloud

Esegui:

```bash
python tools/cloud/prepare_cloud_bundle.py
```

Output atteso:

```text
dist/yolo-fire-detector-cloud.zip
```

Se `configs/generated/latest.cloud.yaml` manca, questo comando termina con errore. In quel caso devi prima rigenerare la configurazione cloud.

Il bundle include automaticamente:

- file Python del progetto
- `cloud_train.ipynb`
- file YAML necessari
- documentazione Markdown
- immagini PNG in `base_fire_images/`
- pesi `.pt` presenti nella root del repository, come `yolov8n.pt`

## Passo 3: apri il notebook in Colab

Carica `cloud_train.ipynb` su Colab oppure riaprilo se lo avevi gia' usato in precedenza.

Prima di fare altro, imposta il runtime GPU:

1. `Runtime`
2. `Change runtime type`
3. seleziona `GPU`

Questo ordine e' importante per evitare che l'ambiente parta con un runtime sbagliato e debba poi essere ricreato o ripulito.

## Passo 4: carica il bundle

Dopo avere aperto il notebook e impostato il runtime corretto, carica `yolo-fire-detector-cloud.zip` in uno di questi posti:

- `/content` nella sessione Colab
- Google Drive

Non serve estrarre manualmente il file zip: lo fa il notebook.

## Passo 5: esegui le celle in ordine

Esegui le celle dall'alto verso il basso, senza saltarne nessuna durante il primo avvio.

## Se riapri o fai ripartire una sessione

Se riapri il notebook o fai ripartire il runtime, il comportamento corretto e' semplice: riesegui il notebook dall'inizio, nello stesso ordine.

La pipeline puo' riusare automaticamente cio' che trova gia' sotto la root persistente:

- il dataset, se la configurazione produce lo stesso fingerprint e `dataset.force_regenerate` e' `false`
- la run di training, se `training.resume` lo consente e il checkpoint richiesto esiste ancora
- gli export gia' prodotti, che restano disponibili nella cartella `exports/`

In pratica, se chiudi Colab e poi riapri tutto, non perdi per forza il lavoro: quello che conta e' cio' che e' stato salvato nella cartella persistente.

## Cosa fanno le celle iniziali

Le prime celle del notebook eseguono automaticamente queste operazioni:

1. montano Google Drive quando l'ambiente e' Colab
2. creano la root persistente della sessione e le sottocartelle necessarie
3. cercano il bundle zip nei percorsi attesi
4. copiano il bundle nella cartella `inputs/`
5. estraggono il repository dentro `repo/`
6. leggono la configurazione cloud gia' inclusa nel bundle

Questo significa che, dopo avere aperto il notebook nel runtime corretto e caricato il file zip, la preparazione restante viene gestita automaticamente.

## Cambiare configurazione

Il notebook legge solo `configs/generated/latest.cloud.yaml`.

Se vuoi cambiare configurazione in modo stabile, il metodo corretto e' rigenerare il file in locale e creare un nuovo bundle.

Se invece vuoi fare un cambiamento temporaneo direttamente in Colab, il notebook contiene una cella speciale che sostituisce rapidamente `configs/generated/latest.cloud.yaml` senza ricreare il bundle.

Questa cella e' utile solo quando devi fare una modifica rapida in Colab.

La cella:

1. carica un singolo file YAML dal runtime
2. controlla che sia una configurazione valida
3. rifiuta configurazioni cloud non sicure, ad esempio con `training.device: cpu`
4. salva un backup della configurazione precedente
5. sostituisce `repo/configs/generated/latest.cloud.yaml`

Dopo una sostituzione riuscita:

1. riesegui la cella di preparazione config
2. se avevi gia' avviato la pipeline con la configurazione precedente, riesegui anche la cella di esecuzione pipeline
3. aggiorna la cella finale degli output solo se vuoi un riepilogo coerente con la nuova run

## Avvio del training

La cella di esecuzione lancia questo comando:

```bash
python run_experiment.py --config configs/generated/latest.cloud.yaml
```

Non e' necessario eseguire manualmente altri script.

## Output

Tutti gli output cloud vengono salvati sotto `project.persistent_root`.

`persistent_root` e' la cartella principale in cui il progetto conserva dati e risultati tra una sessione e l'altra. Dentro quella cartella finiscono dataset, run di training, export finali e file di appoggio del notebook.

Nel flusso cloud standard, il valore usato e' questo:

```text
/content/drive/MyDrive/yolo-fire-detector
```

In pratica, se Google Drive e' montato correttamente, questa cartella e' il punto in cui Colab ritrova il lavoro gia' salvato anche dopo una riapertura.

### Come si cambia `persistent_root`

Il modo piu' semplice per cambiarlo in modo stabile e' usare il configuratore:

```bash
python tools/cloud/cloud_configurator.py
```

Nel flusso attuale non devi impostarlo a mano nel configuratore: il notebook cloud sovrascrive automaticamente `project.persistent_root` in base all'ambiente Colab/Drive.


### Cosa trovi dentro `persistent_root`

La struttura piu' importante e' questa:

```text
<persistent_root>/
  datasets/
  runs/
  exports/
  inputs/
  repo/
```

Significato pratico:

- `datasets/`: dataset YOLO generati o riusati
- `runs/`: cartelle delle run di training
- `exports/`: modelli finali pronti per l'inferenza
- `inputs/`: copia persistente del bundle zip caricato
- `repo/`: copia estratta del progetto usata dal notebook


### Come riportare il modello sul computer locale

Per usare in locale un modello addestrato in cloud, scarica almeno questi file da `exports/`:

- `latest.yaml`
- il file `.pt` indicato in `latest.yaml`
- il file `.yaml` associato alla stessa run, se disponibile

Se vuoi mantenere separati i modelli importati dal flusso locale, copia questi file in:

```text
artifacts/imported/
```

In questo modo `detect.py` puo' usare i modelli importati senza confonderli con gli export locali.

## Problemi comuni

### Il notebook non trova il bundle zip

Verifica che `yolo-fire-detector-cloud.zip` sia presente in uno dei percorsi cercati dal notebook:

- `/content`
- Google Drive

Poi riesegui la cella di sincronizzazione del progetto.

### Errore: GPU obbligatoria ma CUDA non disponibile

Questo errore indica che il runtime Colab non sta esponendo una GPU valida a PyTorch.

Controlli utili:

1. imposta un runtime GPU in Colab
2. riavvia il runtime se hai cambiato il tipo di runtime dopo l'apertura del notebook
3. controlla `torch.cuda.is_available()`
4. controlla l'output di `nvidia-smi`

### Vuoi rigenerare sempre il dataset

Nella configurazione cloud:

```yaml
dataset:
  force_regenerate: true
```

### Vuoi evitare il resume del training

Nella configurazione cloud:

```yaml
training:
  resume: never
```