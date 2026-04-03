# Tools

Questa cartella contiene utility accessorie. Non sono obbligatorie per il flusso minimo del progetto, ma rendono piu' semplice configurazione, ispezione dataset e gestione dei modelli.

## Quali tool conviene conoscere subito

Se stai iniziando da zero, i tool piu' utili sono questi:

1. `tools/cloud/cloud_configurator.py`
2. `tools/cloud/prepare_cloud_bundle.py`
3. `tools/model_registry/drive_model_sync.py`

Il dataset viewer e' utile soprattutto dopo aver gia' generato un dataset.

## Tool cloud

### `tools/cloud/cloud_configurator.py`

E' una GUI Tkinter per creare una configurazione finale senza modificare i file YAML manualmente.

Funzioni principali:

- workflow in 3 tab (`Generale`, `Presets`, `Avanzate`)
- preset dataset/training opzionali da `configs/presets/dataset/` e `configs/presets/training/`
- salvataggio della configurazione finale in `configs/generated/`
- aggiornamento automatico di `latest.local.yaml` oppure `latest.cloud.yaml` in base all'ambiente selezionato

Comando:

```bash
python tools/cloud/cloud_configurator.py
```

### `tools/cloud/prepare_cloud_bundle.py`

Crea lo zip pronto per Google Colab.

Comando base:

```bash
python tools/cloud/prepare_cloud_bundle.py
```

Output atteso:

```text
dist/yolo-fire-detector-cloud.zip
```

Opzioni utili:

- `--include-dataset`: include anche un dataset gia' generato
- `--include-runs`: include run e checkpoint esistenti
- `--keep-notebook-outputs`: mantiene gli output dei notebook nel bundle

## Tool dataset

### `tools/dataset/dataset_viewer.py`

Serve a visualizzare rapidamente campioni del dataset YOLO generato, con immagini e label.

E' utile quando vuoi controllare se:

- i bounding box sono coerenti
- la distribuzione visiva del dataset e' plausibile
- gli esempi negativi e positivi sembrano corretti

## Tool model registry

### `tools/model_registry/drive_model_sync.py`

Gestisce export e import dei modelli tramite Google Drive.

Supporta due modalita':

- `filesystem`: usa una cartella Drive sincronizzata localmente
- `oauth`: usa direttamente le API Google Drive con credenziali OAuth

Comando generale:

```bash
python tools/model_registry/drive_model_sync.py <export|import> [opzioni]
```

### File credenziali OAuth

Template:

```text
tools/model_registry/oauth_credentials.example.json
```

File locale tipico:

```text
tools/model_registry/oauth_credentials.local.json
```

### Export di un modello verso Drive

Esempio in modalita' OAuth:

```bash
python tools/model_registry/drive_model_sync.py export --auth-mode oauth --registry-name yolo-fire-detector-models --drive-parent-id root --oauth-credentials-file tools/model_registry/oauth_credentials.local.json --local-persistent-root artifacts/local
```

Se non specifichi `--run-label`, lo script prova a usare il modello puntato dal `latest.yaml` trovato sotto il root locale.

### Import di un modello da Drive

Esempio in modalita' OAuth:

```bash
python tools/model_registry/drive_model_sync.py import --auth-mode oauth --registry-name yolo-fire-detector-models --drive-parent-id root --oauth-credentials-file tools/model_registry/oauth_credentials.local.json --target-persistent-root artifacts/imported
```

Per impostazione pratica, i modelli importati finiscono direttamente sotto il root scelto, ad esempio `artifacts/imported/`.

### Come scegliere il modello con `--run-label`

`--run-label` supporta questi casi:

- vuoto: usa il modello piu' recente disponibile
- nome run esatto: seleziona quella run
- nome file modello esatto: seleziona il file corrispondente
- prefisso: seleziona tutte le run che iniziano con quel prefisso
- `all`: importa tutte le run disponibili nel registry oppure esporta tutti i `.pt` trovati direttamente sotto il root locale scelto
- `all-r`: nel caso export locale, cerca ricorsivamente tutti i `.pt` sotto il root scelto

Se il selettore corrisponde a piu' modelli, lo script opera su tutti i match compatibili.

### Quando usare il model registry

Il model registry e' utile quando vuoi:

- pubblicare un modello addestrato per riusarlo altrove
- importare un modello su un altro computer senza copiare a mano i file
- mantenere un riferimento `latest.yaml` coerente anche fuori dalla macchina che ha addestrato il modello