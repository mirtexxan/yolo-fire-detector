# Training Presets

Il configuratore usa preset opzionali separati in due famiglie:

- `configs/presets/dataset/`: controllano generazione dataset e trasformazioni
- `configs/presets/training/`: controllano training e iperparametri principali

I preset non sono obbligatori: puoi compilare tutto manualmente nella tab `Generale` e poi rifinire in `Avanzate`.

## Come usare i preset nel configuratore

1. avvia `python tools/cloud/cloud_configurator.py`
2. compila i campi base in `Generale`
3. in `Presets` applica dataset/training preset se ti servono
4. salva in `configs/generated/latest.local.yaml` o `configs/generated/latest.cloud.yaml`

## Preset dataset disponibili

### `configs/presets/dataset/test-rapido.yaml`

Dataset piccolo e veloce per test tecnici del pipeline.

### `configs/presets/dataset/standard.yaml`

Scelta bilanciata per iniziare in modo affidabile.

### `configs/presets/dataset/ampio-vario.yaml`

Maggiore varieta' di trasformazioni per migliorare robustezza generale.

### `configs/presets/dataset/fuochi-piccoli.yaml`

Tarato su fuochi piccoli/lontani e condizioni visive difficili.

### `configs/presets/dataset/anti-falsi-positivi.yaml`

Aumenta il peso dei negativi per ridurre falsi allarmi.

## Preset training disponibili

### `configs/presets/training/test-rapido.yaml`

Poche epoche per verifiche rapide.

### `configs/presets/training/standard.yaml`

Bilanciato per uso generale.

### `configs/presets/training/alta-recall.yaml`

Favorisce la sensibilita' del detector (meno falsi negativi).

### `configs/presets/training/addestramento-lungo.yaml`

Piu' epoche per convergenza piu' stabile (costo maggiore).

### `configs/presets/training/modello-grande.yaml`

Usa un modello piu' capiente (`model_size: m`) con costo computazionale superiore.

## Combinazioni consigliate rapide

1. Avvio consigliato: `dataset/standard` + `training/standard`
2. Test tecnico veloce: `dataset/test-rapido` + `training/test-rapido`
3. Fuochi piccoli: `dataset/fuochi-piccoli` + `training/alta-recall`
4. Ridurre falsi positivi: `dataset/anti-falsi-positivi` + `training/modello-grande`

## Preset e flusso cloud

Nel flusso cloud i preset si scelgono sempre in locale nel configuratore, prima di creare il bundle.

Procedura:

1. `Ambiente = Cloud` nel configuratore
2. preset opzionali (dataset/training)
3. salva `configs/generated/latest.cloud.yaml`
4. esegui `python tools/cloud/prepare_cloud_bundle.py`
5. avvia `cloud_train.ipynb`