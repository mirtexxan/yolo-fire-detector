# Training Presets

Questi preset servono a provare training cloud diversi senza riscrivere ogni volta la config.

Tutti i file sono in `configs/` e vengono inclusi nello zip cloud automaticamente.

## Preset disponibili

- `configs/cloud.balanced-mini-fires.yaml`
  Preset consigliato come compromesso tra costo, tempo e qualita'. Usa i due asset trasparenti `fire1-mini_nobg.png` e `fire2-mini_nobg.png`, dataset medio e YOLOv8n per restare leggero anche in inferenza locale.

- `configs/cloud.default.yaml`
  Baseline attuale. Punto di confronto stabile.

- `configs/cloud.quick-screen.yaml`
  Run rapida per scremare idee. Utile quando vuoi verificare in fretta se una direzione vale la pena.

- `configs/cloud.recall.yaml`
  Focus su recall e confidenza piu' alta sui positivi. Usa meno negativi, piu' dati, modello `s` e fire scale piu' ampia.

- `configs/cloud.robust.yaml`
  Focus su robustezza a blur, rumore, occlusioni e shift colore. Utile se il modello e' fragile fuori dal dominio sintetico pulito.

- `configs/cloud.small-fire.yaml`
  Focus su incendi piccoli o lontani. Aumenta `image_size`, abbassa `fire_scale_min` e usa batch piu' piccolo.

- `configs/cloud.hard-negatives.yaml`
  Focus su hard negatives e riduzione falsi positivi. Utile se vuoi aumentare separazione e confidenza del classificatore.

- `configs/cloud.capacity.yaml`
  Focus su capacita' del modello. Usa YOLOv8m e piu' epoche. E' il preset piu' costoso ma anche quello con piu' margine qualitativo.

- `configs/cloud.fast-debug.yaml`
  Preset rapido per iterare su dataset e pipeline. Riduce costo di generazione e training per controlli veloci, non per misure finali di qualita'.

## Strategia consigliata

Se vuoi una sola scelta pragmatica da dare a uno studente, parti da:

1. `configs/cloud.balanced-mini-fires.yaml`
2. `configs/cloud.recall.yaml` se perde troppi positivi
3. `configs/cloud.hard-negatives.yaml` se compaiono troppi falsi positivi
4. `configs/cloud.capacity.yaml` solo se hai budget GPU e vuoi alzare la qualita'

Se il problema e' che il modello vede il fuoco ma con confidenza troppo bassa, l'ordine sensato e':

1. `configs/cloud.recall.yaml`
2. `configs/cloud.small-fire.yaml`
3. `configs/cloud.capacity.yaml`
4. `configs/cloud.robust.yaml`

`configs/cloud.hard-negatives.yaml` ha senso dopo, soprattutto se alzando la sensibilita' iniziano a comparire falsi positivi.

## Uso nel notebook cloud

Nel notebook `cloud_train.ipynb` puoi cambiare una sola variabile:

```python
READY_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'
```

Se `USE_READY_CONFIG_AS_IS = True`, il notebook copiera' quel preset in `configs/cloud.runtime.yaml` cambiando solo `project.persistent_root`.

Se invece `USE_READY_CONFIG_AS_IS = False`, puoi usare:

```python
BASE_CONFIG_NAME = 'cloud.balanced-mini-fires.yaml'
```

e poi sovrascrivere i parametri principali dalla cella di runtime.