# Training Presets

Questo documento aiuta a scegliere il preset di training piu' adatto senza dover leggere ogni file YAML uno per uno.

Tutti i preset pronti all'uso si trovano in `configs/presets/`.

## Come usare i preset

Un preset e' una base di configurazione. Puoi eseguirlo direttamente con:

```bash
python run_experiment.py --config configs/presets/<nome-preset>.yaml
```

Se vuoi solo generare o riusare il dataset senza avviare il training:

```bash
python run_experiment.py --config configs/presets/<nome-preset>.yaml --skip-training
```

Se preferisci un flusso guidato, usa il configuratore grafico. Il configuratore combina un preset con un runtime override e salva il risultato in `configs/generated/`.

## Preset consigliato per iniziare

Per la maggior parte dei casi, il punto di partenza piu' semplice e affidabile e':

```text
configs/presets/balanced-mini-fires.yaml
```

E' il preset consigliato per una prima prova seria perche':

- usa due immagini di fuoco gia' pronte
- mantiene il modello leggero
- bilancia costo, tempo di addestramento e qualita' finale

## Preset disponibili

### `configs/presets/smoke.yaml`

Serve a verificare rapidamente che la pipeline funzioni da inizio a fine. E' il preset giusto per il primo test del progetto.

### `configs/presets/baseline-lite.yaml`

E' una baseline locale leggera. E' utile quando vuoi una run semplice, con costo contenuto, adatta a CPU o a prove rapide.

### `configs/presets/default.yaml`

E' una baseline generale da usare come riferimento stabile quando vuoi confrontare varianti diverse.

### `configs/presets/balanced-mini-fires.yaml`

E' il preset consigliato per iniziare seriamente. Usa i file `base_fire_images/fire1-mini_nobg.png` e `base_fire_images/fire2-mini_nobg.png` e mantiene un buon compromesso tra leggerezza e qualita'.

### `configs/presets/quick-screen.yaml`

Serve a capire rapidamente se una direzione sperimentale merita attenzione. E' utile per iterazioni brevi.

### `configs/presets/fast-debug.yaml`

Riduce ulteriormente il costo per controlli tecnici sulla pipeline. E' pensato per debug e sviluppo, non per una valutazione finale della qualita'.

### `configs/presets/recall.yaml`

Da usare quando il modello perde troppi casi positivi. Punta a rendere il detector piu' sensibile.

### `configs/presets/hard-negatives.yaml`

Da usare quando il problema principale sono i falsi positivi. Introduce una pressione maggiore sui casi negativi difficili.

### `configs/presets/small-fire.yaml`

Da usare quando il fuoco da rilevare e' spesso piccolo o lontano nell'immagine.

### `configs/presets/robust.yaml`

Da usare quando il modello funziona bene solo in condizioni pulite ma peggiora con blur, rumore, occlusioni o cambi di colore.

### `configs/presets/capacity.yaml`

Da usare quando vuoi aumentare la capacita' del modello e hai piu' budget computazionale. E' il preset piu' costoso tra quelli principali.

### `configs/presets/piazzola-light.yaml`

E' una variante orientata allo scenario piazzola/light. Conviene usarla quando vuoi lavorare vicino a quel dominio specifico.

## Scelta rapida in base all'obiettivo

Se non sai quale preset scegliere, usa questa regola pratica:

1. `balanced-mini-fires.yaml` per partire
2. `recall.yaml` se perde troppo fuoco reale
3. `hard-negatives.yaml` se segnala troppo spesso fuoco dove non c'e'
4. `small-fire.yaml` se il fuoco appare piccolo o distante
5. `robust.yaml` se il problema e' la fragilita' fuori da condizioni pulite
6. `capacity.yaml` se vuoi piu' qualita' e puoi spendere piu' risorse

## Preset e flusso cloud

Nel flusso cloud non selezioni il preset direttamente dal notebook.

Il procedimento corretto e' questo:

1. scegli il preset in locale
2. combina il preset con un runtime cloud usando `tools/cloud/cloud_configurator.py`
3. genera `configs/generated/latest.cloud.yaml`
4. crea il bundle cloud
5. apri il notebook e avvia le celle

## Quando usare il configuratore grafico

Il configuratore e' consigliato se vuoi:

- cambiare il runtime senza modificare a mano i file YAML
- produrre una config finale pronta in `configs/generated/`
- salvare sia una configurazione locale sia una cloud con percorso coerente

Comando:

```bash
python tools/cloud/cloud_configurator.py
```