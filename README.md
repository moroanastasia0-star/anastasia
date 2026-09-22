# Water Ring Alpha Extractor

Estrae dal video originale **solo** il grande anello d'acqua animato (con
goccioline, schizzi, trasparenze e riflessi), rimuovendo completamente lo
sfondo e la superficie d'acqua inferiore, producendo un video con **canale
alpha** (trasparenza reale) utilizzabile in PowerPoint, After Effects,
Premiere, ecc.

**Nessuna intelligenza artificiale generativa è stata usata per creare
l'acqua**: il risultato è ottenuto con tecniche reali di video
segmentation / alpha matting (analisi colore, geometria, componenti
connesse, guided filter) applicate al video originale. Il movimento, la
forma e il timing dell'acqua sono esattamente quelli del video di partenza.

## Come si usa (nessuna conoscenza di programmazione richiesta)

### 1. Installazione (una sola volta)

Apri il Terminale nella cartella del progetto ed esegui:

```
./install.sh
```

Lo script:
- verifica che Python 3 sia installato;
- installa automaticamente FFmpeg se manca (ti verrà chiesta la password
  di amministratore solo per questo passaggio);
- crea un ambiente Python isolato (non modifica nulla del tuo sistema);
- installa tutte le librerie necessarie.

### 2. Avvio dell'applicazione

```
./run.sh
```

Si aprirà automaticamente una pagina nel browser con l'interfaccia grafica.

### 3. Uso dell'interfaccia

1. **Carica il video** originale.
2. Premi **"1. Analizza Video"** — mostra risoluzione, fps, durata.
3. Premi **"Genera anteprima su questo frame"** (sezione 2) — mostra
   l'anello rilevato automaticamente, la maschera e il risultato su sfondo
   nero/bianco/azzurro. Puoi scorrere i frame con lo slider.
4. Se il risultato automatico non ti soddisfa, apri **"Controlli manuali
   avanzati"** e regola i cursori (vedi sotto), poi rigenera l'anteprima.
5. Quando sei soddisfatta, premi **"Esporta tutti i file"**: verranno
   generati tutti i file finali (può richiedere alcuni minuti per un video
   di qualche secondo).
6. Scarica i file dalla lista o recuperali direttamente dalla cartella
   `output/`, `frames/`, `masks/` del progetto.

### Controlli manuali (da usare solo se necessario)

| Controllo | Cosa fa |
|---|---|
| Mask threshold | Quanto un pixel deve differire dallo sfondo per essere considerato acqua |
| Alpha softness | Morbidezza della transizione di trasparenza ai bordi |
| Edge feather | Sfumatura aggiuntiva applicata ai bordi della maschera |
| Lower surface cutoff | Margine (in pixel) sotto l'anello oltre il quale tutto viene reso trasparente (rimuove la superficie inferiore) |
| Preserve droplets | Quanto "raggio" di tolleranza usare per includere goccioline/schizzi vicini all'anello |
| Background removal strength | Quanto aggressivamente rimuovere lo sfondo/rumore residuo |
| Ring position Y | Sposta verticalmente la linea di taglio della superficie inferiore |
| Ring scale | Amplia/riduce la tolleranza con cui vengono incluse goccioline distanti |

## Cosa produce (cartelle del progetto)

- **`output/water_ring_transparent.webm`** — video con canale alpha (VP9),
  solo anello d'acqua, senza testo.
- **`output/water_ring_transparent.mov`** — stessa cosa in ProRes 4444
  (formato professionale con alpha), se il sistema lo supporta.
- **`output/water_ring_with_text.webm`** — versione con anello d'acqua
  **+ il testo originale** (dove è stato possibile separarlo in modo
  affidabile).
- **`frames/water_ring_frames/`** — tutti i singoli fotogrammi PNG RGBA
  trasparenti (solo acqua), stesso ordine/numero del video originale.
- **`frames/water_ring_with_text_frames/`** — stessa cosa con il testo.
- **`output/preview_black.mp4`**, **`preview_white.mp4`**, **`preview_blue.mp4`**
  — l'anello compositato su sfondo nero / bianco / azzurro ghiaccio, per
  verificare visivamente l'assenza di aloni.
- **`output/mask_preview.mp4`** — visualizzazione della maschera alpha nel
  tempo (bianco = opaco, nero = trasparente).
- **`masks/mask_frames_png/`** — gli stessi fotogrammi della maschera in PNG.
- **`output/qc_report.txt`** — report del controllo qualità automatico.

Nota sul WebM e la trasparenza: alcuni lettori video generici (incluso il
demux "veloce" di ffmpeg quando lo si apre senza specificare il codec)
possono mostrare il webm come se non avesse alpha. Il canale alpha è
comunque presente e viene letto correttamente da browser, PowerPoint (se
supporta VP9 alpha) e dalla maggior parte degli editor video professionali.
Se hai dubbi sulla trasparenza, usa preferibilmente il file **ProRes 4444
(.mov)** oppure la sequenza di **PNG**, che sono inequivocabili.

## Come funziona (in breve)

1. **Analisi del video**: risoluzione, fps, durata, frame.
2. **Stima dello sfondo**: il gradiente di sfondo (azzurro pallido -> blu)
   viene ricostruito come un "clean plate" analizzando quali pixel restano
   stabili nel tempo, e riempendo (inpainting) la zona occupata dall'anello.
3. **Segmentazione per differenza percettiva**: ogni fotogramma viene
   confrontato con questo sfondo pulito in spazio colore Lab, ottenendo una
   trasparenza morbida (non un taglio binario) che rispetta la reale densità
   ottica dell'acqua.
4. **Rilevamento geometrico dell'anello**: per ogni fotogramma vengono
   analizzate le componenti connesse per identificare l'anello (che cambia
   forma nel tempo, non è mai forzato a un cerchio perfetto) e la sua
   posizione/dimensione approssimativa.
5. **Rimozione della superficie inferiore**: la superficie d'acqua inferiore
   viene esclusa usando una linea di taglio geometrica che segue il profilo
   reale del bordo inferiore dell'anello (non un semplice rettangolo),
   anche nei punti in cui la superficie tocca visivamente l'anello.
6. **Coerenza temporale**: leggero smoothing tra fotogrammi vicini per
   ridurre lo sfarfallio, senza interpolare o inventare movimento.
7. **Raffinamento bordi**: un guided filter (edge-aware) allinea i bordi
   della maschera ai bordi reali dell'immagine.
8. **Decontaminazione colore**: ai bordi semitrasparenti viene rimosso il
   colore "contaminato" dallo sfondo (unmixing rispetto al plate), per
   evitare aloni blu/bianchi quando l'anello viene messo su sfondi diversi.
9. **Compositing ed esportazione**: RGBA -> PNG / WebM alpha / ProRes 4444
   + anteprime di verifica.
10. **Controllo qualità automatico**: su 5 fotogrammi campione (inizio,
    25%, centro, 75%, fine) viene verificato che l'anello sia presente e
    completo, che le trasparenze siano preservate, che la superficie
    inferiore sia stata eliminata, che lo sfondo sia trasparente e che non
    ci siano aloni o bordi artificiali.

## Limiti noti

- Il testo centrale viene separato in modo affidabile solo se rimane scuro
  e statico come nel video originale; se il testo cambia posizione/colore
  in altri video, la separazione automatica potrebbe non funzionare bene
  (in quel caso viene comunque garantito un OUTPUT 1 pulito senza testo).
- I controlli manuali di "posizione/scala dell'anello" agiscono sulla linea
  di taglio geometrico e sulla tolleranza di inclusione delle goccioline,
  non deformano l'acqua originale (che va sempre mantenuta fedele al video
  di partenza).
- Il file ProRes 4444 è di dimensioni elevate (qualità molto alta, poco
  compresso): è normale che sia molto più pesante del WebM.

## Uso da riga di comando (opzionale, per utenti avanzati)

```
source .venv/bin/activate
python3 src/main.py input/source_video.mp4 --out .
```

## Struttura del progetto

```
/src        codice della pipeline (modulare)
/ui         interfaccia grafica locale (Gradio)
/input      video sorgente
/output     video finali (webm, mov, preview, report)
/frames     sequenze di fotogrammi PNG RGBA
/masks      fotogrammi della maschera alpha
```
