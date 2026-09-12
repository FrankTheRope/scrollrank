# La metà GPU: rendering + inferenza `ink_9um` senza possedere una GPU

Tutto qui sotto viene dal tutorial ufficiale (https://scrollprize.org/tutorial5,
sezione "Ink detection at 9 µm") e va eseguito su una macchina Linux con GPU
NVIDIA. Opzioni senza hardware proprio:

* **Gratis**: Kaggle Notebooks o Google Colab (quote GPU settimanali; verifica
  le condizioni attuali, cambiano).
* **A noleggio**: RunPod, Vast.ai, Lambda — una GPU consumer da 24 GB costa
  indicativamente pochi decimi di dollaro l'ora. Scegli un'istanza con
  ≥ 100 GB di disco.

Regola d'oro: **la GPU serve solo per il passo 3.** Scarica gli output (TIFF di
predizione, pochi MB–GB) sul tuo PC e fai tutto il resto con ScrollRank.

## 0. Setup (una volta per macchina)

```bash
git clone https://github.com/ScrollPrize/villa.git
cd villa/vesuvius
uv sync --extra models
uv run --extra models python -c "import torch; print(torch.cuda.is_available())"   # deve stampare True

# checkpoint pubblici cross-scroll (2 seed, 7 checkpoint ciascuno)
uvx --from huggingface_hub hf download scrollprize/ink_9um \
  hybrid_3d2d-seed42/step-075000.pth --local-dir checkpoints/ink_9um
uvx --from huggingface_hub hf download scrollprize/ink_9um \
  hybrid_3d2d-seed43/step-075000.pth --local-dir checkpoints/ink_9um
```

Per il rendering serve anche un build di VC3D (`vc_render_tifxyz`); segui
https://scrollprize.org/tutorial_VC3D#installing-vc3d. Se il build C++ ti blocca,
chiedi su Discord: è il primo ostacolo di tutti.

## 1. Scegli segmento e rotolo

```bash
scrollrank catalog --ls          # elenca i segmenti pubblici dei 13 rotoli eleggibili
aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/PHerc0800/segments/<SEG>/
```

Se il segmento ha già una cartella `surface-volumes/`, scaricala e salta il passo 2.

## 2. Rendering (streaming da S3, scarica solo i chunk necessari)

```bash
aws s3 sync --no-sign-request \
  s3://vesuvius-challenge-open-data/<SCROLL>/segments/<SEG>/mesh/<MESH>.tifxyz/ work/<SEG>.tifxyz

vc_render_tifxyz \
  --volume volume-cache/<VOLUME_ID>.zarr \
  --remote-url s3://vesuvius-challenge-open-data/<SCROLL>/volumes/<VOLUME_ID>-8.640um-1.2m-116keV-masked.zarr/ \
  --segmentation work/<SEG>.tifxyz \
  --zarr-output work/<SEG>_9um.zarr \
  --scale 1 --group-idx 0 --num-slices 28 --cache-gb 16 \
  --voxel-size 8.64 --voxel-unit micrometer
```

(Per i rotoli eleggibili la voxel size è 8.64 µm; il tutorial usa 9.362 µm per
PHerc0139. I modelli sono stati addestrati a ~9 µm, quindi 8.64 è nel range.)

## 3. Inferenza: lo "sweep" che alimenta `scrollrank aggregate`

```bash
for SEED in 42 43; do
  for STEP in 030000 050000 075000; do
    for L0 in 2 4 6; do             # finestra di profondità: prova offset diversi
      uv run --extra models python -m vesuvius.ink_detection.inference.infer \
        work/<SEG>_9um.zarr \
        checkpoints/ink_9um/hybrid_3d2d-seed$SEED/step-$STEP.pth \
        preds/<SEG>_s${SEED}_st${STEP}_l${L0}.tif \
        --overlap 0.5 --blend-mode hann --batch-size 4 \
        --layer-start $L0 --layer-end $((L0+17)) --direction both
    done
  done
done
```

Ogni run scrive anche `*_reverse.tif` (superficie vista dall'altro lato).
Tempo: dell'ordine di un'ora per run per segmento su una GPU media — riduci con
`--mask-path` a una regione, o con meno combinazioni.

## 4. Torna su CPU

```bash
scrollrank aggregate preds/<SEG>_*.tif --out out/<SEG>_ens
scrollrank score out/<SEG>_ens/mean.tif --pixel-size-um 8.64 --out out/<SEG>_score
```

Guarda `overlay.png` e `consistency.tif`. Le finestre in cima alla classifica
sono quelle su cui vale la pena spendere altre ore GPU (render più profondo,
altri checkpoint) e, se vedi tratti ripetibili, iniziare lo pseudo-labeling
seguendo il tutorial — tenendo la regione di submission separata dal training.
