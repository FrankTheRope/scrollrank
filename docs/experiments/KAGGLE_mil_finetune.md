# Running `mil_finetune_ink9um.py` on a Kaggle T4

Verified cell sequence for `docs/experiments/mil_finetune_ink9um.py` (the geometry-guided fine-tuning of the
public `ink_9um` checkpoint on the PHerc1667 w028 crop).  Copy the cells **verbatim, in order**.  Everything
below was executed for real, either on this box or against the live services, and the numbers quoted
(download sizes, timings, checkpoint bytes) are measured, not estimated.

## 0. Preconditions (do this before running any cell)

* **Settings -> Accelerator -> GPU T4 x1.**  The script also runs on CPU, but a 400-step arm is ~18 min on
  20 CPU threads versus ~1-3 min on a T4.
* **Settings -> Internet -> On.**  This requires a phone-verified Kaggle account.  Cells 1, 2, 3 and 5 are all
  network-bound (github.com, PyPI, huggingface.co) and a notebook with Internet **off** dies at the first
  `git fetch` with a DNS/connect error that says nothing about the real cause.  If you cannot enable
  Internet, use the [offline route](#offline-route-no-internet) below.  It reaches the same result, but it
  is **not** a drop-in substitute: it needs a one-time preparation step (villa source, checkpoint **and a
  wheel dataset**) carried out on a machine that *does* have Internet.  There is no way to install the
  seven missing Python packages from inside an Internet-off notebook without such a dataset.
* **A Kaggle dataset** (call it `<ds>`) that carries the three files the run needs:
  `mil_1667_w028.npz` (128 MB), `mil_finetune_ink9um.py`, `ink9um_api.py`.
  It is mounted at `/kaggle/input/<ds>`.

## 1. Internet precondition check + villa source at a pinned ref

`vesuvius` is **not** pip-installable from the repo root: the villa monorepo has no `setup.py` /
`pyproject.toml` at its root (the only project file is `villa/vesuvius/pyproject.toml`), and that project
pins `requires-python = ">=3.14,<3.15"` while Kaggle runs 3.11.  The source itself is 3.11-clean, and the
local "editable install" this experiment was developed against is nothing but a path entry pointing at
`villa/vesuvius/src`.  So: **fetch the source and put it on `PYTHONPATH`; do not pip-install villa.**

A full clone is 2.2 GB of working tree plus 771 MB of history for a 22 MB package.  The pinned, blobless,
sparse fetch below moved **31 MB in 4.2 s** (measured) and imports identically.

The cell is **idempotent and self-healing**: `git fetch` over the Kaggle network is the fragile step, and a
run that dies there leaves a `/kaggle/working/villa` that is a git repo but has no `vesuvius/src`.  Re-running
the cell then re-enters the `if`, so it must start from a clean directory - without the `rm -rf` the second
attempt aborts on `error: remote origin already exists.` (measured: exit 3) and never retries the fetch, and
the only message the user sees is about a git remote.  Verified: against exactly such a broken directory the
cell below succeeds in 4.2 s, and a second run is a 7 ms no-op.

**Cell 1 (Python) - the precondition check:**

```python
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=5).close()
    print("internet: on")
except OSError as e:
    raise SystemExit(f"Kaggle Internet is OFF ({e}). Notebook Settings -> Internet -> On (needs phone "
                     f"verification), or use the offline route: attach villa/vesuvius/src, "
                     f"step-075000.pth AND a wheel dataset as Kaggle datasets, then skip cells 1, 2 and 5 "
                     f"and replace cell 3 with its --no-index form.")
```

**Cell 2 (bash) - the villa source:**

```bash
%%bash
set -euo pipefail
VILLA=/kaggle/working/villa
SHA=23adee047dea06526151d3a152a7d85de8da478b     # the ref every local run in this experiment used
if [ ! -d "$VILLA/vesuvius/src" ]; then
  rm -rf "$VILLA"        # a half-finished earlier attempt is NOT resumable: git remote add would abort
  mkdir -p "$VILLA" && cd "$VILLA"
  git init -q .
  git remote add origin https://github.com/ScrollPrize/villa.git
  git config core.sparseCheckout true
  git sparse-checkout set --cone vesuvius
  git fetch -q --depth 1 --filter=blob:none origin "$SHA"
  git checkout -q FETCH_HEAD
fi
du -sh "$VILLA"
ls "$VILLA/vesuvius/src"
```

## 2. Dependencies (cell 3)

Only the modules Kaggle does not already ship.  `timm` and `einops` are **hard** requirements of the
model-building path (`vesuvius/models/build/primus_wrapper.py` imports `timm.layers.RotaryEmbeddingCat` and
`einops`), and installing `timm` without `--no-deps` also pulls `torchvision`, `huggingface_hub` and
`safetensors`, which the same import chain needs.  Leaving them out makes `load_ink9um` raise
`ModuleNotFoundError` before a single step of any arm runs.

```python
!pip install -q pynrrd donfig zarr numcodecs nest_asyncio timm einops
```

Here pip also resolves `zarr`'s own hard dependency **`google-crc32c`** for you (`zarr/codecs/__init__.py`
does `from zarr.codecs.crc32c_ import Crc32cCodec`, and that module's line 7 is an unconditional
`import google_crc32c`).  The offline route below installs with `--no-deps`, so there it has to be listed
explicitly - which is why the guard cell imports it by name.

## 3. Dependency guard (cell 4) - fails in seconds instead of 20 minutes in

Run this after cell 3 on **either** route; on the offline route it is what proves the wheel install landed.
`torch` must still be the image's own build and `numpy` its own version - nothing in cell 3 may replace
them.

```python
import importlib
for m in ("nrrd", "donfig", "zarr", "numcodecs", "google_crc32c", "fsspec", "tifffile", "yaml", "requests",
          "aiohttp", "nest_asyncio", "timm", "torchvision", "einops", "huggingface_hub", "sklearn", "PIL"):
    importlib.import_module(m)
import numpy, torch
print("deps ok | numpy", numpy.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

## 4. The checkpoint (cell 5) - one file, not the whole model repo

`scrollprize/ink_9um` holds seed42 and seed43 x 7 steps, 1.94 GB in total; the run needs exactly one
138,360,231-byte file.  Leave it in the default HF cache (`/root/.cache`) so it is not written into the
committed `/kaggle/working` output.

```python
from huggingface_hub import hf_hub_download
CKPT = hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed43/step-075000.pth")
import os; print(CKPT, os.path.getsize(CKPT))     # expect 138360231
```

## 5. Smoke run (cell 6, ~25 s on CPU, faster on the T4)

Replace `<ds>` with your dataset directory.  The smoke is **not** protocol-clean (400x600 crop, 6 steps, no
held-out monitor, so the data-driven guards are inactive); it only proves the pipeline runs end to end.

```bash
%%bash
set -euo pipefail
DS=/kaggle/input/<ds>
mkdir -p /kaggle/working/exp && cd /kaggle/working/exp
cp "$DS/mil_finetune_ink9um.py" "$DS/ink9um_api.py" .
export PYTHONPATH=/kaggle/working/villa/vesuvius/src
CKPT=$(ls /root/.cache/huggingface/hub/models--scrollprize--ink_9um/snapshots/*/hybrid_3d2d-seed43/step-075000.pth)
python mil_finetune_ink9um.py --data "$DS/mil_1667_w028.npz" --ckpt "$CKPT" \
  --out /kaggle/working/ft_smoke --arm all --device cuda --smoke
```

## 6. The real run (cell 7, all four arms, ~15-35 min on a T4)

```bash
%%bash
set -euo pipefail
DS=/kaggle/input/<ds>
cd /kaggle/working/exp
export PYTHONPATH=/kaggle/working/villa/vesuvius/src
CKPT=$(ls /root/.cache/huggingface/hub/models--scrollprize--ink_9um/snapshots/*/hybrid_3d2d-seed43/step-075000.pth)
python mil_finetune_ink9um.py --data "$DS/mil_1667_w028.npz" --ckpt "$CKPT" \
  --out /kaggle/working/ft_t4 --arm all --train decoder --steps 400 --batch-size 16 \
  --anchor both --curve-every 25 --monitor-tiles 6 --device cuda
```

Peak GPU memory at `--batch-size 16 --train decoder` is ~2-3 GB (measured: 4.3 M trainable parameters,
262 MiB of retained fp32 activations, plus the 138 MB student and the 138 MB frozen teacher), so a 16 GB T4
is not a constraint even at the default `--infer-batch 32`.

**`--save-ckpt` is deliberately omitted.**  Each saved arm is a 138 MB file in `/kaggle/working`, and the
supervised arm is the illegitimate upper bound whose weights nobody wants.  If you do want the fine-tuned
weights, run that arm alone with the flag, e.g.
`... --arm mil_free --train decoder --steps 400 --batch-size 16 --device cuda --save-ckpt`;
`results.json` merges arms from separate invocations by fingerprint, so the table still comes out complete.
An arm the guard rolled back to step 0 writes **no** checkpoint (those are the input weights, not a
fine-tuned model) and says so.

## 7. What to read in the output (cell 8)

```python
import json
r = json.load(open("/kaggle/working/ft_t4/results.json"))
b = r["arms"]["baseline"]
print("stage 0:", b["stage0_gap"], b["baseline_reproduces"])   # must be True, gap < 0.005
print(r["table_markdown"])
for a, v in r["arms"].items():
    print(a, v["arm_status"], "| weights from step", v.get("weights_from_step"),
          "| guard:", v.get("guard"))
```

* `baseline_reproduces` must be **True** on the T4 as well.  Do **not** reuse the CPU number: the per-arm
  fingerprint includes `device` and `infer_batch`, so a T4 `results.json` never merges with a local CPU one -
  the T4 baseline arm is its own stage-0 control.  The fp16 autocast shift is negligible (measured
  corr 0.999993, max |dp| 4.8e-3 between fp32 and fp16 patch probabilities), so the 0.005 gate passes; note
  that a failure only **prints a warning**, it does not abort the run.
* `arm_status` / the `steps / guard` column is the honest description of each row.  An arm that reads
  `scoring step 0 = PRISTINE` was rolled back by a guard and is the *input* model - its `+0.0000` delta means
  "training was discarded", not "training did nothing".
* `curve[*]["monitor_sigma"]` records the bootstrap spread of the guard statistics on that curve point; a
  threshold crossing smaller than `--guard-sigma` of it is logged as a note and does not stop the arm.

## Offline route (no Internet)

An Internet-off notebook cannot fetch the villa source, the checkpoint **or the Python packages the stock
image is missing**.  All three have to be prepared once, on a machine that does have Internet, and uploaded
as Kaggle datasets.  After that the notebook itself needs no network at all:

* **cells 1, 2 and 5 are dropped** (they only fetch things that now arrive as datasets);
* **cell 3 is replaced** by the `--no-index` form below - it is *not* skippable, because every one of the
  seven missing packages is a hard, unguarded top-level import somewhere in the villa chain
  (`nrrd` and `zarr` in `vesuvius/data/volume.py`, `donfig` under `zarr`, `numcodecs` under
  `zarr.codecs.blosc`, `nest_asyncio` in `vesuvius/utils/catalog.py`, `timm` and `einops` in
  `vesuvius/models/build/primus_wrapper.py`), so an Internet-off notebook that skips it dies at
  `import nrrd`;
* **cells 4 and 8 are run unchanged**, and the run cells 6 / 7 differ only in `PYTHONPATH` and `--ckpt`.

### One-time preparation (on a machine with Internet)

1. **villa source** - run cell 2 there (or any equivalent checkout of ref `23adee04`) and upload the
   resulting `villa/vesuvius/src` (22 MB) as dataset `<villa-ds>`;
2. **checkpoint** - `hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed43/step-075000.pth")`,
   138,360,231 bytes, as dataset `<ckpt-ds>` (it can live in the same dataset as the npz and the two .py
   files);
3. **wheels** - 8 files, 12 MB in total.  The command below asks for **Kaggle's** interpreter tags
   (CPython 3.11, manylinux x86-64), *not* the tags of the machine you run it on, so it works from any
   Python.  Upload the resulting `wheels/` directory as dataset `<wheels-ds>`.

```bash
python -m pip download --no-deps -d wheels --only-binary=:all: \
  --python-version 3.11 --implementation cp --abi cp311 --platform manylinux2014_x86_64 \
  pynrrd donfig zarr numcodecs google-crc32c nest_asyncio timm einops
```

Measured result (all 8 resolve as `cp311`/`manylinux` or pure-Python wheels): `pynrrd-1.1.3`,
`donfig-0.8.1.post1`, `zarr-3.1.6`, `numcodecs-0.16.5-cp311-manylinux2014_x86_64` (8.8 MB),
`google_crc32c-1.8.0-cp311-manylinux2014_x86_64`, `nest_asyncio-1.6.0`, `timm-1.0.29` (2.6 MB),
`einops-0.8.2`.  `google-crc32c` is on the list because `--no-deps` will not pull it and `import zarr`
fails without it; the other transitive names (`numpy`, `typing_extensions`, `packaging`, `pyyaml`,
`torch`, `torchvision`, `huggingface_hub`, `safetensors`) are all already on the Kaggle image.

### Cell 3, offline replacement

`--no-deps` is deliberate: it keeps pip from touching the image's `numpy` or `torch`.  pip leaves any of
these that the image already has untouched (no `--upgrade`), so the cell is safe to re-run.

```python
!pip install -q --no-index --no-deps --find-links=/kaggle/input/<wheels-ds> pynrrd donfig zarr numcodecs google-crc32c nest_asyncio timm einops
```

Then run **cell 4 unchanged** - it is the proof that the offline install landed, and the only thing that
distinguishes a working offline notebook from one that will die 20 minutes later.  Verified end to end here:
this exact `--no-index --no-deps --find-links=...` install of these 8 names, placed ahead of a stock
environment on `PYTHONPATH`, passed the cell-4 guard and then built the model through `load_ink9um`
(34,546,498 parameters).  Verified negatively too: with `google_crc32c` genuinely absent, `import zarr`
raises `ModuleNotFoundError` at `zarr/codecs/crc32c_.py` line 7.

### The run cells, offline form

Only `PYTHONPATH` and `--ckpt` change.  This is cell 7 (the real run); cell 6 (the smoke) is the same
with `--out /kaggle/working/ft_smoke --arm all --device cuda --smoke` in place of the last two lines.

```bash
%%bash
set -euo pipefail
DS=/kaggle/input/<ds>
mkdir -p /kaggle/working/exp && cd /kaggle/working/exp
cp "$DS/mil_finetune_ink9um.py" "$DS/ink9um_api.py" .
export PYTHONPATH=/kaggle/input/<villa-ds>/src        # the directory that contains vesuvius/
python mil_finetune_ink9um.py --data "$DS/mil_1667_w028.npz" \
  --ckpt /kaggle/input/<ckpt-ds>/step-075000.pth \
  --out /kaggle/working/ft_t4 --arm all --train decoder --steps 400 --batch-size 16 \
  --anchor both --curve-every 25 --monitor-tiles 6 --device cuda
```

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ERROR: ... does not appear to be a Python project` | `pip install -e villa` on the monorepo root | do not pip-install villa; use `PYTHONPATH=.../vesuvius/src` (cell 2) |
| `ModuleNotFoundError: timm` / `einops` / `nrrd` / `zarr` | cell 3 skipped, or its wheel dataset incomplete | rerun cell 3 exactly as written for your route (online or `--no-index`), then cell 4 |
| `error: remote origin already exists.` | cell 2 was interrupted mid-`git fetch`, leaving a half-built `/kaggle/working/villa` | the cell as written above is self-healing (`rm -rf "$VILLA"`); on an older copy of the notebook run `!rm -rf /kaggle/working/villa` and re-run cell 2 |
| `Could not resolve host: github.com` | Internet off | Settings -> Internet -> On, or the offline route (which needs the wheel/villa/ckpt datasets prepared first) |
| `ModuleNotFoundError: google_crc32c` (raised inside `import zarr`) | offline cell 3 run without `google-crc32c` in the wheel dataset | re-do the `pip download` with all 8 names and re-upload `<wheels-ds>` |
| `cannot import ink9um_api (...)` | `ink9um_api.py` not next to the script, or villa not on `PYTHONPATH` | the `cp` and `export PYTHONPATH` lines of cell 6 |
| baseline AUC far from 0.7714 / 0.7623 | wrong checkpoint, wrong npz, or a changed input pipeline | check `results.json["arms"]["baseline"]["stage0_gap"]` and the checkpoint byte size |
