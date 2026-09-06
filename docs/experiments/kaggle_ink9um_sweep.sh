#!/bin/bash
# ink_9um sweep on Kaggle (2× T4) over a folder of locally rendered segments.
#
# On the PC:   zip -r renders_1447.zip work/1447_render   (one zip per scroll)
#              upload as a Kaggle dataset (Datasets → New → the zip; Kaggle
#              unpacks it under /kaggle/input/<dataset-name>/)
# In the notebook (GPU T4 x2, Internet ON), one cell:
#              !bash /kaggle/input/<dataset-name>/kaggle_ink9um_sweep.sh /kaggle/input/<dataset-name>/1447_render
# Output:      /kaggle/working/preds/<segment>_s<seed>_d<depth>[_reverse].tif  → download the folder
#
# 2 seeds × 3 depth windows × both directions = 12 predictions per segment;
# the two seeds run in parallel on the two T4s.
set -e
RENDERS=${1:?folder with <segment>_render.zarr}
SEEDS=${SEEDS:-"42 43"}
DEPTHS=${DEPTHS:-"0-16 7-23 14-30"}      # low / default / high, 17 slices each, out of 31
OUT=/kaggle/working/preds
mkdir -p $OUT

if [ ! -d /kaggle/working/villa ]; then
  cd /kaggle/working
  git clone --depth 1 https://github.com/ScrollPrize/villa.git
  cd villa/vesuvius
  pip install -q -e . --no-deps --ignore-requires-python   # no volume-cartographer (Ceres); villa asks for py3.14, runs on 3.12
  pip install -q pynrrd zarr tifffile imagecodecs huggingface_hub
  for S in $SEEDS; do
    python -c "from huggingface_hub import hf_hub_download as h; h('scrollprize/ink_9um', 'hybrid_3d2d-seed$S/step-075000.pth', local_dir='/kaggle/working/ckpt')"
  done
fi
cd /kaggle/working/villa/vesuvius

# one process per seed, each pinned to its own GPU (the inference script drives a single device)
run_seed () {
  S=$1; GPU=$2
  for Z in "$RENDERS"/*_render.zarr; do
    SEG=$(basename "$Z" _render.zarr)
    for D in $DEPTHS; do
      L0=${D%-*}; L1=${D#*-}
      T="$OUT/${SEG}_s${S}_d${L0}.tif"
      [ -f "$T" ] && continue
      echo "=== $SEG seed $S depth $D gpu $GPU $(date +%H:%M:%S)"
      CUDA_VISIBLE_DEVICES=$GPU python -m vesuvius.ink_detection.inference.infer "$Z" \
        "/kaggle/working/ckpt/hybrid_3d2d-seed$S/step-075000.pth" "$T" \
        --overlap 0.5 --blend-mode hann --batch-size 4 \
        --layer-start $L0 --layer-end $L1 --direction both 2>&1 | grep -E "INFO Selected|Error|error"
    done
  done
}
GPU=0
for S in $SEEDS; do run_seed $S $GPU & GPU=$((GPU + 1)); done
wait
echo "=== done: $(ls $OUT | wc -l) files in $OUT"
