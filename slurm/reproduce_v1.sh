#!/bin/bash
#SBATCH --job-name=ailand-repro
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#
# Reproduce the aiLand v1 O96 pretraining, as closely as this repo can.
#
# Matched to the paper (Raoult et al. 2026, Sect. 2.2.3):
#   1998-2019 training, 2022 held out for validation and checkpoint selection
#   6 x 512 backbone, smooth L1 over the rollout scaled by 1/R
#   two phases: R=4 (24 h) then R=8 (48 h)
#   Adam, peak 5e-4 -> 3e-7 cosine, 1000-step linear warmup, grad clip 5.0
#   batch 46,152 = 4 rollout windows x 11,538 O96 land points, v1's effective batch
#
# The reference numbers are Table B1, "Trained on O96 / evaluated on O96",
# single-timestep, glacier and coastal points excluded:
#   swvl1 0.010400  swvl2 0.003115  swvl3 0.000746
#   stl1  1.056     stl2  0.138     stl3  0.032    snowc 0.01462
#   2d    0.857     2t    0.876     skt   1.081    H 11.57  LE 9.68
#
#   PRESET=v1        sbatch slurm/reproduce_v1.sh   # exact v1 output set
#   PRESET=v1+fluxes sbatch slurm/reproduce_v1.sh   # + evaporation, runoff, GPP

set -euo pipefail
REPO=/perm/pad/ailand
PY=/usr/local/apps/python3/3.12.9-01/bin/python3.12
export PYTHONPATH="$REPO/src"
cd "$REPO"; mkdir -p slurm/logs

DATA="${DATA:-data/o96_1998_2022.zarr}"
PRESET="${PRESET:-v1}"
OUT="${OUT:-repro_${PRESET//+/_}}"
MAXSAMP="${MAXSAMP:-12000000}"
STATSAMP="${STATSAMP:-2000000}"
VALMAX="${VALMAX:-400000}"
STARTS="${STARTS:-}"
BATCH="${BATCH:-46152}"
EPOCHS1="${EPOCHS1:-80}"
EPOCHS2="${EPOCHS2:-8}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

$PY -u -m ailand.train_mlp \
    --profile o96 --preset "$PRESET" --data "$DATA" --temporal \
    --width 512 --depth 6 \
    --rollout 4 8 --epochs "$EPOCHS1" "$EPOCHS2" --lr 5e-4 3e-5 \
    --warmup 1000 --batch-size "$BATCH" --max-samples "$MAXSAMP" \
    --train-years 1998 2019 --val-years 2022 2022 \
    --stat-samples "$STATSAMP" --val-max "$VALMAX" \
    ${STARTS:+--start-times "$STARTS"} \
    --point-block 100 --device cuda --outdir "$OUT"

echo "=== Table B1 protocol: single-timestep, glacier+coastal excluded ==="
$PY -u -m ailand.evaluate \
    --profile o96 --preset "$PRESET" --data "$DATA" --temporal \
    --modeldir "$OUT" --npoints 800 --mask-glacier-coastal --single-step \
    --split 2022-01-01 --quiet --no-plot

echo "=== free autoregressive rollout over 2022 (Table 5 protocol) ==="
$PY -u -m ailand.evaluate \
    --profile o96 --preset "$PRESET" --data "$DATA" --temporal \
    --modeldir "$OUT" --npoints 800 --mask-glacier-coastal \
    --split 2022-01-01 --quiet --no-plot
