#!/bin/bash
#SBATCH --job-name=ailand-mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.out
#
# Train the aiLand MLP on a GPU node at the aiLand v1 network size (6 x 512).
#
# The Atos gpu / gpu_debug partitions carry 4x NVIDIA A100 (ga100) per node --
# the same hardware v1 trained on. This uses one of them; see the note at the
# end on going to all four.
#
#   sbatch slurm/train_gpu.sh                                  # defaults below
#   DATA=data/o96_full.zarr OUT=o96_full_gpu sbatch slurm/train_gpu.sh
#   sbatch --partition=gpu_debug --time=00:30:00 slurm/train_gpu.sh

set -euo pipefail

REPO=/perm/pad/ailand
PY=/usr/local/apps/python3/3.12.9-01/bin/python3.12
export PYTHONPATH="$REPO/src"
cd "$REPO"
mkdir -p slurm/logs

DATA="${DATA:-data/o96_2000.zarr}"
OUT="${OUT:-o96_gpu}"
PRESET="${PRESET:-v1+fluxes}"
WIDTH="${WIDTH:-512}"
DEPTH="${DEPTH:-6}"
EPOCHS1="${EPOCHS1:-60}"
EPOCHS2="${EPOCHS2:-8}"
BATCH="${BATCH:-8192}"
MAXSAMP="${MAXSAMP:-2000000}"
NPOINTS_EVAL="${NPOINTS_EVAL:-200}"

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "=== training: $DATA -> models/$OUT ==="
$PY -u -m ailand.train_mlp \
    --profile o96 --preset "$PRESET" --data "$DATA" --temporal \
    --width "$WIDTH" --depth "$DEPTH" \
    --rollout 4 8 --epochs "$EPOCHS1" "$EPOCHS2" \
    --batch-size "$BATCH" --max-samples "$MAXSAMP" \
    --train-years 2020 2021 \
    --device cuda --outdir "$OUT"

echo "=== evaluation: pooled over $NPOINTS_EVAL points, held-out 2022 ==="
$PY -u -m ailand.evaluate \
    --profile o96 --preset "$PRESET" --data "$DATA" --temporal \
    --modeldir "$OUT" --npoints "$NPOINTS_EVAL" --quiet --no-plot

# All four A100s, as v1 does (distributed data parallelism):
#   #SBATCH --gres=gpu:4 --ntasks-per-node=4
#   srun $PY -m torch.distributed.run --nproc_per_node=4 -m ailand.train_mlp ...
# train_mlp would need a DistributedDataParallel wrapper and a DistributedSampler.
# One A100 is ample for 2,000 points; revisit at the full 11,538 or at N320.
