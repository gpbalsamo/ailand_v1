#!/bin/bash
#SBATCH --job-name=ailand-anemoi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.out
#
# Phase 1 (R=4) pretraining of aiLand v1 on the anemoi-training stack --
# see docs/ANEMOI.md before running this for real; config_validation is
# disabled for a known upstream schema bug (documented there), and the
# land-point masking question is not yet resolved, so this will currently
# train over the full 40320-point O96 grid (ocean included), not the
# 11,538-point land set.
#
#   sbatch slurm/anemoi_train.sh                       # Phase 1 (R=4, 80 epochs)
#   sbatch --export=PHASE=2 slurm/anemoi_train.sh       # Phase 2 (R=8, 8 epochs), resumes Phase 1

set -euo pipefail

REPO=/perm/pad/ailand_v1
export PYTHONPATH="$REPO/src"
cd "$REPO"
mkdir -p slurm/logs

source .venv-anemoi/bin/activate

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

PHASE="${PHASE:-1}"
if [ "$PHASE" = "1" ]; then
    echo "=== Phase 1: R=4, 80 epochs ==="
    anemoi-training train --config-path configs/anemoi --config-name ailand_v1 \
        hydra.run.dir=models/anemoi/phase1
else
    echo "=== Phase 2: R=8, 8 epochs, resuming Phase 1 checkpoint ==="
    PHASE1_CKPT="${PHASE1_CKPT:-models/anemoi/phase1/checkpoint/last.ckpt}"
    anemoi-training train --config-path configs/anemoi --config-name ailand_v1 \
        +task=ailand_forecaster_r8 \
        training.max_epochs=8 \
        training.transfer_learning=True \
        "system.input.warm_start=$PHASE1_CKPT" \
        hydra.run.dir=models/anemoi/phase2
fi
