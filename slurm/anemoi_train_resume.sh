#!/bin/bash
#SBATCH --job-name=ailand-anemoi-resume
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.out
#
# Resume Phase 1 (R=4) pretraining from the last checkpoint of a run that hit
# the sbatch --time wall clock before finishing its 80 epochs.
#
# anemoi-training's own resume mechanism (train/train.py: run_id / last_checkpoint)
# reuses the SAME run's checkpoint directory when `training.run_id` is set to that
# run's id (as opposed to `training.fork_run_id`, which starts a NEW run warm-started
# from another run's weights only -- that's what Phase 2 uses). Setting `run_id` here
# restores full trainer state (optimizer, LR scheduler step count, current epoch,
# global_step) from checkpoint/<run_id>/last.ckpt, so training continues the cosine
# LR schedule and epoch count exactly where it left off instead of restarting them.
#
#   sbatch --export=RUN_ID=<run-id> slurm/anemoi_train_resume.sh

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

RUN_ID="${RUN_ID:?Set RUN_ID to the checkpoint dir under models/anemoi/checkpoint/ to resume, e.g. sbatch --export=RUN_ID=e87471d5-9e0f-4236-96fb-288f89cb557c slurm/anemoi_train_resume.sh}"
CKPT="models/anemoi/checkpoint/$RUN_ID/last.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "=== Phase 1 resume: R=4, 80 epochs total, run_id=$RUN_ID ==="
srun anemoi-training train --config-dir "$REPO/configs/anemoi" --config-name ailand_v1 \
    system.hardware.num_gpus_per_node=4 \
    system.hardware.num_nodes=1 \
    "training.run_id=$RUN_ID" \
    hydra.run.dir=models/anemoi/phase1_resume
