#!/bin/bash
#SBATCH --job-name=ailand-mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.out
#
# Train the aiLand MLP on a GPU node.
#
# The Atos gpu / gpu_debug partitions carry 4x NVIDIA A100 (ga100) per node,
# which is the same hardware aiLand v1 trained on. This script uses one GPU;
# see the note at the bottom on going to all four.
#
#   sbatch slurm/train_gpu.sh
#   sbatch --partition=gpu_debug --time=00:30:00 slurm/train_gpu.sh   # quick test

set -euo pipefail
mkdir -p slurm/logs

REPO=/perm/pad/ailand
PY=/usr/local/apps/python3/3.12.9-01/bin/python3.12
export PYTHONPATH="$REPO/src"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Full O96 land set, all three years, the v1 state vector plus the extra fluxes.
$PY -m ailand.train_mlp \
    --profile o96 \
    --preset "v1+fluxes" \
    --data data/o96_full.zarr \
    --temporal \
    --width 512 --depth 6 \
    --rollout 4 8 \
    --epochs 80 8 \
    --batch-size 8192 \
    --train-years 2020 2021 \
    --device cuda \
    --outdir o96_gpu

$PY -m ailand.evaluate \
    --profile o96 --preset "v1+fluxes" --data data/o96_full.zarr \
    --temporal --modeldir o96_gpu --npoints 200 --quiet --no-plot

# To use all four A100s, v1 uses distributed data parallelism:
#   #SBATCH --gres=gpu:4 --ntasks-per-node=4
#   srun $PY -m torch.distributed.run --nproc_per_node=4 -m ailand.train_mlp ...
# train_mlp would need torch.nn.parallel.DistributedDataParallel wrapping and a
# DistributedSampler; single-GPU is enough for the point counts here.
