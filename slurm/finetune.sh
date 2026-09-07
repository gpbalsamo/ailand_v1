#!/bin/bash
#SBATCH --job-name=ailand-ft
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#
# Fine-tune aiLand-base on FLUXNET-Shuttle observations.
#
#   S1  v1's frozen-backbone baseline (diagnostic head only)
#   S5  v1's most aggressive: full model at one rate
#   S6  new: soil observations on the prognostic head + anchor against the base
#
# Scored at held-out SITES, against the towers.
set -euo pipefail
REPO=/perm/pad/ailand
PY=/usr/local/apps/python3/3.12.9-01/bin/python3.12
export PYTHONPATH="$REPO/src"
cd "$REPO"; mkdir -p slurm/logs
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u -m ailand.finetune \
    --base "${BASE:-repro_v1_fluxes}" \
    --data "${DATA:-data/fluxnet_o96.zarr}" \
    --strategies ${STRATS:-S1 S5 S6} \
    --rollout 4 --max-samples "${MAXSAMP:-2000000}" --batch-size 4096 \
    --soil-match "${SOILMATCH:-mean}" \
    --anchor-data "${ANCHOR:-data/o96_1998_2022.zarr}" --anchor-samples 300000 \
    --device cuda --outdir "${OUT:-finetune}"
