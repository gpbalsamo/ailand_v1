#!/bin/bash
#SBATCH --job-name=ailand-extract
#SBATCH --partition=gpil
#SBATCH --qos=nf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#
# Extract from the anemoi O96 store. Must run on a compute node: the login
# session is capped at 8 GB and this needs ~5 GB resident.
#
#   YEARS="1998 2022" OUT=data/o96_1998_2022.zarr sbatch slurm/extract.sh
set -euo pipefail
REPO=/perm/pad/ailand
export PYTHONPATH="$REPO/src"
cd "$REPO"; mkdir -p slurm/logs
YEARS="${YEARS:-2020 2022}"
OUT="${OUT:-data/o96_full.zarr}"
NPOINTS="${NPOINTS:-11538}"
/usr/local/apps/python3/3.12.9-01/bin/python3.12 -u -m ailand.extract \
    --npoints "$NPOINTS" --years $YEARS --out "$OUT" --block 100
