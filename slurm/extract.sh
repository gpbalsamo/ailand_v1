#!/bin/bash
#SBATCH --job-name=ailand-extract
#SBATCH --partition=gpil
#SBATCH --qos=nf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#
# Extract the full O96 land set (all 11,538 points, 2020-2022) from the anemoi
# store. Reading is per-timestep, so point count is nearly free -- the cost is
# ~63 ms per 6-hourly step regardless of how many points are kept.
set -euo pipefail
mkdir -p slurm/logs
REPO=/perm/pad/ailand
export PYTHONPATH="$REPO/src"
cd "$REPO"
/usr/local/apps/python3/3.12.9-01/bin/python3.12 -m ailand.extract \
    --npoints 11538 --years 2020 2022 --out data/o96_full.zarr
