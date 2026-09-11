# `v1_anemoi`: aiLand on ECMWF's anemoi-training stack

Branch off `master`, reimplementing the aiLand v1 pipeline
(`src/ailand/mlp.py`, `train_mlp.py`, `finetune.py`) on `anemoi-training`
instead of the hand-rolled PyTorch loop, so the two can be compared, and so
aiLand converges onto ECMWF's own production ML tooling. Started because
anemoi-models shipped a genuine point-wise (no graph coupling) MLP
architecture — see "Why this exists" below.

Status: **Phase 0 done** (environment verified, real training data reachable
directly). **Phase 1 config written and schema-validated**, except for one
upstream bug (documented below) and one open design question (land masking).
No training job has been run yet — that's the natural next step, and a real
GPU-hours commitment, so it hasn't been launched without a separate go-ahead.

---

## Why this exists

`anemoi-models` 0.19.0 added `PointWiseForwardMapper` /
`PointWiseMLPProcessor` / `PointWiseBackwardMapper`
(`anemoi.models.layers.{mapper,processor}`) — a per-gridpoint MLP
encoder/processor/decoder with no graph message-passing, wired into the
standard `anemoi.models.models.AnemoiModelEncProcDec` via a shipped example
config, `anemoi/training/config/model/point_wise.yaml`. That config already
uses:

* `residual.datasets.data._target_: SkipConnection` — the same residual /
  increment formulation as `ailand.mlp.rollout_batch`
  (`state = x[:, prog_idx] + prog_n * tend`).
* `bounding.datasets.data: [...]` — the same idea as `ailand.config.BOUNDS`,
  applied as post-processing, with named classes (`ReluBounding`,
  `HardtanhBounding`, `FractionBounding`, ...) instead of a raw `(lo, hi)`
  table.

This is close enough to v1's own described architecture (Raoult et al. 2026,
Sect. 2.2) that it reads as more than coincidence — v1's own data pipeline
already uses `anemoi-datasets` (see `data/EXTERNAL.md`), so this is plausibly
close to what the paper's authors themselves trained on.

## Environment

```bash
python3 -m venv --system-site-packages .venv-anemoi
source .venv-anemoi/bin/activate
pip install -r requirements-anemoi.txt   # anemoi-training/models/graphs/datasets, pinned
pip install -e .
```

Kept as a **separate venv** from `requirements.txt` deliberately —
anemoi-training pulls its own torch/pytorch-lightning/torch-geometric/CUDA
stack, not guaranteed to coexist with the v0/v1 hand-rolled pipeline's pins.

**Two real infrastructure gotchas hit while installing, both fixed, worth
knowing if they recur:**

1. A poisoned local pip HTTP cache made `nvidia_cusparse_cu12` (and later
   other large CUDA wheels) stall at a fixed byte offset on every retry.
   Fix: `pip install --no-cache-dir`, or `pip cache purge` first. `curl`
   fetching the exact same URL was never affected, which is what gave this
   away — if a pip download only ever stalls at the *same* byte count across
   retries, suspect the cache, not the network.
2. `$HOME` was already at 95.5/100GB quota (mostly a pre-existing 7.2GB
   `~/.local` ML stack from unrelated work), and the SSD-local `$TMPDIR`
   scratch has a hard **3GB** quota (`quota` command, "Quota for TMPDIR on
   local SSD") — nowhere near enough for the ~4GB of CUDA wheels. `$PERM` (10TB
   quota, ~4.4TB used) has ample room; for anything bigger or more
   throughput-sensitive (e.g. repeatedly reading the O96/N320 training data),
   prefer `$SCRATCH` (`/ec/res4/scratch/pad`, Lustre, 50TB quota, 30TB free) —
   both faster and the conventional place for this kind of transient/large
   data on this HPC. Point `TMPDIR=`/`PIP_CACHE_DIR=` there for any future
   reinstall.

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # cuda False is expected on a login node
python3 -c "import anemoi.training, anemoi.models, anemoi.graphs, anemoi.datasets"
anemoi-training --help
```

## The real training data, directly

No extraction step needed — `anemoi.datasets.open_dataset` reads the actual
`/lus` v1 training stores this repo already knew the paths to
(`src/ailand/extract.py`'s `O96`/`N320` constants) directly, confirmed live:

```python
from anemoi.datasets import open_dataset
ds = open_dataset("/lus/h1aiws01/project/ai-ml/datasets/aifs-rd-an-oper-isc8-mars-o96-1998-2024-6h-v1-ecland-era5met.zarr")
# shape (39444, 76, 1, 40320), frequency 6h, 76 named variables --
# matches ailand.config's O96_STATIC/O96_MET/O96_TEMPORAL/O96_GEO/O96_PRESETS
# exactly, e.g. theta_cap_0, t_ml_137, cos_julian_day, stl1-3, swvl1-3,
# snowc, 2t, 2d, skt, slhf, sshf.

open_dataset(path, select=["stl1", "swvl1", "2t", ...])  # subsets variables --
# used in data/ailand_o96.yaml to restrict the 76-variable store to exactly
# v1's state vector, so extras (stl4, sd, rsn, aco2*, e, ro, 10u/10v, ...)
# never become inputs or targets by accident.
```

## Config layout (`configs/anemoi/`)

Generated once via `anemoi-training config generate --output configs/anemoi`
(the full upstream tree, diffable against a fresh `generate` to see exactly
what's aiLand-specific). New files added on top:

| File | What it does |
|---|---|
| `ailand_v1.yaml` | Top-level Phase 1 config, forked from `point_wise.yaml` |
| `system/input/ailand_o96.yaml` | Points at the real `/lus` O96 store |
| `system/output/ailand.yaml` | Output root `models/anemoi` (matches the repo's `models/` convention) |
| `data/ailand_o96.yaml` | v1's variable split: 41 forcing, 7 prognostic (implicit), 5 diagnostic, via `select` |
| `model/ailand_point_wise.yaml` | Resized point-wise model + v1's bounds (`swvl*` Relu, `snowc` Hardtanh [0,1]) |
| `task/ailand_forecaster.yaml` | Phase 1: `multistep_input: 1` (v1 has no lookback window), rollout fixed at 4 |
| `task/ailand_forecaster_r8.yaml` | Phase 2: rollout fixed at 8, used via `+task=` on resume |
| `training/training_loss/ailand.yaml` | Huber (delta=1.0), tendency + NaN-mask + time-step scalers |
| `training/ailand_pretrain.yaml` | Grad clip norm 5.0, LR 5e-4→3e-7 cosine w/ 1000-step-equivalent warmup, max_epochs=80 |
| `diagnostics/ailand_evaluation.yaml` | Same as shipped `evaluation.yaml`, mlflow's required-but-unused `tracking_uri` filled in |

**Two-phase rollout, one file each, not one growing schedule.** v1 trains
R=4 for 80 epochs then R=8 for 8 epochs — two distinct epoch counts, not a
smooth ramp. `anemoi-training`'s `RolloutConfig.epoch_increment` *can* grow
the rollout within a single run, but mapping v1's two-phase schedule onto
that would need per-phase epoch counts anemoi's scheduler doesn't expose
directly. Two runs (`slurm/anemoi_train.sh` with `PHASE=1`/`PHASE=2`, the
second resuming via `transfer_learning=True` + `system.input.warm_start`)
matches the paper's actual protocol more faithfully.

**Confirmed by reading source, not assumed:** the `stdev_tendency` scaler
(`anemoi/training/losses/scalers/variable_tendency.py`,
`BaseTendencyScaler.get_scaling_values`) only rescales variables in
`data_indices.model.output.prognostic` — it's a no-op on diagnostics. So
Phase 1 needs only **one** Huber loss term, not two (one tendency-scaled for
prognostics, one plain-normalised for diagnostics) as might be assumed by
analogy with `ailand.mlp.Normaliser`'s separate `tend` vs `d_mean`/`d_std`.
The diagnostics get their normalisation from the ordinary mean-std
`InputNormalizer` in `data/ailand_o96.yaml`, same effect, no custom loss class
needed for this phase.

## Validating the config

```bash
anemoi-training config validate --config-path configs/anemoi --config-name ailand_v1
```

## Open items (before submitting a real training job)

1. **Land-point masking — unresolved.** `open_dataset` with no further
   arguments returns all 40,320 global grid points; the paper's Table B1 and
   this repo's own reproduction score over the **11,538 O96 land points**
   only, excluding glacier/coastal too. Two candidate mechanisms found, neither
   wired up yet:
   - `anemoi.datasets`' `Masked` dataset wrapper
     (`usage/gridded/masked.py`), keyed by an `lsm_0`-derived boolean mask,
     applied to the raw dataset;
   - restricting the *graph* node builder instead (`graph/point_wise.yaml`'s
     `AnemoiDatasetNodes`) to land nodes, leaving the dataloader reading the
     full grid.

   Training or scoring before this is resolved will run over ocean points
   too and will not be comparable to Table B1.

2. **Confirmed upstream bug: `config_validation` rejects the PointWise
   model family.** `anemoi-training config validate` (and the
   `config_validation: True` flag, which gates the same `BaseSchema(**cfg)`
   check at train time) raises ~28 errors like:

   ```
   model.BaseModelSchema.processor.`anemoi.models.layers.processor.PointWiseMLPProcessor`.num_channels
     Field required [type=missing]
   model.BaseModelSchema.encoders.0.mapper.`anemoi.models.layers.mapper.PointWiseForwardMapper`.trainable_size
     Extra inputs are not permitted [type=extra_forbidden]
   ```

   **Reproduced on the pristine, unmodified shipped `point_wise.yaml`** (with
   dummy `system.input.dataset`/`graph`/`output.root`/`diagnostics.log.mlflow.tracking_uri`
   overrides just to get far enough to reach this check) — so this is not a
   mistake in `ailand_v1.yaml`. The schema classes
   (`anemoi.models.schemas.{processor,encoder,decoder}.PointWise*Schema`)
   require `num_channels` as a *direct* field and forbid `trainable_size` on
   the encoder/decoder mappers, but the shipped YAML only sets
   `num_channels` once at `model.num_channels` and *does* pass
   `trainable_size` to the mappers (as `${model.edge_trainable_parameters.*}`)
   — a mismatch between the YAML template and its own pydantic schema for a
   just-shipped model family. `ailand_v1.yaml` sets `config_validation: False`
   with a comment pointing here; whether the actual `AnemoiModelEncProcDec`
   construction path (which may inject `num_channels` into sub-configs in
   code, not YAML) tolerates this at runtime is untested — that's part of
   what the first real training attempt will tell us. Worth reporting
   upstream regardless.

3. **`num_channels`/`num_layers` sizing is an approximation.** `512` /
   `6` in `model/ailand_point_wise.yaml` approximates v1's 6×512
   (~1.3M params), not yet checked against an actual instantiated parameter
   count (blocked on (1) and (2) above — need a working model instantiation
   to run `torchinfo.summary` against).

4. **LR-scheduler epoch/step approximation.** v1: 1000-step warmup. This
   config uses `t_in_epochs: true, warmup_t: 2` (epochs) as a rough stand-in,
   since `max_epochs` rather than a precomputed `max_steps` governs Phase 1.
   Not yet checked against this dataset's actual steps/epoch at the real
   batch size.

## Still to build (Phases 2-4 of the original plan)

* **S1/S5** — `anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage`
  (freezes named submodules by path, `training.submodules_to_freeze`) covers
  S1 directly. Needs the FLUXNET-Shuttle data (`data/fluxnet_o96.zarr` /
  `fluxnet_n320.zarr`) wired up as an anemoi dataloader source first — those
  aren't anemoi-datasets recipes (no `/lus` origin), so this needs either a
  small adapter or a custom `data` config reading our Zarr layout directly.
* **S6** — soil-observation term (reuse `ailand.finetune.soil_offsets`'
  per-site mean-offset correction as a preprocessing step, not a new loss
  class) + the anchor term against a frozen pretrained model on a *second*
  stream of ordinary global-ecLand batches. Nothing in `anemoi-training`'s
  loss library does two-dataset-per-step training — likely needs a
  `Forecaster` task subclass.
* **S7** — evaporative-fraction partition term
  (`ailand.finetune.py` lines 269-294: `EF = LE/(LE+H)`, the
  `EF_MIN_ENERGY` double-sided guard, Huber on `ef_p - ef_o`) has no
  equivalent in anemoi-training's loss library — a new
  `anemoi.training.losses.base.FunctionalLoss` subclass, ported close to
  verbatim.

Verification at every phase: score with the repo's *existing*
`ailand.evaluate.score` / `ailand.finetune.evaluate_sites`, not new metric
code, so numbers stay directly comparable to `docs/RESULTS.md`.
