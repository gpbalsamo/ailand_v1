# `v1_anemoi`: aiLand on ECMWF's anemoi-training stack

Branch off `master`, reimplementing the aiLand v1 pipeline
(`src/ailand/mlp.py`, `train_mlp.py`, `finetune.py`) on `anemoi-training`
instead of the hand-rolled PyTorch loop, so the two can be compared, and so
aiLand converges onto ECMWF's own production ML tooling. Started because
anemoi-models shipped a genuine point-wise (no graph coupling) MLP
architecture — see "Why this exists" below.

Status: **Phase 0 done.** **Phase 1 config written, schema-validated, and
verified end to end on CPU**: a full epoch (1 training step, 1 validation
step, R=4 rollout, real O96 land data) ran to completion and saved a
checkpoint (`EXIT_CODE=0`, `Trainer.fit stopped: max_epochs=1 reached`).
Land-point masking is resolved and verified exact (11,538 nodes). Five other
real bugs surfaced by CPU smoke-testing are fixed and documented below. No
GPU job has been run yet — that's the natural next step, now that CPU
smoke-testing has exhausted what it can catch, and a real GPU-hours
commitment (`slurm/anemoi_train.sh`, sized to Nina Raoult's own v1 job: 1
node, 4×A100, `qos=ng`).

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

## Land-point masking — resolved

Restricted at the **graph** level, not the dataset level. A point-wise graph
has no edges, so `anemoi.graphs.processors.post_process.RemoveUnconnectedNodes`
would otherwise drop *every* node as "unconnected" — but its `ignore`
attribute inverts that: nodes where `ignore` is `True` are kept regardless of
connectivity. `graph/ailand_point_wise.yaml` adds a `land_mask` node
attribute — `ailand.anemoi_ext.ThresholdAnemoiDatasetVariable` (a small new
class; anemoi-graphs ships `NonzeroAnemoiDatasetVariable`, `!= 0`, and
`NonmissingAnemoiDatasetVariable`, not NaN, but nothing with a numeric
threshold — needed one, since "any nonzero land fraction" keeps far more
coastal/mixed cells than the paper's land set) reading `lsm_0 > 0.5`, matching
`ailand.extract`'s `--lsm-threshold` default exactly — then
`RemoveUnconnectedNodes(nodes_name: data, ignore: land_mask)` drops everything
else.

Verified twice: once standalone in Python (`AnemoiDatasetNodes` +
`ThresholdAnemoiDatasetVariable` + `RemoveUnconnectedNodes` on the real O96
store → **exactly 11,538 nodes**, matching the paper's own land-point count
to the digit), and again inside a real `anemoi-training train` run
("Removing 28782 nodes from data" → 40320 − 28782 = 11,538).

## Bugs found and fixed by CPU smoke-testing

None of these showed up in `anemoi-training config validate` (a static
schema check) — only running the real pipeline surfaced them. In order hit:

1. **`select` was silently inert.** Placed under `data/ailand_o96.yaml`'s own
   `datasets.data.dataset.select`, which nothing actually reads.
   `dataloader/native_grid.yaml`'s `dataset_config.dataset` resolves straight
   to `${system.input.dataset}` — *that* is the recipe the real
   `NativeGridDataset` reader and the graph builder both open. Moved `select`
   there (`system/input/ailand_o96.yaml`). Caught by an initial run loading
   all 76 raw variables and folding the 28 not listed under
   `forcing`/`diagnostic` into "prognostic" — `sd`, `rsn`, `aco2gpp`, `e`,
   `ro`, `10u`/`10v`, etc. would all have been fed back autoregressively.
2. **`mpi4py` importable but non-functional on this node** (no `libmpi.so`
   at all). `pytorch_lightning`'s SLURM/cluster-environment autodetection
   tries `from mpi4py import MPI` and expects a plain `ImportError` on
   failure — but mpi4py's own ABI probe raises `RuntimeError` first,
   uncaught, crashing every run before Trainer setup even started. Not our
   dependency (system-level, `Required-by:` empty) and not safe to touch.
   Fixed by shadowing it: `.venv-anemoi/lib/python3.13/site-packages/mpi4py/__init__.py`
   is a one-line stub that raises a clean `ImportError` — the venv's own
   site-packages is earlier on `sys.path` than the system one, so this
   shadows cleanly without touching anything shared. We don't use MPI
   (`DDPGroupStrategy` over NCCL for the real 4-GPU job, not MPI collectives).
3. **`num_channels` genuinely not auto-injected** — confirmed by reading the
   real constructor signatures (`PointWiseMLPProcessor`/`PointWiseForwardMapper`/
   `PointWiseBackwardMapper.__init__`, all keyword-only, no default) and by
   hitting the resulting `TypeError` at model-build time. The shipped
   `point_wise.yaml` sets `num_channels` once at `model.num_channels` and
   never propagates it to the sub-blocks that need it — a template bug, not
   just the schema-validator mismatch noted below. Fixed by adding
   `num_channels: ${model.num_channels}` to each of `processor`,
   `encoders.0.mapper`, `decoders.0.mapper` in `model/ailand_point_wise.yaml`,
   and dropping `trainable_size` from the mappers (none of the three real
   signatures have a matching parameter; absorbed by `**kwargs` at runtime
   either way, so it was a no-op, just a schema-validator complaint).
4. **NaNs in real forcing data** (517,650 in one batch) — `AssertionError:
   NaNs found in processed tensor after Processors`. Same class of problem
   as `docs/RESULTS.md`'s "Lessons learnt #3" (114 missing `slhf` values in
   our own extract cascading into NaN predictions), at real scale here on
   the actual `/lus` store: some static/vegetation fields (`theta_cap_0`/
   `theta_pwp_0`, `lai_hv`/`lai_lv`, ...) are undefined at some of the
   11,538 land points (bare soil, certain soil types). `ailand.mlp.Normaliser`
   sidesteps this by computing mean/std NaN-aware directly
   (`np.nanmean`/`np.nanstd`); the anemoi-native equivalent is an imputer
   *before* the normalizer in the processor chain —
   `anemoi.models.preprocessing.imputer.InputImputer` with `default: "mean"`,
   added to `data/ailand_o96.yaml`.
5. **Plotting callbacks hardcode atmospheric variable names.**
   `diagnostics/plot/settings_base.yaml`'s default `parameters` list
   (`z_500`, `t_850`, `10u`, `10v`, `tp`, `cp`, ...) and two `BatchOutputPlot`
   blocks in `detailed.yaml` don't exist in a land-surface state —
   `KeyError: 'z_500'`, uncaught, `exit code 1`, right after the validation
   sanity check. Not a training blocker either way (this repo has its own
   plotting in `ailand.evaluate`) — disabled cleanly via a new
   `diagnostics/plot/ailand.yaml` (`callbacks: []`) rather than hand-fixing
   every hardcoded list. Revisit with an aiLand `parameters` list
   (`stl1`, `swvl1`, `snowc`, `2t`, `slhf`, ...) later if per-epoch sample
   plots turn out to be useful.
6. **`anemoi-training train`'s `--config-path` is not a filesystem path.**
   Unlike `config validate` (which takes a literal directory), `train`'s
   bare `@hydra.main(config_path=None, ...)` resolves a relative
   `--config-path` against the calling module's own package location —
   `--config-path configs/anemoi` was silently reinterpreted as the Python
   package path `anemoi.training.train.configs.anemoi` and failed to find
   it. Hydra's `--config-dir` is the flag that actually adds a filesystem
   directory to the search path. `slurm/anemoi_train.sh` and the command
   above both use `--config-dir`.

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
| `training/scalers/ailand.yaml` | Same as shipped `global.yaml` minus `general_variable`/`pressure_level` (no atmospheric groups; `pressure_level` crashes without one) |
| `diagnostics/ailand_evaluation.yaml` | Same as shipped `evaluation.yaml`, mlflow's required-but-unused `tracking_uri` filled in, plotting disabled |
| `diagnostics/plot/ailand.yaml` | Disables the plot callbacks (their variable lists are atmospheric) |
| `graph/ailand_point_wise.yaml` | Land-point masking (see above) |
| `src/ailand/anemoi_ext.py` | `ThresholdAnemoiDatasetVariable`, the node-attribute class the land mask needs |

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

Schema check (known to reject the PointWise family regardless — see open
item #1 below; useful for catching *other* mistakes, not this one):

```bash
anemoi-training config validate --config-path configs/anemoi --config-name ailand_v1
```

**The real check is an actual CPU run** — cheap (a few minutes with these
limits), catches everything the schema check can't, and is what actually
found bugs #1-#5 above. Land, data loading, model build, one training step,
one validation step, checkpoint save — all for real, on a tiny slice, before
spending GPU-hours:

```bash
PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 anemoi-training train \
  --config-dir "$PWD/configs/anemoi" --config-name ailand_v1 \
  system.hardware.accelerator=cpu \
  dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 \
  dataloader.num_workers.training=1 dataloader.num_workers.validation=1 \
  dataloader.batch_size.training=1 dataloader.batch_size.validation=1 \
  training.max_epochs=1
```

Last run: `EXIT_CODE=0`. Model 1.6M params; `Parameter stl1/stl2/.../snowc is
being scaled by statistic_tendencies` (confirms the tendency scaler hits
exactly the 7 prognostic variables, nothing else); one training step
(~2m40s on CPU — R=4 rollout, expect far faster on an A100); one validation
step; checkpoints written with metadata; `Trainer.fit stopped: max_epochs=1
reached`. Loss values themselves (`~1.6e5` train, `~2.4e5` val) aren't
diagnostic from a single random-init step — what matters here is that it
completed, not the number.

Use `num_workers=1`, not `0` — anemoi's `MultiDataset.__iter__` assumes
`worker_id` is set by the normal DataLoader worker-init path, which never
runs at `num_workers=0` (`AttributeError: 'MultiDataset' object has no
attribute 'worker_id'`); with `num_workers=0` you'd also need to separately
override `dataloader.prefetch_factor=null` and
`dataloader.persistent_workers=false`, both invalid for zero workers — not
worth it when `num_workers=1` just works.

**A note on the login node**: this node runs other users' unrelated,
long-running jobs (large `rsync` transfers, `anemoi-datasets` builds) that
can make a CPU smoke test appear to hang — one attempt sat at `0:00` CPU
time for 15+ minutes with zero output even with `PYTHONUNBUFFERED=1`, which
turned out to be pure node contention, not a bug: killing it and retrying
moments later ran cleanly in under 4 minutes. If a smoke test seems stuck,
check `ps aux` for what else is running before assuming the config is at
fault.

## First real GPU run: Phase 1 (R=4, 80 epochs) + rollout-8 continuation

Two more real bugs surfaced once the CPU-verified config actually ran on GPU —
same pattern as above: config validation and the CPU dry run stayed silent on
both.

7. **LR scheduler interval mismatch, flatlined the first attempt.**
   `t_in_epochs: true` with `t_initial: 80` assumes the scheduler is stepped
   once per epoch, but `pl_lr_scheduler.interval` was left at its shipped
   default of `step` — Lightning called `.step()` on every training step,
   passing the cumulative step count, which timm's `CosineLRScheduler` takes
   literally against `t_initial=80`. By ~80 steps into epoch 0 the LR had
   already collapsed to `lr_min` (3e-7) and stayed there — caught after ~4
   GPU-hours on job `35721973` (`train_multi_dataset_loss_epoch` pinned at
   1.66e5 for all 38 completed epochs, no measurable learning). Fixed by
   setting `pl_lr_scheduler.interval: epoch` in
   `configs/anemoi/training/ailand_pretrain.yaml` (commit `6d90e67`).
8. **Phase 2's `+task=` failed with "Multiple values for task".**
   `slurm/anemoi_train.sh`'s Phase 2 branch used Hydra's add-key syntax
   (`+task=ailand_forecaster_r8`), but `task` already has a default in the
   base config (`defaults: [..., task: ailand_forecaster]`) — Hydra rejects
   adding a key that already exists. Plain override (`task=...`) is correct.
9. **Phase 2 silently skipped loading the Phase 1 weights.** Setting
   `training.transfer_learning=True` alone does nothing by itself —
   `anemoi.training.train.train.AnemoiTrainer` only enters the weight-loading
   branch (and picks `transfer_learning_loading` vs. a plain
   `load_from_checkpoint` within it) when `training.load_weights_only` is
   also `True`. Without it, two things went wrong at once: the model started
   from random init (the whole loading block was skipped), and `ckpt_path`
   defaulted to the Phase 1 checkpoint anyway (`self.last_checkpoint` when
   not weights-only), so Lightning tried to restore *full trainer state*
   (`current_epoch=79`) against `max_epochs=8` and crashed with
   `MisconfigurationException`. Fixed by adding
   `training.load_weights_only=True` alongside `transfer_learning=True`.

Both #8 and #9 were bugs in `slurm/anemoi_train.sh`'s Phase 2 branch, never
exercised before (see file header at the time: "not yet run on GPU"); fixed
and verified end to end, commit `2e5b437`.

**Phase 1 also needed a mid-run resume**, unrelated to a bug: the original job
(`35800355`) hit its own `--time=06:00:00` SLURM limit at epoch 57/80 and was
killed. `anemoi-training` resumes full trainer state (optimizer, LR schedule,
epoch/step counters) via `training.run_id=<run-id>` reusing the same
checkpoint directory — distinct from `fork_run_id`/`warm_start`, which start a
*new* run from another run's weights only (that's what Phase 2 uses). See
`slurm/anemoi_train_resume.sh`. A resume job (`35897945`) picked up from
`last.ckpt` (epoch 56) and finished the remaining 24 epochs cleanly.

**Run stats** (1 node, 4× NVIDIA A100-SXM4-40GB, 48 CPUs, `qos=ng`):

| | rollout | epochs | wall clock | GPU-hours | SBU | steps | throughput |
|---|---|---|---|---|---|---|---|
| Phase 1 (2 jobs: original + resume) | R=4 | 80 | 7h 57m | ≈31.8 | 3,260.6 | 336,000 | ~12.2 it/s |
| Rollout-8 continuation | R=8 | 8 | 1h 28m | ≈5.9 | 604.1 | 33,600 | ~6.4–6.7 it/s |
| **Total** | | | **9h 26m** | **≈37.7** | **3,864.7** | | |

(Excludes two near-instant failed Phase 2 attempts before bugs #8/#9 were
fixed — 19s and 37s, ~6.4 SBU combined.) R=8 throughput is roughly half of
R=4's, as expected for double the rollout length per step. Model: 1.6M
trainable params (matching the hand-rolled pipeline's `v1+fluxes` preset).
Data: 33,596 training / 1,456 validation anchors from the real
`ecland-era5met` O96 zarr store. Checkpoint storage: Phase 1 run
`e87471d5-9e0f-4236-96fb-288f89cb557c` 2.2 GB, rollout-8 run
`0904de8a-fa38-4287-a476-89e1e6d62f38` 316 MB.

**Not yet checked: exact per-epoch loss trend.** The tqdm progress-bar postfix
(`train_multi_dataset_loss_epoch`, `val_multi_dataset_loss_epoch`) only prints
3 significant figures and looks frozen at `train≈2.04e4`/`val≈2.13e4` from
epoch 2 onward for the rest of both runs. Confirmed this is a display-rounding
artifact, not a real plateau, by diffing model weights directly between the
epoch-40 and epoch-53 Phase 1 checkpoints: encoder embedding weight changed
~18% in relative norm, processor MLP biases 8–26% — the model kept learning
throughout. No exact per-epoch loss curve is available since the configured
`mlflow` logger's output directory (`models/anemoi/logs/mlflow`) was never
created (logger likely disabled) — worth wiring up mlflow or tensorboard
properly before relying on logged loss values rather than this workaround.

## Open items (minor, non-blocking)

1. **Upstream schema bug, still worth reporting.** `anemoi-training config
   validate` (and the `config_validation: True` flag, gating the same
   `BaseSchema(**cfg)` check at train time) rejects the PointWise model
   family's own shipped template — `num_channels` "Field required" on
   `PointWiseMLPProcessor`/`PointWiseForwardMapper`/`PointWiseBackwardMapper`,
   `trainable_size` "Extra inputs are not permitted" on the mappers.
   **Reproduced on the pristine, unmodified shipped `point_wise.yaml`** (with
   dummy dataset/graph/output/mlflow overrides just to reach the check), so
   this is an upstream mismatch between the YAML template and its own
   pydantic schema, not a mistake here. Confirmed the *runtime* path has the
   matching real bug too (see fix #3 above) — `config_validation: False`
   stays set with a comment pointing here, since our config is now
   internally consistent even though the schema still complains.

2. **Param count checked, not yet apples-to-apples verified against v1.**
   The CPU smoke test's model summary reports **1.6M params** — matching the
   hand-rolled pipeline's own `v1+fluxes` preset at 6×512 exactly (see
   `slurm/logs/ailand-repro.34049198.out`: "parameters: 1,636,880"). `512`/`6`
   in `model/ailand_point_wise.yaml` is confirmed right in that sense; not
   yet checked whether the point-wise encoder/decoder's own parameter
   overhead (vs. `AiLandMLP`'s plain `Linear`) changes the *effective*
   capacity at matched param count.

3. **LR-scheduler epoch/step approximation.** v1: 1000-step warmup. This
   config uses `t_in_epochs: true, warmup_t: 2` (epochs) as a rough stand-in,
   since `max_epochs` rather than a precomputed `max_steps` governs Phase 1.
   Not yet checked against this dataset's actual steps/epoch at the real
   batch size (46,152, per `docs/RESULTS.md`).

4. **Checkpoint metadata saves the full 40,320-point grid's lat/lon**, not
   the 11,538-point masked subset the model actually trained on (seen in the
   CPU smoke test's checkpoint-saving log lines). Likely just describes the
   underlying dataset's coordinate reference system for `anemoi-inference`
   tooling, independent of which subset was used for training — but not
   confirmed harmless, worth a second look before relying on
   `anemoi-inference` for scoring rather than adapting `ailand.evaluate`.

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
