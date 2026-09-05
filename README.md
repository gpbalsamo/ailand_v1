# ailand — playing with the aiLand v0 prototype

This repo is a working copy of the **aiLand v0** prototype: the XGBoost notebook
`train_ai_land_example.ipynb`, taken verbatim from
[`pinnstorm/ec-land-db`](https://github.com/pinnstorm/ec-land-db)
(local checkout: `/perm/pad/ec-land-db`).

The published **aiLand v1** is a different animal: Raoult et al. (2026),
*aiLand v1: Physics-Based Land Surface Emulator with Observational Fine-Tuning*,
EGUsphere preprint [egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620).
A copy of the paper is in [`docs/papers/`](docs/papers/).

The first commit here is the untouched v0 notebook
(md5 `610e257e24ec4d0e01b086867d68fbde`), so everything since is visible as a diff.

## Layout

```
src/ailand/       config, data, train, infer, evaluate, mlp, train_mlp
tests/            tests for the ML path (the upstream repo tested only ingest)
data/             mock ecLand Zarr store (10 land points, 2020-2022, 6-hourly, 37 vars)
docs/papers/      aiLand v1 preprint (PDF + extracted text)
docs/images/      reference figures from ec-land-db
docs/upstream/    upstream README / setup.py / config.yaml / stale environment.yml
notebooks/upstream/  sibling ec-land-db notebooks + the pristine v0 notebook
models/           trained models, metadata and rollouts (gitignored)
figures/          generated figures (gitignored)
train_ai_land_example.ipynb   interactive front-end to the package
```

Everything the code needs is inside this directory — no relative paths escaping to
`../tests/mock_data`, and no dependency on `/perm/pad/ec-land-db`.

## Running it

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e .
```

Training, inference and evaluation are separate entry points, so each stage can be
run, cached and swapped independently:

```bash
# XGBoost
python -m ailand.train    --preset v1+runoff --temporal --scale-targets
python -m ailand.infer    --preset v1+runoff --temporal --point 5   # rollout -> netCDF
python -m ailand.evaluate --preset v1+runoff --temporal             # scores + plot

# MLP, with v1's two-phase multi-step rollout loss
python -m ailand.train_mlp --preset v1+runoff --temporal --rollout 4 8 --epochs 60 10
python -m ailand.evaluate  --preset v1+runoff --temporal --modeldir mlp_v1_runoff_temporal

pytest tests/
```

Useful flags: `--temporal` (time of day, day of year, TOA insolation),
`--scale-targets` (v1's tendency scalers), `--n-estimators`, `--train-years`,
`--split`, `--data`, `--multi-strategy`. `train_ai_land_example.ipynb` is a thin
interactive front-end over the same functions; the pristine upstream notebook is
kept at `notebooks/upstream/train_ai_land_example_v0_pristine.ipynb`.

**Kernel caveat.** The notebook's recorded kernel is `ec_land_db`, but
`~/.local/share/jupyter/kernels/ec_land_db/kernel.json` points at the *system*
interpreter `/usr/local/apps/python3/3.12.9-01/bin/python3.12` — there is no
virtualenv behind it, despite the upstream README's install instructions.
`requirements.txt` here is pinned from that interpreter, because it is the one
that actually produced the stored outputs.

The upstream `environment.yml` (python 3.8.16, numpy 1.24.2, xarray 2023.1.0,
zarr 2.13.6, py-xgboost 1.7.6) does not describe any environment in use here and
almost certainly no longer solves; it is kept in `docs/upstream/` for reference only.

## What changed from the pristine v0 notebook

| Change | Why |
|---|---|
| Paths resolved from the repo root | `../tests/mock_data/...` did not exist from this directory |
| `objevtive=` → `objective="reg:squarederror"` | The v0 kwarg was a typo, silently accepted by XGBoost and ignored, so the intended MAE objective never applied |
| `random_state=SEED` | `subsample=0.6` made v0 non-reproducible |
| Prognostic state indexed by name | v0 assumed the targets were the last N features *in order*; a positional assumption that breaks silently on any edit |
| Variable-specific physical bounds | v0 applied `np.clip(x, 0, None)` to everything, which is meaningless for soil temperature in K and misses the upper bound on snow cover |
| Diagnostic branch (`skt`, `aco2gpp`) | Predicted as absolute values and *not* fed back, mirroring v1's diagnostic head. Feeding them back from truth would leak into the rollout |
| Scoring cell on the held-out year | v0 judged the rollout by eye from one figure; no metric was ever computed on 2022 |
| Split into `train` / `infer` / `evaluate` scripts | A notebook cannot be run per-stage, cached, or tested |
| Presets for the v0 / v0+snow / v1 state vectors | Makes the state-vector choice measurable rather than assumed |
| `tests/` for the ML path | Upstream tested only the GRIB→Zarr ingest |

## Which state vector actually works

Three variable-set presets are built in (`ailand/config.py`), all trained
identically — 1000 trees, 2020–21, scored on the 2022 rollout at point `x=5`:

* **`v0`** — the original 9 targets, runoff included.
* **`v0+snow`** — adds the snow prognostics `sd`/`rsn` and the diagnostics `skt`/`aco2gpp`.
* **`v1`** — the aiLand v1 state vector (Raoult et al. 2026, Table 1): only
  `stl1-3`, `swvl1-3` and `snowc` are prognostic, and runoff is dropped entirely.

Held-out 2022 R² (higher is better; negative means worse than predicting the mean):

| variable | v0 | v0+snow | **v1** |
|---|---|---|---|
| `swvl1` | 0.621 | 0.750 | **0.845** |
| `swvl2` | −0.545 | −0.071 | **0.741** |
| `swvl3` | −1.235 | −1.502 | **0.668** |
| `stl1` | 0.842 | 0.775 | **0.939** |
| `stl2` | 0.880 | 0.876 | **0.987** |
| `stl3` | 0.946 | 0.914 | **0.994** |
| `snowc` | 0.551 | −1.490 | **0.713** |
| `sd` | – | −0.189 | – |
| `rsn` | – | −3.284 | – |
| `sro` | −14.405 | −13.346 | – |
| `ssro` | 0.422 | 0.354 | – |
| `skt` | – | 0.775 | 0.704 |
| `aco2gpp` | – | −0.130 | −0.082 |
| **mean** | **−1.325** | **−1.197** | **+0.723** |

Two findings, both of which reproduce v1's design decisions from the bottom up:

**1. Runoff does not belong in the prognostic state.** `sro`/`ssro` are fluxes, not
states. Their own rollout is worthless (`sro` R² ≈ −14), and because they also sit in
the *input* vector, that noise propagates into everything downstream. Dropping them
takes `swvl2` from −0.545 to 0.741 and `swvl3` from −1.235 to 0.668. This single
change is worth more than any amount of extra training.

**2. Adding the snow prognostics makes snow worse, not better.** Carrying `sd` and
`rsn` looks obviously right — `snowc` cannot close a snow budget without them — but
they roll out poorly themselves (R² −0.19 and −3.28) and feed that error straight back
into `snowc`, which collapses from 0.551 to −1.490. v1 makes the opposite choice:
`snowc` is **promoted to prognostic even though it is diagnostic in ecLand**, and the
underlying snow prognostics are left out of the state entirely.

Note how weak v0 is once measured at all: mean held-out R² of −1.325, with negative
skill on two of three soil moisture layers. The v0 figure looks convincing because the
plotted range is dominated by the seasonal cycle.

**`snowc` in this store is a percentage (0–99.9), not a fraction** — the v0 plot label
"Snow Cover Fraction (-)" is wrong, and a `[0,1]` bound on it destroys the signal.

## Temporal forcings and target scaling

`--temporal` adds v1's temporal/astronomical inputs — `cos/sin_julian_day`,
`cos/sin_local_time` (local solar time, so the diurnal cycle is in phase regardless
of longitude) and `insolation` (top-of-atmosphere, from the Spencer solar-position
fits). v0 had no notion of season or time of day at all: both entered only
indirectly through the meteorological forcing.

`--scale-targets` divides each prognostic increment by its own standard deviation
before fitting — v1's tendency scaler. Held-out 2022 R², preset `v1`, 1000 trees:

| variable | baseline | +temporal | +scaled | +both |
|---|---|---|---|---|
| `swvl1` | 0.845 | 0.881 | 0.878 | **0.898** |
| `swvl2` | 0.741 | 0.886 | 0.871 | **0.949** |
| `swvl3` | 0.668 | 0.956 | 0.857 | **0.967** |
| `stl1` | 0.939 | **0.951** | 0.946 | 0.944 |
| `stl2` | 0.987 | 0.991 | 0.991 | 0.991 |
| `stl3` | 0.994 | 0.988 | **0.995** | 0.991 |
| `snowc` | **0.713** | 0.517 | 0.702 | 0.416 |
| `skt` | 0.704 | 0.653 | **0.705** | 0.652 |
| mean | 0.723 | 0.747 | 0.763 | 0.746 |

Both help, and they compound on soil moisture: `swvl3` goes from 0.668 to **0.967**,
which is the single largest gain of any change in this repo. That matches v1's claim
that the temporal forcings specifically improve the soil column.

Two caveats worth keeping:

* **Snow gets worse, and drags the mean down.** These 10 points sit at ~51.6 °N in
  the Netherlands/Germany, where mean snow cover is 1.8% — snow is a rare, episodic
  event here, so `snowc` R² is noisy and easily degraded by anything that sharpens
  the fit elsewhere. Judge it on the paper's cold-biome results, not on these points.
* **Scaling helps less than you would expect, for a structural reason.** XGBoost's
  default `multi_strategy="one_output_per_tree"` fits each target with its own trees,
  so per-target scaling is nearly a no-op — what gain there is comes from the
  regularisation terms, which are not scale-invariant. Scaling only truly bites when
  all targets share a structure, i.e. `--multi-strategy multi_output_tree` (which is
  far worse here: mean R² −5.6) or a neural network. It is in the MLP that the
  tendency scalers do their real work.

## Runoff belongs in the output — as a diagnostic

Surface and subsurface runoff (Qs / Qsb; GRIB `sro` / `ssro`) are *fluxes generated by*
the soil column, not states *of* it. v0 carried them as prognostic variables, feeding
them back into the input vector every step. Preset `v1+runoff` keeps the v1 state
vector and recovers them as **diagnostic** outputs instead — predicted from the current
state at each step, bounded at zero, never fed back:

| | as prognostic (v0) | as diagnostic (`v1+runoff`) |
|---|---|---|
| `sro` R² | **−14.405** | **+0.234** |
| `ssro` R² | 0.422 | 0.229 |
| effect on `swvl2` | −0.545 | 0.949 |
| effect on `swvl3` | −1.235 | 0.967 |

So runoff can absolutely be an aiLand output — it simply must not be part of the state.
Its skill is still modest in absolute terms (R² 0.12–0.23; runoff is intermittent and
spiky), but it is now positive rather than catastrophic, and it costs the soil states
nothing: the prognostic scores are identical to the `v1` preset. v1 itself does not
output runoff at all, so this is an addition beyond the paper.

## The MLP

`ailand/mlp.py` + `ailand/train_mlp.py` implement the v1 architecture: a shared
prognostic backbone of Linear → LayerNorm → ReLU blocks projecting to increments,
plus a diagnostic branch, residual updates and bounds as post-processing. Training
follows v1's strategy — smooth L1 (Huber, β=1) accumulated over the rollout and
scaled by 1/R, increments normalised by tendency scalers, Adam with cosine decay and
linear warmup, gradient clipping at norm 5.0, and two phases of increasing horizon
(R=4 → R=8). Scaled down: 4 × 256 (287k params) against v1's 6 × 512 (1.3M), because
this trains on 10 grid points rather than 171,039.

Held-out 2022, preset `v1+runoff --temporal`, same data for both:

| variable | XGBoost (+temporal +scaled) | **MLP** |
|---|---|---|
| `swvl1` | 0.898 | **0.964** |
| `swvl2` | 0.949 | **0.968** |
| `swvl3` | 0.967 | **0.992** |
| `stl1` | 0.944 | **0.986** |
| `stl2` | 0.991 | **0.996** |
| `stl3` | 0.991 | **0.999** |
| `snowc` | 0.416 | **0.860** |
| `skt` | 0.670 | 0.663 |
| `sro` | 0.116 | 0.127 |
| `ssro` | **0.242** | 0.160 |
| `aco2gpp` | −0.072 | −0.074 |

The MLP wins on every prognostic variable, and the largest gain is `snowc`
(0.416 → 0.860) — the variable the rollout loss should help most, since snow errors
are exactly the kind that compound. Held-out `stl1` RMSE is 0.857 K.

This is also where the two things XGBoost structurally cannot do become available:
the model is **differentiable** end to end (v1's stated reason for choosing an MLP —
gradient-based parameter estimation and data assimilation), and `mlp.rollout_batch`
backpropagates through the autoregressive update, which is what makes the multi-step
loss possible in the first place.

## Results on the real O96 data (GPU)

Full O96 land set — **11,538 land points**, 2020–2021 training, held-out 2022, scored
as a continuous autoregressive rollout pooled over 500 points. aiLand v1 network size
(6 × 512, 1.64M params), one NVIDIA A100, **26 minutes** end to end
(`slurm/train_gpu.sh`, job 33311304).

| | RMSE | R² | v1 (paper) |
|---|---|---|---|
| `stl1` | 2.93 K | **0.980** | 1.47 K (1-yr, N320) |
| `stl2` | 2.20 K | **0.988** | — |
| `stl3` | 1.78 K | **0.992** | — |
| `swvl1` | 0.025 m³ m⁻³ | **0.973** | 0.014 (90-day) |
| `swvl2` | 0.018 m³ m⁻³ | **0.983** | — |
| `swvl3` | 0.020 m³ m⁻³ | **0.977** | — |
| `snowc` | 0.073 | **0.969** | — |
| `2t` | 4.91 K | 0.945 | **0.61–0.69 K** |
| `2d` | 1.92 K | 0.990 | — |
| `skt` | 9.39 K | 0.834 | **1.06 K** |
| `slhf` (LE) | — | **−0.337** | 10.1–11.1 W m⁻² |
| `sshf` (H) | — | **−0.670** | 11.7–12.5 W m⁻² |
| `e` (evaporation) | — | **−0.335** | — (v1 outputs LE, not E) |
| `sro` / `ssro` | — | 0.388 / 0.384 | not output by v1 |
| `aco2gpp` | — | −0.131 | not output by v1 |

**The prognostic state is close to v1.** Soil temperature and moisture and snow cover
all sit at R² 0.97–0.99, with `stl1` RMSE within a factor of two of v1's — reasonable
given O96 (~125 km) against v1's N320 (~31 km), 68 epochs against 88, and 3M of the
33.7M available rollout windows.

**The diagnostic branch is not.** `2t` at 4.91 K against v1's 0.61–0.69 K, `skt` at
9.39 K against 1.06 K, and the turbulent fluxes at negative R² — worse than predicting
their own climatology. This is a real gap, not a scaling artefact, and it is the most
useful thing this run tells us. Likely causes, in order of suspicion: a single-block
diagnostic head with no per-variable loss weighting, so 9 diagnostics of very different
difficulty share one undifferentiated term; global rather than per-point standardisation
of highly spatially variable fluxes; and simply far less training than v1.

Two internal consistency checks that did pass: `corr(slhf, e) = 0.9992` — the same
quantity in energy and water units, as it must be — and `e` and `slhf` scoring within
0.002 of each other in R².

It is worth noting that LE and H are exactly the two variables aiLand v1 fine-tunes on
FLUXNET, and the ones its abstract reports improving by 30% and 20%. Our being weakest
precisely there is consistent with them being the hard part, and is the direct argument
for `docs/STRATEGY.md`.

## aiLand v0 vs aiLand v1

| | v0 (this notebook) | v1 (Raoult et al., 2026) |
|---|---|---|
| Architecture | XGBoost, gradient-boosted trees | MLP: 6 hidden layers × 512, LayerNorm + ReLU, ~1.3M params; shared prognostic backbone + diagnostic branch |
| Differentiable | no | **yes** — the stated reason for choosing an MLP (data assimilation, parameter estimation) |
| Training data | 10 land points, 2020–21, mock store | global N320, **171,039 land points**, 1998–2019, 6-hourly; 2022 held out |
| Data pipeline | hand-rolled `xarray` stacking | `anemoi-datasets`, YAML recipes, Zarr chunked in time |
| Prognostic targets | 9, incl. runoff | `stl1-3`, `swvl1-3`, `snowc` (increments) |
| Diagnostic targets | none | `2t`, `2d`, `skt`, `LE`, `H` (absolute) |
| Temporal forcing | none | time of day, day of year, TOA insolation |
| Normalisation | none | feature-wise stats; increments divided by **tendency scalers** (std of 6-h increments) |
| Loss | RMSE (typo'd objective) | smooth L1 (Huber, β=1), accumulated over rollout, scaled by 1/R |
| Rollout training | none (single step) | Phase 1: R=4 (24 h), 80 epochs / 160k steps; Phase 2: R=8 (48 h), 8 epochs |
| Optimiser | – | Adam, peak LR 5e−4 → 3e−7 cosine, 1000-step warmup, grad clip 5.0 |
| Hardware | 1 CPU | 4 GPUs, DDP, mixed precision |
| Bounds | `clip(x, 0, None)` | variable-specific bounds as post-processing |
| Observations | none | **fine-tuning on FLUXNET** (FluxDataKit): 199 training / 42 validation sites, NaN-masked loss, 5 strategies S1–S5 |
| Evaluation | 1 point, by eye | RMSE/MAE/ACC vs climatology; 138 × 90-day integrations; 1-yr and 4-yr continuous rollouts; 6 k-means biomes; O96 ↔ N320 transfer |

v1 headline results: 90-day RMSE 1.19 K (`stl1`) and 0.014 m³ m⁻³ (`swvl1`); stable over
4-year autoregressive integration; fine-tuning cuts LE RMSE by 30%, H by 20%, per-site
Bowen ratio error by 42%, and energy-balance closure residual from 13.4 to <3 W m⁻².

The v1 code and the `aiLand-base` checkpoint are at
[10.5281/zenodo.20764680](https://doi.org/10.5281/zenodo.20764680); training data at
[10.21957/0f6t-7f73](https://doi.org/10.21957/0f6t-7f73) (N320) and
[10.21957/fs25-c406](https://doi.org/10.21957/fs25-c406) (O96).

## The v1 training data is already on this machine

See [`data/EXTERNAL.md`](data/EXTERNAL.md). The `anemoi-datasets` stores backing v1
are on `/lus` — O96 (114 GB) and N320 (1.2 TB), 1998–2024, 6-hourly, 76 variables
including the temporal/astronomical forcings and the tendency statistics v1 normalises by.
They are far too large to copy here, so they are referenced rather than migrated.

## Suggested next steps

1. **Hold out grid points, not just time.** Point `x=5` is in the training set; only
   2022 is genuinely independent. This is the weakest part of the evaluation.
2. **Move to the real data.** 10 points at one latitude cannot show cross-biome or
   cross-resolution behaviour, and make `snowc` scores nearly meaningless. The O96
   store on `/lus` (114 GB) is the tractable next step — see `data/EXTERNAL.md`.
3. **Add the missing v1 diagnostics.** The mock store has no `2t`, `2d`, `slhf` or
   `sshf`, so the diagnostic branch is running on `skt` and `aco2gpp` alone. The
   anemoi stores have all five.
4. **Fix `aco2gpp`.** The one variable with negative skill under every configuration
   and both model types. GPP is not a memoryless function of the current state and
   instantaneous forcing; it needs phenology or a memory term.
5. **Scale the MLP toward v1** (6 × 512, longer rollouts, GPU) once trained on more
   than 10 points — `--device cuda` is already wired.
6. **Exploit the differentiability.** Gradients through the emulator are the whole
   point of v1's architecture choice: parameter sensitivity, then observation-
   constrained parameter estimation.
7. **Fine-tune on observations.** v1's second stage (FLUXNET via FluxDataKit, five
   strategies S1–S5) is what corrects ecLand's own biases rather than just emulating
   them, and needs the flux diagnostics from (3) first.
