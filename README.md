# ailand-ecland

Reproducing and extending **aiLand**, the machine-learning emulator of ECMWF's
ecLand land surface model — starting from the original XGBoost prototype and
working towards the published v1.

*Private repo, for ECMWF colleagues.*

---

## What this is

Two things sit side by side here:

* **aiLand v0** — the XGBoost notebook `train_ai_land_example.ipynb`, taken verbatim
  from [`pinnstorm/ec-land-db`](https://github.com/pinnstorm/ec-land-db). The first
  commit is that file untouched (md5 `610e257e24ec4d0e01b086867d68fbde`), so every
  change since is visible as a diff.
* **aiLand v1** — Raoult et al. (2026), *aiLand v1: Physics-Based Land Surface Emulator
  with Observational Fine-Tuning*, EGUsphere preprint
  [egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620). Reimplemented
  here as `ailand.mlp` / `ailand.train_mlp` and being reproduced against the paper's
  own Table B1 numbers. v1 builds on **Wesselkamp et al. (2025)**, the LSTM/XGBoost/MLP
  comparison that selected the architecture — see [Lineage](#lineage).

The goal in order: reproduce v1 to exclude bugs and misrepresentation, then extend it —
with runoff and evaporation as outputs, and with soil moisture and soil temperature
observations from the 775-site FLUXNET Shuttle pool.

**Read first:** [`docs/METHODS.md`](docs/METHODS.md) explains every script and every
ML term in ecLand language, and attributes each recipe to the v1 paper, the wider
literature, or a choice made here. [`docs/STRATEGY.md`](docs/STRATEGY.md) is the plan
for going beyond v1 using observations.

---

## Lineage

Three generations of the same idea, and this repo works across all of them:

| | What | Where |
|---|---|---|
| **v0** | XGBoost on a 10-point mock dataset. The example notebook that started this. | [`pinnstorm/ec-land-db`](https://github.com/pinnstorm/ec-land-db) |
| **Wesselkamp et al. (2025)** | The systematic comparison: LSTM vs gradient boosting vs feed-forward networks as prognostic state emulators of ecLand. Found the LSTM best for long-range soil temperature and snow, XGBoost robust for soil moisture, and the MLP the best accuracy/efficiency trade-off — which is why v1 is an MLP. | GMD **18**, 921–937 |
| **v1** | The published emulator: MLP with a diagnostic branch, global N320 pretraining, then observational fine-tuning on FLUXNET. | EGUsphere preprint |

v1 describes itself as "building substantially on the prototype introduced in
Wesselkamp et al. (2025)" — wider and deeper, with longer rollouts, an extended input
space (the temporal and astronomical forcings) and a set of diagnostic outputs. So the
progression is: *does ML work at all* (v0) → *which architecture* (Wesselkamp 2025) →
*make it global, stable and observation-corrected* (v1).

### References

- Raoult, N., Pinnington, E., Santa Cruz, M., Pinault, F., Raoult, B., Zelenka, N.,
  Arduini, G., Balsamo, G., Boussetta, S., Chantry, M., de Rosnay, P., Dueben, P., and
  Rüdiger, C. (2026). *aiLand v1: Physics-Based Land Surface Emulator with Observational
  Fine-Tuning.* EGUsphere preprint.
  [doi:10.5194/egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620)
  — code and the `aiLand-base` checkpoint at
  [doi:10.5281/zenodo.20764680](https://doi.org/10.5281/zenodo.20764680); training data at
  [doi:10.21957/0f6t-7f73](https://doi.org/10.21957/0f6t-7f73) (N320) and
  [doi:10.21957/fs25-c406](https://doi.org/10.21957/fs25-c406) (O96).

- Wesselkamp, M., Chantry, M., Pinnington, E., Choulga, M., Boussetta, S., Kalweit, M.,
  Bödecker, J., Dormann, C. F., Pappenberger, F., and Balsamo, G. (2025). *Advances in
  land surface forecasting: a comparison of LSTM, gradient boosting, and feed-forward
  neural networks as prognostic state emulators in a case study with ecLand.*
  Geoscientific Model Development **18**, 921–937.
  [doi:10.5194/gmd-18-921-2025](https://doi.org/10.5194/gmd-18-921-2025)

- Boussetta, S., Balsamo, G., et al. (2021). *ECLand: The ECMWF land surface modelling
  system.* Atmosphere **12**, 723. — the physical model being emulated;
  source at [`ecmwf-ifs/ecland`](https://github.com/ecmwf-ifs/ecland).

Related repos in this line of work: [`fluxnet-shuttle-ecland`](https://github.com/gpbalsamo/fluxnet-shuttle-ecland)
(775 flux-tower sites run through ecLand — the observational resource behind
[`docs/STRATEGY.md`](docs/STRATEGY.md)), [`plumber2-ecland`](https://github.com/gpbalsamo/plumber2-ecland),
and [`ecland-portal`](https://github.com/gpbalsamo/ecland-portal).

---

## Lessons learnt

The most useful output of this exercise is not the model, it is the list of things
that were quietly wrong. Every one produced plausible-looking results.

### 1. Physically obvious is not empirically right

Adding the snow prognostics `sd` and `rsn` looked obligatory — `snowc` cannot close a
snow budget without them. Measured, it made snow *worse*: `snowc` fell from 0.55 to
−1.49, because `sd` and `rsn` roll out badly themselves and feed that error back. v1
makes the opposite choice: promote `snowc` to prognostic although it is diagnostic in
ecLand, and leave the snow mass variables out of the state. We reproduced that decision
from measurement before finding it in Table 1.

### 2. Read the error signature, not just the error

Turbulent fluxes scored R² −0.34 with near-zero bias. That combination is a *phase*
error, not an underfit — and it was: diagnostics are trained on the target at t+1 from
the input at t, but the rollout wrote each prediction at t. A six-hour shift, harmless
for `2t` (which loses 2% of R² to it) and fatal for fluxes, whose diurnal cycle it moves
a quarter period. Fixing the index alone, with **no retraining**, moved `slhf` −0.34 →
0.96, `sshf` −0.67 → 0.96, `skt` 9.39 K → 2.13 K.

### 3. Silent NaN propagation deletes whole variables

114 missing values out of 33.7 million made a plain `mean`/`std` NaN, which made every
`slhf` prediction NaN, which the scorer then masked out — so the variable vanished from
the results table with no error anywhere. Statistics are now NaN-aware and an
entirely-missing column raises.

### 4. Scale the targets, or the loss is only about the biggest number

Increment magnitudes span four orders of magnitude (`stl1` ~2.5 K against `ssro`
~1.5×10⁻⁴ m). An unweighted least-squares loss is *entirely* soil temperature and snow.
This is v1's "tendency scaler", and it is the same idea as dividing by observation error
variance in a variational cost function. The same trap caught the diagnostics separately:
in physical units `slhf` ~10⁷ J m⁻² swamps evaporation ~10⁻³ m.

### 5. Judge by metric, never by figure

v0's headline figure looks convincing because its y-range is dominated by the seasonal
cycle, and no metric was ever computed on the held-out year at all — the reported R² of
0.99 was `eval_set` pointed at the training data, so it is in-sample and means nothing.
Scored properly on 2022, v0's soil and snow state comes out at mean R² **0.841**, which
is a perfectly respectable prototype. But that is something you can only say once you
measure it.

### 6. Compare like with like before concluding anything is wrong

The MLP looked far short of v1 until three things were corrected: the misalignment
above, **excluding glacier and coastal points** (which v1 does throughout, and which are
22.3% of the O96 land set with ~3× the soil temperature error), and scoring a free
3-year rollout against the paper's *single-timestep* table. Corrected, the prognostic
state is at parity — see below.

### 7. Small silent bugs in your own tooling

`land[::step][:npoints]` truncates rather than spans, so the first global sample
silently contained nothing south of 30°S. Sampling is systematic, not stratified — a
weakness, given the biomes this work cares about are the rare ones.

---

## Where the reproduction stands

Full v1 recipe: O96, **11,538 land points, 1998–2019 training**, 2022 held out for
validation and best-checkpoint selection, 6×512, R=4→R=8, Adam 5e-4→3e-7 cosine with
1000-step warmup, batch 46,152. One A100, 5 h.

Scored exactly as the paper's Table B1 — single-timestep, glacier and coastal excluded:

| | paper O96 | ours (2 yr) | **ours (22 yr)** | ratio |
|---|---|---|---|---|
| `swvl1` | 0.01040 | 0.01050 | **0.010638** | 1.02× |
| `swvl2` | 0.003115 | 0.003250 | **0.003180** | 1.02× |
| `swvl3` | 0.000746 | 0.000791 | **0.000738** | **0.99×** |
| `stl1` | 1.056 K | 1.082 K | **1.117 K** | 1.06× |
| `stl2` | 0.138 K | 0.166 K | **0.154 K** | 1.12× |
| `stl3` | 0.032 K | 0.0128 K | **0.0109 K** | **0.34×** |
| `snowc` | 0.01462 | 0.05435 | **0.01717** | 1.17× |
| `2d` | 0.857 K | 2.099 K | 2.039 K | 2.38× |
| `2t` | 0.876 K | 1.608 K | 1.486 K | 1.70× |
| `skt` | 1.081 K | 1.986 K | 1.866 K | 1.73× |
| `H` | 11.57 W m⁻² | 17.62 | 17.24 | 1.49× |
| `LE` | 9.68 W m⁻² | 15.44 | 14.95 | 1.54× |

**The prognostic state reproduces v1** — every one of the seven at parity or better,
mean R² 0.989. The 22-year training period did what it was expected to do for snow
(`snowc` 3.72× → 1.17×), confirming that two years simply cannot represent snow
variability.

**It did not close the diagnostic gap**, which barely moved (`2t` 1.61 → 1.49 K).
Five hypotheses have now been tested and rejected:

| Hypothesis | Test | Verdict |
|---|---|---|
| Too little training data | 2 yr → 22 yr | **No** — fixed snow (3.7× → 1.17×), not diagnostics |
| Diagnostic head too small | 1 → 3 blocks, 3× flux weights | **No** — no change |
| Missing albedo input | checked v1's Table 1 | **No** — v1 does not use albedo either; our input set already matches its 14 static + 7 dynamic + 3 temporal fields |
| RMSE aggregation convention | pooled vs per-gridpoint | Partly — `LE` 1.54× → 1.38×, rest ~5% |
| Too few distinct start times | 1,040 → 30,000, same budget | **No** — `2t` 1.4862 → 1.4836 K, i.e. nothing |

What remains is the gradient budget: we are still ~5–10× short of the paper's, whose
every update spans all land points. That is a resource difference, not a method one.
With the prognostic state at parity and stable in free rollout, this is a good enough
reproduction to build on.

Free autoregressive rollout through 2022, same points: mean R² 0.974, `stl1` 2.45 K,
`swvl1` 0.0191, `2t` 1.55 K — so the model is stable, not just accurate one step out.

### Adding runoff, evaporation and GPP is free

The `v1+fluxes` run is identical to `v1` on v1's own variables (`stl1` 1.121 vs 1.117 K,
`swvl1` 0.010642 vs 0.010638) while additionally predicting evaporation at R² **0.964**,
GPP at **0.984**, and runoff at 0.335 / 0.435. The extra diagnostic outputs cost nothing.

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

## Which state vector

Presets in `ailand/config.py`, all trained identically — 1000 trees, 2020–21, scored on
the 2022 rollout at point `x=5`:

* **`v0`** — v0's soil and snow state: `swvl1-3`, `stl1-3`, `snowc`.
* **`v0+snow`** — adds the snow prognostics `sd`/`rsn` and the diagnostics `skt`/`aco2gpp`.
* **`v1`** — the aiLand v1 state vector (Raoult et al. 2026, Table 1), which is the same
  seven prognostic variables, plus v1's diagnostic outputs.
* **`v0-asdistributed`** — the notebook exactly as shipped, with runoff carried in the
  state. Kept only so the original remains runnable; see the note below.

Held-out 2022 R²:

| variable | v0 | v0+snow |
|---|---|---|
| `swvl1` | **0.845** | 0.750 |
| `swvl2` | **0.741** | −0.071 |
| `swvl3` | **0.668** | −1.502 |
| `stl1` | **0.939** | 0.775 |
| `stl2` | **0.987** | 0.876 |
| `stl3` | **0.994** | 0.914 |
| `snowc` | **0.713** | −1.490 |
| mean | **0.841** | −1.197 |

**v0's state vector was already the right one.** Its soil and snow emulation scores 0.84
on a held-out year; the gains later in this repo come from the temporal forcings, target
scaling and the move to an MLP, not from changing what is in the state.

**Adding the snow prognostics makes snow worse.** Carrying `sd` and `rsn` looks
physically obligatory — `snowc` cannot close a snow budget without them — but they roll
out poorly themselves and feed that error straight back into `snowc`, which collapses
from 0.713 to −1.490. v1 makes the opposite choice: `snowc` is promoted to prognostic
even though it is diagnostic in ecLand, and the snow mass variables are left out of the
state entirely. Worth knowing before anyone else proposes the same thing.

**A note on runoff.** The notebook as distributed also listed `sro`/`ssro` among its
targets, which puts a flux into the state vector and, because the state is also the
input, feeds its noise back into soil moisture. That is a data-configuration slip rather
than anything intrinsic to the v0 design, so the `v0` preset above states the design as
intended. Runoff is a genuinely useful *output* — see below — just not a state.

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

## Runoff as a diagnostic output

aiLand v1 does not output runoff at all. Surface and subsurface runoff (Qs / Qsb; GRIB
`sro` / `ssro`) are fluxes generated by the soil column, so they belong on the
**diagnostic** branch — predicted from the current state at each step, bounded at zero,
never fed back. Preset `v1+runoff` does exactly that, and `v1+fluxes` adds evaporation
and GPP alongside.

It works, and it is free: `sro` reaches R² +0.234 and `ssro` +0.229, while the
prognostic scores stay identical to the `v1` preset. Skill is modest in absolute terms —
runoff is intermittent and spiky — but it is real, and it is an output the published
emulator does not provide.

For completeness, the same variables carried as *state* instead score `sro` −14.4, and
drag `swvl2` to −0.545 and `swvl3` to −1.235 with them: a flux in the state vector is
also an input, so its noise propagates into the soil column. That is the reason for the
diagnostic branch, not a criticism of any particular configuration.

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

## Full O96 results (free rollout)

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
