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
src/ailand/       the package: config, data, train, infer, evaluate
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
python -m ailand.train    --preset v1                  # fit, save to models/v1/
python -m ailand.infer    --preset v1 --point 5        # autoregressive rollout -> netCDF
python -m ailand.evaluate --preset v1                  # score held-out year + plot
pytest tests/
```

Useful flags: `--scale-targets` (v1's tendency scalers), `--n-estimators`,
`--train-years`, `--split`, `--data`. `train_ai_land_example.ipynb` is a thin
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

1. **Normalise the increment targets** (`--scale-targets` is already wired). Target std
   spans four orders of magnitude, so an unweighted loss is driven almost entirely by
   soil temperature and snow. This is v1's tendency scaler.
2. **Add the temporal/astronomical forcings** — time of day, day of year, TOA insolation.
   v1 reports these specifically improve soil temperature, they are trivial to compute,
   and the `/lus` stores already carry them precomputed as `cos/sin_julian_day`,
   `cos/sin_local_time` and `insolation`.
3. **Hold out grid points, not just time.** Point `x=5` is in the training set; only 2022
   is genuinely independent.
4. **Add multi-step rollout loss.** Single-step training is why errors compound; v1's
   two-phase R=4 → R=8 schedule is the direct fix — and it is not expressible in XGBoost,
   which is one concrete reason v1 is an MLP.
5. **Switch to a small MLP in torch** (already installed) once the above is in place.
   That is the actual v0 → v1 jump, and it is what buys differentiability — the property
   v1 needs for data assimilation and parameter estimation, and the one thing gradient-
   boosted trees can never provide.
6. **Fix `aco2gpp`.** It is the one variable with negative skill in every preset. GPP is
   not a memoryless function of the current state and instantaneous forcing.
