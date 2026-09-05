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
data/          mock ecLand Zarr store (10 land points, 2020-2022, 6-hourly, 37 vars)
docs/papers/   aiLand v1 preprint (PDF + extracted text)
docs/images/   reference figures from ec-land-db
docs/upstream/ the upstream README / setup.py / config.yaml / stale environment.yml
notebooks/upstream/  the sibling ec-land-db notebooks (zarr store creation, exploration)
models/        trained model artefacts (gitignored)
figures/       generated figures (gitignored)
train_ai_land_example.ipynb   the working notebook
```

Everything the notebook needs is now inside this directory — no relative paths
escaping to `../tests/mock_data`, and no dependency on `/perm/pad/ec-land-db`.

## Running it

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab train_ai_land_example.ipynb
```

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
| `PRESET` switch | Lets you compare the v0, v0+snow and v1 state vectors directly |

**`snowc` in this store is a percentage (0–99.9), not a fraction** — the v0 plot
label "Snow Cover Fraction (-)" is wrong, and any `[0,1]` bound on it destroys the signal.

## Does adding the snow prognostics help?

The mock store carries four variables v0 never used: `sd`, `rsn`, `skt`, `aco2gpp`.
Adding `sd`/`rsn` to the prognostic state looks like an obvious win — `snowc` cannot
close a snow budget without them. Measured on the 2022 rollout at point `x=5`, it is not:

| variable | v0 RMSE | v0 R² | +snow RMSE | +snow R² |
|---|---|---|---|---|
| swvl1 | 0.0505 | **0.632** | 0.0416 | **0.750** |
| swvl2 | 0.0836 | −0.513 | 0.0704 | −0.071 |
| swvl3 | 0.0896 | −1.238 | 0.0948 | −1.502 |
| stl1 | 2.997 | **0.834** | 3.486 | **0.775** |
| stl2 | 2.162 | 0.880 | 2.202 | 0.876 |
| stl3 | 1.151 | 0.947 | 1.464 | 0.914 |
| snowc | 2.296 | **0.560** | 5.462 | **−1.490** |
| sd | – | – | 0.000317 | −0.189 |
| rsn | – | – | 51.37 | −3.284 |
| sro | 8.21e−05 | −14.40 | 7.93e−05 | −13.35 |
| ssro | 2.22e−04 | 0.339 | 2.19e−04 | 0.354 |

Soil moisture improves clearly; soil temperature degrades slightly; **snow cover gets
much worse**, because `sd` and `rsn` themselves roll out poorly (R² −0.19 and −3.28)
and then feed that error back into `snowc`.

This is exactly the choice v1 made and we did not: in v1 (Table 1) the prognostic
state is *only* `stl1-3`, `swvl1-3` and `snowc`, with **`snowc` promoted to prognostic
even though it is diagnostic in ecLand**, and `sd`/`rsn` left out of the state entirely.
Set `PRESET = "v1"` to run that configuration.

Note also how weak the v0 numbers are once measured: negative R² on `swvl2`, `swvl3`
and `sro` in *both* configurations. The v0 figure looks convincing because the plotted
range is dominated by the seasonal cycle; the skill against ec-land is not there.

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

1. **Add the temporal/astronomical forcings** (time of day, day of year, TOA insolation).
   v1 reports these specifically improve soil temperature, and they are trivial to compute.
2. **Normalise the increment targets by their own standard deviation.** Target std spans
   four orders of magnitude here (`stl1` 2.54 K vs `ssro` 1.5e−4 m), so a single summed
   RMSE is driven almost entirely by temperature and snow. This is v1's "tendency scaler".
3. **Hold out grid points, not just time.** Point `x=5` is in the training set; only 2022
   is genuinely independent.
4. **Add multi-step rollout loss.** Single-step training is why errors compound; v1's
   two-phase R=4 → R=8 schedule is the direct fix.
5. **Switch to a small MLP in torch** (already installed) once the above is in place —
   that is the actual v0 → v1 jump, and it is what buys differentiability.
