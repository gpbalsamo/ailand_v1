# ailand_v1

Reproducing and extending **aiLand**, the machine-learning emulator of ECMWF's
ecLand land surface model — from the original XGBoost prototype (v0) to the
published MLP architecture (v1), then beyond it with FLUXNET-Shuttle
observational fine-tuning.

*Private repo, for ECMWF colleagues.*

---

## What this is

* **aiLand v0** — the XGBoost notebook `train_ai_land_example.ipynb`, taken verbatim
  from [`pinnstorm/ec-land-db`](https://github.com/pinnstorm/ec-land-db).
* **aiLand v1** — Raoult et al. (2026), *aiLand v1: Physics-Based Land Surface Emulator
  with Observational Fine-Tuning*, EGUsphere preprint
  [egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620). Reimplemented
  here as `ailand.mlp` / `ailand.train_mlp` and reproduced against the paper's own
  Table B1 numbers. v1 builds on **Wesselkamp et al. (2025)**, the LSTM/XGBoost/MLP
  comparison that selected the architecture.

The goal, in order: reproduce v1 to exclude bugs and misrepresentation, then extend
it — with runoff and evaporation as outputs, and with soil moisture and soil
temperature observations from the 775-site FLUXNET Shuttle pool.

**More detail:** [`docs/METHODS.md`](docs/METHODS.md) explains every script and every
ML term in ecLand language, and attributes each recipe to the v1 paper, the wider
literature, or a choice made here. [`docs/STRATEGY.md`](docs/STRATEGY.md) is the plan
for going beyond v1 using observations. [`docs/RESULTS.md`](docs/RESULTS.md) is the
full experiment log — every ablation, rejected hypothesis, and lesson learnt behind
the headline numbers below.

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

Related repos: [`fluxnet-shuttle-ecland`](https://github.com/gpbalsamo/fluxnet-shuttle-ecland)
(775 flux-tower sites run through ecLand — the observational resource behind
[`docs/STRATEGY.md`](docs/STRATEGY.md)), [`plumber2-ecland`](https://github.com/gpbalsamo/plumber2-ecland),
and [`ecland-portal`](https://github.com/gpbalsamo/ecland-portal).

---

## Results

### v1 reproduction

Full v1 recipe on O96 (11,538 land points, 1998–2019 training, 2022 held out),
scored exactly as the paper's Table B1 — single-timestep, glacier and coastal excluded:

| | paper | ours | ratio |
|---|---|---|---|
| `swvl1` | 0.01040 | 0.010638 | 1.02× |
| `stl1` | 1.056 K | 1.117 K | 1.06× |
| `snowc` | 0.01462 | 0.01717 | 1.17× |
| `2t` | 0.876 K | 1.486 K | 1.70× |
| `LE` | 9.68 W m⁻² | 14.95 | 1.54× |

**The prognostic state reproduces v1** — all seven prognostic variables at parity or
better, mean R² 0.989. The diagnostic branch (`2t`, `skt`, the turbulent fluxes) still
lags; five candidate causes have been tested and rejected (data volume, head size,
missing inputs, RMSE convention, sample diversity) — see `docs/RESULTS.md`.

Adding runoff, evaporation and GPP as extra diagnostic outputs is free: evaporation
reaches R² 0.964, GPP 0.984, runoff 0.335/0.435, with no cost to the seven variables
above.

### FLUXNET-Shuttle fine-tuning

Fine-tuned `aiLand-base` on the Shuttle pool at N320, scored 6-hourly at held-out
towers:

| | LE | H | `swvl1` | Bowen MAE | EB residual |
|---|---|---|---|---|---|
| base | 51.29 | 59.21 | 0.0925 | 6.996 | 9.193 |
| S1 (freeze) | 45.41 | **54.29** | 0.0925 | 8.446 | 4.792 |
| **S7 (constrained + partition)** | **44.28** | 56.09 | **0.0861** | **6.728** | **≈0** |

Every strategy improves the fluxes 9–10%. Only the constrained strategies (S6/S7)
also improve the prognostic soil state (`swvl1`, `stl1-3`), and adding the
evaporative-fraction partition term (S7) drives energy-balance closure to
essentially zero. Full strategy comparison, the O96→N320 ablation, and scoring
against the paper's own tables are in `docs/RESULTS.md`.

---

## Reproducing

```bash
git clone git@github.com:gpbalsamo/ailand_v1.git
cd ailand_v1
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

The v1 training data (`anemoi-datasets` Zarr stores, O96 114 GB / N320 1.2 TB) is
already on `/lus` and referenced rather than migrated — see
[`data/EXTERNAL.md`](data/EXTERNAL.md).

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
