# External datasets (referenced, not migrated)

`data/ecland_cy49r2_2020_2022.zarr` (1 MB, 10 land points) is the only store copied
into this repo. The real aiLand training data lives on `/lus` and is far too large
to migrate; the paths below are the pointers.

## aiLand v1 training data (anemoi-datasets format)

These match the paper's description exactly — 1998–2024, 6-hourly, ecLand forced by
ERA5, on both grids used in the resolution-transfer experiment (Appendix B).

| Grid | Path | Size |
|---|---|---|
| O96 (~125 km) | `/lus/h1aiws01/project/ai-ml/datasets/aifs-rd-an-oper-isc8-mars-o96-1998-2024-6h-v1-ecland-era5met.zarr` | 114 GB |
| N320 (~31 km) | `/lus/h1aiws01/project/ai-ml/datasets/aifs-rd-an-oper-iscb-mars-n320-1998-2024-6h-v2-ecland-era5met.zarr` | 1.2 TB |
| N320 (v1) | `/lus/h1aiws01/project/ai-ml/datasets/aifs-rd-an-oper-iscb-mars-n320-1998-2024-6h-v1-ecland-era5met.zarr` | – |

O96 store shape: `(39444, 76, 1, 40320)` float32 — (time, variable, ensemble, gridpoint),
`frequency: 6h`, `start_date: 1998-01-02`, `end_date: 2024-12-31`, `resolution: O96`.

The 76 variables include everything v1 needs and v0 lacks:

* temporal/astronomical forcing: `cos_julian_day`, `sin_julian_day`, `cos_local_time`,
  `sin_local_time`, `insolation`, `cos_latitude`, `sin_latitude`, `cos_longitude`, `sin_longitude`
* soil hydraulic statics: `theta_cap_0` (field capacity), `theta_pwp_0` (wilting point)
* vegetation statics: `hveg_rsmin_0`, `lveg_rsmin_0` (min stomatal resistance),
  `hveg_z0m_0`, `lveg_z0m_0`, `hveg_cov_0`, `lveg_cov_0`
* diagnostics v1 predicts: `2t`, `2d`, `skt`, `slhf` (LE), `sshf` (H)
* prognostics: `stl1-4`, `swvl1-4`, `snowc`, `sd`, `rsn`, `asn`, `tsn`, `src`
* carbon: `aco2gpp`, `aco2nee`, `aco2rec`

The stores also carry `statistics_tendencies_*` groups — the precomputed tendency
statistics that v1 uses to normalise increments before the loss.

Other ecLand-derived stores exist under `/lus/h1resw02/project/ai-ml/datasets/`
(O400, N400, N320 at 1-hourly and 6-hourly, various expvers).
