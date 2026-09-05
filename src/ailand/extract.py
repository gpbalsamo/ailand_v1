"""Extract a subset of an anemoi-datasets store into this repo's Zarr layout.

The aiLand v1 training data on /lus is in anemoi format: a single
``(time, variable, ensemble, gridpoint)`` array with the variable names in the
store attributes. This turns a slice of it into the same ``(time, x)`` named-variable
layout as the mock store, so the rest of the package works unchanged.

It also brings in the variables the mock store lacks -- evaporation ``e``, the
turbulent fluxes ``slhf``/``sshf``, and ``2t``/``2d`` -- which are exactly aiLand
v1's diagnostic outputs.

Usage::

    python -m ailand.extract --npoints 200 --years 2020 2022 --out data/o96_subset.zarr
"""

import argparse

import numpy as np
import xarray as xr
import zarr

from . import config

O96 = ("/lus/h1aiws01/project/ai-ml/datasets/"
       "aifs-rd-an-oper-isc8-mars-o96-1998-2024-6h-v1-ecland-era5met.zarr")

#: Variables to pull. Grouped the way the package uses them.
KEEP_STATIC = [
    "cl", "dl", "cvh_0", "cvl_0", "tvh_0", "tvl_0", "hveg_cov_0", "lveg_cov_0",
    "hveg_rsmin_0", "lveg_rsmin_0", "hveg_z0m_0", "lveg_z0m_0",
    "theta_cap_0", "theta_pwp_0", "sdor_0", "isor_0", "anor_0", "slor_0",
    "z", "lsm_0", "c3_c4_0",
]
KEEP_MET = [
    "t_ml_137", "q_ml_137", "u_ml_137", "v_ml_137",
    "ssrd", "strd", "tp", "sf", "sp", "lai_hv", "lai_lv",
]
KEEP_TEMPORAL = [
    "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
    "insolation", "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
]
KEEP_PROG = ["stl1", "stl2", "stl3", "swvl1", "swvl2", "swvl3", "snowc"]
#: v1's five diagnostics, plus evaporation, runoff and GPP.
KEEP_DIAG = ["2t", "2d", "skt", "slhf", "sshf", "e", "sro", "ssro", "aco2gpp"]

KEEP = KEEP_STATIC + KEEP_MET + KEEP_TEMPORAL + KEEP_PROG + KEEP_DIAG


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=O96)
    p.add_argument("--npoints", type=int, default=200,
                   help="number of land points, sampled evenly over the land mask")
    p.add_argument("--years", nargs=2, default=("2020", "2022"))
    p.add_argument("--out", default=str(config.REPO / "data" / "o96_subset.zarr"))
    p.add_argument("--lsm-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)

    z = zarr.open(args.source, mode="r")
    variables = dict(z.attrs)["variables"]
    dates = np.asarray(z["dates"][:], dtype="datetime64[s]")
    lat = np.asarray(z["latitudes"][:])
    lon = np.asarray(z["longitudes"][:])

    missing = [v for v in KEEP if v not in variables]
    if missing:
        raise KeyError(f"variables absent from source: {missing}")

    t0 = int(np.searchsorted(dates, np.datetime64(f"{args.years[0]}-01-01T00:00:00")))
    t1 = int(np.searchsorted(dates, np.datetime64(f"{int(args.years[1]) + 1}-01-01T00:00:00")))
    print(f"time slice {dates[t0]} .. {dates[t1 - 1]}  ({t1 - t0} steps)")

    # Land mask from one timestep.
    lsm = z["data"][t0, variables.index("lsm_0"), 0, :]
    land = np.flatnonzero(np.isfinite(lsm) & (lsm > args.lsm_threshold))
    print(f"{land.size} land points at lsm > {args.lsm_threshold}")
    # Span the land-point index evenly. NOTE: `land[::step][:npoints]` looks
    # equivalent but TRUNCATES -- it keeps the first npoints of the strided set and
    # drops the tail, which on a north-to-south ordered grid silently deletes the
    # southernmost land. linspace spans the full index range by construction.
    pts = land[np.linspace(0, land.size - 1, args.npoints).astype(np.int64)]
    pts = np.unique(pts)
    print(f"sampling {pts.size} of them, lat {lat[pts].min():.1f} to {lat[pts].max():.1f}")

    idx = [variables.index(v) for v in KEEP]
    out = np.empty((t1 - t0, len(KEEP), pts.size), dtype="float32")
    src = z["data"]
    chunk = 200
    for a in range(t0, t1, chunk):
        b = min(a + chunk, t1)
        block = src[a:b, :, 0, :][:, idx, :][:, :, pts]
        out[a - t0:b - t0] = block
        print(f"  read {b - t0}/{t1 - t0} steps", end="\r", flush=True)
    print()

    ds = xr.Dataset(
        {name: (("time", "x"), out[:, i, :]) for i, name in enumerate(KEEP)},
        coords={
            "time": dates[t0:t1].astype("datetime64[ns]"),
            "x": pts.astype("int32"),
            "lat": ("x", lat[pts].astype("float32")),
            "lon": ("x", lon[pts].astype("float32")),
        },
        attrs={"SOURCE": args.source, "GRID": "O96", "note": "extracted by ailand.extract"},
    )
    ds = ds.chunk({"time": -1, "x": -1})
    ds.to_zarr(args.out, mode="w", consolidated=True)
    nan = {v: float(np.isnan(ds[v].values).mean()) for v in ds.data_vars}
    bad = {k: v for k, v in nan.items() if v > 0}
    print(f"Wrote {args.out}: {ds.sizes['time']} times x {ds.sizes['x']} points, "
          f"{len(ds.data_vars)} variables")
    if bad:
        print("variables containing NaN:", {k: f"{v:.2%}" for k, v in bad.items()})
    return ds


if __name__ == "__main__":
    main()
