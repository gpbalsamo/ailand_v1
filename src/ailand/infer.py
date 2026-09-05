"""Run the trained emulator autoregressively ("ai-land" forecast).

The prognostic state is stepped forward by adding the predicted 6-hourly
increment and applying physical bounds; meteorological forcing and static fields
are read from the store at each step, so this is a forced offline run in exactly
the sense ecLand is run offline.

Usage::

    python -m ailand.infer --preset v0+snow --point 5 --out models/v0_snow/rollout.nc
"""

import argparse
import json

import numpy as np
import xarray as xr
import xgboost as xgb

from . import config, data


def load(modeldir):
    modeldir = config.MODELS / modeldir if not str(modeldir).startswith("/") else modeldir
    meta = json.loads((modeldir / "meta.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(modeldir / "prognostic.json")
    model_diag = None
    if meta["diagnostic"]:
        model_diag = xgb.XGBRegressor()
        model_diag.load_model(modeldir / "diagnostic.json")
    return model, model_diag, meta


def rollout(model, model_diag, meta, feats_arr, batch=None, verbose=True):
    """Step the prognostic state forward over the whole time axis, in place.

    :returns: ``(feats_arr, diag_arr)``
    """
    prog_idx = meta["prog_idx"]
    lo, hi = data.bounds_arrays(meta["prognostic"])
    scalers = np.asarray(meta.get("tendency_scalers"), dtype="float32")
    rescale = scalers if meta.get("scale_targets") else None

    n = len(feats_arr)
    diag_arr = (
        np.full((n, len(meta["diagnostic"])), np.nan, dtype="float32")
        if meta["diagnostic"]
        else np.empty((n, 0), dtype="float32")
    )

    for t in range(n - 1):
        if verbose and t % 1000 == 0:
            print(f"on step {t}...")
        pred = model.predict(feats_arr[[t]])[0]
        if rescale is not None:
            pred = pred * rescale
        if model_diag is not None:
            diag_arr[t] = model_diag.predict(feats_arr[[t]])[0]
        feats_arr[t + 1, prog_idx] = np.clip(feats_arr[t, prog_idx] + pred, lo, hi)
    if model_diag is not None:
        diag_arr[-1] = model_diag.predict(feats_arr[[-1]])[0]
    return feats_arr, diag_arr


def to_dataset(feats_arr, diag_arr, times, meta):
    """Package the rollout as an xarray Dataset."""
    feat = meta["features"]
    out = {v: ("time", feats_arr[:, feat.index(v)]) for v in meta["prognostic"]}
    out.update({v: ("time", diag_arr[:, i]) for i, v in enumerate(meta["diagnostic"])})
    ds = xr.Dataset(out, coords={"time": times})
    ds.attrs.update(
        preset=meta["preset"],
        point=meta.get("point", -1),
        lat=meta.get("lat", np.nan),
        lon=meta.get("lon", np.nan),
        description="ai-land v0 autoregressive rollout forced by ecLand meteorology",
    )
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default=config.DEFAULT_PRESET, choices=sorted(config.PRESETS))
    p.add_argument("--modeldir", default=None, help="defaults to models/<preset>")
    p.add_argument("--data", default=None)
    p.add_argument("--point", type=int, default=5, help="grid point index to run at")
    p.add_argument("--out", default=None, help="netCDF path for the rollout")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    modeldir = args.modeldir or args.preset.replace("+", "_")
    model, model_diag, mmeta = load(modeldir)

    feats_arr, times, _truth, rmeta = data.rollout_inputs(
        preset=args.preset, point=args.point, path=args.data
    )
    meta = {**mmeta, **rmeta}
    if mmeta["features"] != rmeta["features"]:
        raise ValueError("model was trained on a different feature set than requested")

    feats_arr, diag_arr = rollout(model, model_diag, meta, feats_arr, verbose=not args.quiet)
    ds = to_dataset(feats_arr, diag_arr, times, meta)

    out = args.out or (config.MODELS / modeldir / f"rollout_x{args.point}.nc")
    ds.to_netcdf(out)
    print(f"Wrote {out}  ({ds.sizes['time']} steps, {len(ds.data_vars)} variables)")
    return ds


if __name__ == "__main__":
    main()
