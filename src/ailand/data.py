"""Loading the ecLand Zarr store and building feature/target arrays."""

import numpy as np
import xarray as xr

from . import config


def open_store(path=None, years=None):
    """Open the ecLand Zarr store, optionally restricted to a slice of years."""
    ds = xr.open_zarr(path or config.DATA)
    if years is not None:
        ds = ds.sel(time=slice(*years))
    return ds


def training_arrays(preset=config.DEFAULT_PRESET, years=("2020", "2021"), path=None):
    """Build the stacked (sample, feature) training arrays.

    Features are taken at time ``t`` and targets at ``t+1``. Prognostic targets
    are increments ``state(t+1) - state(t)``; diagnostic targets are absolute
    values at ``t+1``.

    :returns: ``(X, y_prog, y_diag, meta)``
    """
    prog, diag, feat = config.resolve(preset)
    ds = open_store(path, years)

    missing = [v for v in feat + diag if v not in ds]
    if missing:
        raise KeyError(f"variables absent from store: {missing}")

    now = ds.isel(time=slice(0, -1))
    nxt = ds.isel(time=slice(1, None))

    X = (
        now[feat].to_array().astype("float32").stack(z=("x", "time")).transpose()
    )
    # Index the previous state by NAME, not by position. The v0 notebook assumed
    # the targets were the last len(targets) features in order, which breaks
    # silently as soon as either list is edited.
    prog_idx = [feat.index(v) for v in prog]
    y_prog = (
        nxt[prog].to_array().astype("float32").stack(z=("x", "time")).transpose()
        - X[:, prog_idx].values
    )
    y_diag = (
        nxt[diag].to_array().astype("float32").stack(z=("x", "time")).transpose()
        if diag
        else None
    )

    meta = {
        "preset": preset,
        "features": feat,
        "prognostic": prog,
        "diagnostic": diag,
        "prog_idx": prog_idx,
        "years": list(years) if years else None,
    }
    return (
        X.values,
        y_prog.values,
        y_diag.values if y_diag is not None else None,
        meta,
    )


def tendency_scalers(y_prog):
    """Standard deviation of the 6-hourly increments, per prognostic variable.

    This is aiLand v1's "tendency scaler". Increment magnitudes span four orders
    of magnitude in this dataset (stl1 ~2.5 K vs ssro ~1.5e-4 m), so an unweighted
    summed-squared-error loss is driven almost entirely by soil temperature and
    snow, and the remaining variables contribute almost no gradient.
    """
    scale = y_prog.std(axis=0)
    scale[scale == 0] = 1.0
    return scale.astype("float32")


def rollout_inputs(preset=config.DEFAULT_PRESET, point=5, years=None, path=None):
    """Feature matrix and truth for a single grid point, for autoregressive rollout.

    :returns: ``(feats_arr, times, truth, meta)`` where ``feats_arr`` is a
        writable ``(time, feature)`` array whose prognostic columns will be
        overwritten step by step.
    """
    prog, diag, feat = config.resolve(preset)
    ds = open_store(path, years)
    sel = ds.isel(x=point, time=slice(0, -1))
    feats_arr = sel[feat].to_array().values.T.astype("float32").copy()
    truth = sel[prog + diag]
    meta = {
        "preset": preset,
        "features": feat,
        "prognostic": prog,
        "diagnostic": diag,
        "prog_idx": [feat.index(v) for v in prog],
        "point": point,
        "lat": float(sel.lat.values),
        "lon": float(sel.lon.values),
    }
    return feats_arr, sel.time.values, truth, meta


def bounds_arrays(prognostic):
    """Lower/upper bound vectors aligned with ``prognostic``."""
    lo = np.array(
        [config.BOUNDS[v][0] if config.BOUNDS[v][0] is not None else -np.inf
         for v in prognostic],
        dtype="float32",
    )
    hi = np.array(
        [config.BOUNDS[v][1] if config.BOUNDS[v][1] is not None else np.inf
         for v in prognostic],
        dtype="float32",
    )
    return lo, hi
