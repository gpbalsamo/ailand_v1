"""Loading the ecLand Zarr store and building feature/target arrays."""

import numpy as np
import xarray as xr

from . import config

SOLAR_CONSTANT = 1361.0  # W m-2


def _solar_terms(doy):
    """Spencer (1971) Fourier fits for solar declination and the equation of time.

    :param doy: day of year as a float (1-based, including the fraction of day)
    :returns: ``(declination_rad, eqtime_minutes, earth_sun_factor)``
    """
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    # Earth-Sun distance correction.
    e0 = 1.000110 + 0.034221 * np.cos(gamma) + 0.001280 * np.sin(gamma)
    return decl, eqtime, e0


def add_temporal(ds):
    """Attach the temporal/astronomical forcing fields to a store.

    Adds :data:`config.TEMPORAL` and :data:`config.GEO` as ``(time, x)`` variables,
    derived from the time coordinate and each point's latitude/longitude. The v0
    notebook had no notion of time of day or season at all: seasonality entered
    only indirectly through the meteorological forcing.
    """
    time = ds["time"]
    lat = ds["lat"].astype("float64")
    lon = ds["lon"].astype("float64")

    doy = time.dt.dayofyear.astype("float64")
    hour = time.dt.hour.astype("float64") + time.dt.minute.astype("float64") / 60.0
    doy_frac = doy + hour / 24.0

    # Season: annual cycle, continuous across the year boundary.
    ang = 2.0 * np.pi * (doy_frac - 1.0) / 365.25
    ds = ds.assign(cos_julian_day=np.cos(ang), sin_julian_day=np.sin(ang))

    # Time of day, in *local* solar time, so the diurnal cycle is in phase
    # regardless of longitude.
    local_hour = (hour + lon / 15.0) % 24.0
    ang_day = 2.0 * np.pi * local_hour / 24.0
    ds = ds.assign(cos_local_time=np.cos(ang_day), sin_local_time=np.sin(ang_day))

    # Top-of-atmosphere insolation from the solar zenith angle.
    decl, eqtime, e0 = _solar_terms(doy_frac)
    tst = hour * 60.0 + eqtime + 4.0 * lon           # true solar time, minutes
    ha = np.deg2rad(tst / 4.0 - 180.0)               # hour angle
    latr = np.deg2rad(lat)
    cos_sza = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)
    ds = ds.assign(insolation=(SOLAR_CONSTANT * e0 * cos_sza).clip(min=0.0))

    # Static geographic encodings.
    ds = ds.assign(
        cos_latitude=np.cos(latr),
        sin_latitude=np.sin(latr),
        cos_longitude=np.cos(np.deg2rad(lon)),
        sin_longitude=np.sin(np.deg2rad(lon)),
    )

    # Broadcast anything that came out 1-D onto the full (time, x) grid.
    template = ds[config.STATIC[0]]
    for name in config.TEMPORAL + config.GEO:
        ds[name] = ds[name].broadcast_like(template).transpose(*template.dims)
    return ds


def open_store(path=None, years=None, temporal=False, prof="mock"):
    """Open the ecLand Zarr store, optionally restricted to a slice of years."""
    ds = xr.open_zarr(path or config.DATA)
    # The O96 store already carries the temporal forcings, precomputed.
    if temporal and config.profile(prof)["derive_temporal"]:
        ds = add_temporal(ds)
    if years is not None:
        ds = ds.sel(time=slice(*years))
    return ds


def training_arrays(preset=config.DEFAULT_PRESET, years=("2020", "2021"), path=None,
                    temporal=False, geo=False, prof="mock"):
    """Build the stacked (sample, feature) training arrays.

    Features are taken at time ``t`` and targets at ``t+1``. Prognostic targets
    are increments ``state(t+1) - state(t)``; diagnostic targets are absolute
    values at ``t+1``.

    :returns: ``(X, y_prog, y_diag, meta)``
    """
    prog, diag, feat = config.resolve(preset, temporal=temporal, geo=geo, prof=prof)
    ds = open_store(path, years, temporal=temporal or geo, prof=prof)

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
        "temporal": bool(temporal),
        "geo": bool(geo),
        "profile": prof,
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


def rollout_inputs_multi(preset=config.DEFAULT_PRESET, points=None, years=None,
                         path=None, temporal=False, geo=False, prof="mock"):
    """Feature array and truth for MANY grid points at once.

    :returns: ``(feats, times, truth, meta)`` with ``feats`` shaped
        ``(npoint, time, feature)``.
    """
    prog, diag, feat = config.resolve(preset, temporal=temporal, geo=geo, prof=prof)
    ds = open_store(path, years, temporal=temporal or geo, prof=prof)
    sel = ds.isel(time=slice(0, -1))
    if points is not None:
        sel = sel.isel(x=points)
    feats = (sel[feat].to_array().astype("float32")
             .transpose("x", "time", "variable").values.copy())
    truth = sel[prog + diag]
    meta = {
        "preset": preset, "features": feat, "prognostic": prog, "diagnostic": diag,
        "prog_idx": [feat.index(v) for v in prog],
        "temporal": bool(temporal), "geo": bool(geo), "profile": prof,
        "npoints": feats.shape[0],
    }
    return feats, sel.time.values, truth, meta


def rollout_inputs(preset=config.DEFAULT_PRESET, point=5, years=None, path=None,
                   temporal=False, geo=False, prof="mock"):
    """Feature matrix and truth for a single grid point, for autoregressive rollout.

    :returns: ``(feats_arr, times, truth, meta)`` where ``feats_arr`` is a
        writable ``(time, feature)`` array whose prognostic columns will be
        overwritten step by step.
    """
    prog, diag, feat = config.resolve(preset, temporal=temporal, geo=geo, prof=prof)
    ds = open_store(path, years, temporal=temporal or geo, prof=prof)
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
        "temporal": bool(temporal),
        "geo": bool(geo),
        "profile": prof,
        "lat": float(sel.lat.values),
        "lon": float(sel.lon.values),
    }
    return feats_arr, sel.time.values, truth, meta


def bounds_arrays(names, table=None, prof=None):
    """Lower/upper bound vectors aligned with ``names``.

    Unlisted variables are unbounded, so this works for both the prognostic state
    (:data:`config.BOUNDS`) and the diagnostic outputs (:data:`config.DIAG_BOUNDS`).
    """
    if table is None:
        table = config.profile(prof)["bounds"] if prof else config.BOUNDS
    lo, hi = [], []
    for v in names:
        a, b = table.get(v, (None, None))
        lo.append(-np.inf if a is None else a)
        hi.append(np.inf if b is None else b)
    return np.array(lo, dtype="float32"), np.array(hi, dtype="float32")
