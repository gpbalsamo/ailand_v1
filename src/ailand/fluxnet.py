"""FLUXNET Shuttle observations, collocated onto the aiLand grid.

Source: ``/perm/pad/fluxnet-shuttle-ecland`` (775 sites run through ecLand).

aiLand v1 fine-tunes on latent and sensible heat only, because those were the
only observations it had. This collection also carries **soil moisture and soil
temperature** at 608 and 674 sites, at depths spanning all four ecLand layers.
That is what allows the loss to act on the prognostic state directly rather than
only on the diagnostic branch -- see ``docs/STRATEGY.md``.

Conventions, verified against the data:

* tower ``Qle``/``Qh`` are W m-2 **positive upward**; the store's ``slhf``/``sshf``
  are 6-hourly accumulated J m-2 **positive downward**, so
  ``slhf = -Qle * 21600``;
* ``SWC`` is volumetric **percent**, ``swvl`` is m3 m-3, so ``swvl = SWC / 100``;
* ``TS`` is degrees C, ``stl`` is K;
* QC flags follow FLUXNET2015: 0 = measured, 1-3 = gap-filled tiers;
* flux and soil files for the same site carry **different period suffixes**, so
  they must be globbed by site code rather than matched by filename.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/perm/pad/fluxnet-shuttle-ecland"
GROUP = "shuttle-all775-era5"
FLUX_DIR = os.path.join(ROOT, "flux", GROUP)
SOIL_DIR = os.path.join(ROOT, "soil", GROUP)

SECONDS_PER_STEP = 21600.0  # 6 h
KELVIN = 273.15

#: ecLand soil layer boundaries and mid-depths (m).
LAYER_EDGES = [0.07, 0.28, 1.00, 2.89]
LAYER_MID = [0.035, 0.175, 0.64, 1.945]


def inventory():
    """One row per site: code, coordinates, and the flux/soil files that exist."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(FLUX_DIR, "*_Flux.nc"))):
        m = re.match(r"(.+?)_(\d{4})-(\d{4})_FLUXNET2015_Flux\.nc$", os.path.basename(path))
        if not m:
            continue
        rows[m.group(1)] = dict(site=m.group(1), flux=path,
                                flux_y0=int(m.group(2)), flux_y1=int(m.group(3)))
    for path in sorted(glob.glob(os.path.join(SOIL_DIR, "soil_*.nc"))):
        m = re.match(r"soil_(.+?)_(\d{4})-(\d{4})\.nc$", os.path.basename(path))
        if not m or m.group(1) not in rows:
            continue
        rows[m.group(1)].update(soil=path, soil_y0=int(m.group(2)),
                                soil_y1=int(m.group(3)))
    out = []
    for site, r in rows.items():
        with xr.open_dataset(r["flux"]) as f:
            r["lat"] = float(f.latitude.values.squeeze())
            r["lon"] = float(f.longitude.values.squeeze())
            r["igbp"] = (str(f["IGBP_veg_short"].values.astype(str)).strip()
                         if "IGBP_veg_short" in f else "?")
            r["n_halfhour"] = int(f.sizes["time"])
        out.append(r)
    df = pd.DataFrame(out).sort_values("site").reset_index(drop=True)
    df["has_soil"] = df.get("soil", pd.Series(dtype=object)).notna() if "soil" in df else False
    return df


def match_grid(sites, lat, lon, land_idx):
    """Nearest land grid point for each site, by great-circle distance.

    :param sites: inventory DataFrame
    :param lat: grid latitudes (degrees)
    :param lon: grid longitudes (degrees)
    :param land_idx: indices of grid points eligible for matching
    :returns: the DataFrame with ``grid_idx`` and ``grid_dist_km`` added
    """
    glat = np.deg2rad(lat[land_idx])
    glon = np.deg2rad(lon[land_idx])
    out_idx, out_d = [], []
    for slat, slon in zip(sites["lat"].values, sites["lon"].values):
        a, o = np.deg2rad(slat), np.deg2rad(slon)
        # Great-circle distance via the haversine formula.
        d = 2 * 6371.0 * np.arcsin(np.sqrt(
            np.sin((glat - a) / 2) ** 2
            + np.cos(a) * np.cos(glat) * np.sin((glon - o) / 2) ** 2))
        k = int(np.argmin(d))
        out_idx.append(int(land_idx[k]))
        out_d.append(float(d[k]))
    sites = sites.copy()
    sites["grid_idx"] = out_idx
    sites["grid_dist_km"] = out_d
    return sites


def dedup_by_cell(sites, score_col="n_halfhour"):
    """Keep one site per grid cell, the one with the longest record.

    v1 does the same at N320, where it removed 5 co-located sites. On a coarser
    grid many more sites collide, and fitting several conflicting towers to one
    model column is not something the loss can resolve.
    """
    keep = (sites.sort_values(score_col, ascending=False)
                 .drop_duplicates("grid_idx", keep="first"))
    return keep.sort_values("site").reset_index(drop=True)


def utc_offset_hours(tz_name, lon):
    """Standard-time UTC offset for a site.

    FLUXNET timestamps are in **local standard time** (no daylight saving), and
    the files carry the zone in a ``time_zone`` attribute. The model axis is UTC.
    Getting this wrong shifts every observation by the site's offset -- up to
    twelve hours, which for a diurnally-driven flux is worse than no data at all,
    and it is invisible in Europe where the offset is one hour.

    Falls back to ``lon / 15`` when the attribute is missing or "unknown", which
    happens for sites processed with FluxnetLSM's lutz lookup patched out.
    """
    if tz_name and str(tz_name).lower() not in ("unknown", "nan", "none", ""):
        try:
            import datetime as _dt
            from zoneinfo import ZoneInfo
            z = ZoneInfo(str(tz_name))
            # Standard time is the smaller of the two offsets in the year;
            # the larger one is daylight saving, which FLUXNET does not use.
            offs = [z.utcoffset(_dt.datetime(2015, m, 15)).total_seconds() / 3600.0
                    for m in (1, 7)]
            return min(offs)
        except Exception:
            pass
    return round(lon / 15.0)


def _to_utc(times, offset_hours):
    return pd.DatetimeIndex(times) - pd.Timedelta(hours=float(offset_hours))


def _halfhour_to_step(obs_time, obs, model_times, how, window="preceding"):
    """Aggregate a half-hourly series onto the 6-hourly model axis.

    :param how: ``"mean"`` for fluxes (which the store holds as accumulations
        over a window) or ``"nearest"`` for states, which are instantaneous.
    :param window: whether the store's accumulation covers the 6 h *preceding*
        each timestamp or the 6 h *following* it. Verified empirically against
        the store's own fluxes -- see ``verify_window``.
    """
    obs_time = pd.DatetimeIndex(obs_time)
    if how == "nearest":
        s = pd.Series(obs, index=obs_time).reindex(pd.DatetimeIndex(model_times),
                                                   method="nearest",
                                                   tolerance=pd.Timedelta("1h"))
        return s.to_numpy(dtype="float32")
    # Label each half-hourly sample with the model step whose window contains it.
    off = pd.Timedelta("6h") if window == "preceding" else pd.Timedelta(0)
    binned = (pd.Series(obs, index=obs_time)
              .resample("6h", closed="right" if window == "preceding" else "left",
                        label="right" if window == "preceding" else "left")
              .mean())
    return binned.reindex(pd.DatetimeIndex(model_times)).to_numpy(dtype="float32")


def site_observations(row, model_times, qc_max=0, min_depth_match=0.5):
    """Observations for one site, on the model time axis and in model units.

    :param qc_max: keep only samples whose QC flag is <= this. FLUXNET2015
        semantics: 0 = measured, 1-3 = successively coarser gap-filling.
    :param min_depth_match: reject a soil sensor whose depth differs from the
        ecLand layer mid-depth by more than this fraction of the layer thickness.
    :returns: dict of variable name -> array aligned with ``model_times``
    """
    out = {}
    with xr.open_dataset(row["flux"]) as f:
        off = utc_offset_hours(f.time.attrs.get("time_zone"), row["lon"])
        out["utc_offset_hours"] = off
        t = _to_utc(f.time.values, off)
        for cands, dst in ((("Qle_cor", "Qle"), "slhf"), (("Qh_cor", "Qh"), "sshf")):
            # v1 uses the energy-balance-closed variant, in which H and LE are
            # scaled to satisfy H + LE = Rn - G. Not every site has it, so fall
            # back to the raw flux and record which was used.
            src = next((c for c in cands if c in f), None)
            if src is None:
                continue
            out[f"{dst}_source"] = src
            a = f[src].values.squeeze().astype("float64")
            qc = f.get(f"{src}_qc")
            if qc is not None:
                a = np.where(qc.values.squeeze() <= qc_max, a, np.nan)
            # W m-2 positive up  ->  J m-2 per 6 h step, positive down.
            out[dst] = -_halfhour_to_step(t, a, model_times, "mean") * SECONDS_PER_STEP

    if not isinstance(row.get("soil"), str):
        return out

    with xr.open_dataset(row["soil"]) as s:
        t = _to_utc(s.time.values,
                    utc_offset_hours(s.time.attrs.get("time_zone"), row["lon"]))
        for prefix, dst_fmt, conv in (("SWC", "swvl{}", lambda v: v / 100.0),
                                      ("TS", "stl{}", lambda v: v + KELVIN)):
            cands = [(c, float(s[c].attrs["depth_m"])) for c in s.data_vars
                     if re.fullmatch(prefix + r"_\d+", c) and "depth_m" in s[c].attrs]
            if not cands:
                continue
            for layer in range(3):  # aiLand carries three soil layers
                mid = LAYER_MID[layer]
                thick = LAYER_EDGES[layer] - (LAYER_EDGES[layer - 1] if layer else 0.0)
                name, depth = min(cands, key=lambda c: abs(c[1] - mid))
                if abs(depth - mid) > min_depth_match * thick:
                    continue  # no sensor close enough to represent this layer
                a = s[name].values.squeeze().astype("float64")
                qc = s.get(f"{name}_qc")
                if qc is not None:
                    a = np.where(qc.values.squeeze() <= qc_max, a, np.nan)
                key = dst_fmt.format(layer + 1)
                out[key] = conv(_halfhour_to_step(t, a, model_times, "nearest"))
                out[f"{key}_depth"] = np.full(1, depth, dtype="float32")
    return out


def verify_window(row, ds, grid_idx, year="2015"):
    """Which accumulation window matches the store's own fluxes?

    The store's ``slhf`` is an accumulation, but its metadata does not say over
    which 6 h. Correlating the tower series aggregated both ways against the
    model's own value at the collocated cell settles it -- and getting it wrong
    is a three-hour phase error on the variable being fine-tuned.
    """
    sub = ds.isel(x=grid_idx).sel(time=year)
    mt = sub.time.values
    with xr.open_dataset(row["flux"]) as f:
        t = _to_utc(f.time.values,
                    utc_offset_hours(f.time.attrs.get("time_zone"), row["lon"]))
        src = "Qle_cor" if "Qle_cor" in f else "Qle"
        a = f[src].values.squeeze().astype("float64")
    res = {}
    for w in ("preceding", "following"):
        o = -_halfhour_to_step(t, a, mt, "mean", window=w) * SECONDS_PER_STEP
        m = sub["slhf"].values
        ok = np.isfinite(o) & np.isfinite(m)
        res[w] = float(np.corrcoef(o[ok], m[ok])[0, 1]) if ok.sum() > 100 else np.nan
    return res


OBS_VARS = ["slhf", "sshf", "swvl1", "swvl2", "swvl3", "stl1", "stl2", "stl3"]


def split_sites(sites, frac_val=0.2, seed=0, stratify="igbp"):
    """Train/validation site split, stratified by vegetation class.

    v1 splits 199/42 stratified by IGBP so each land-cover class keeps its share
    in both subsets. The split is over *sites*, not time: the point is to test
    whether observational fine-tuning transfers to towers the model never saw.
    """
    rng = np.random.default_rng(seed)
    val = []
    for _, grp in sites.groupby(stratify):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        val.extend(idx[:max(1, int(round(len(idx) * frac_val)))])
    sites = sites.copy()
    sites["split"] = np.where(sites.index.isin(val), "val", "train")
    return sites


def build(store, out, sites=None, years=("1998", "2022"), qc_max=0,
          max_dist_km=90.0, verbose=True):
    """Assemble the fine-tuning dataset: model inputs plus tower observations.

    Model inputs come from the collocated grid cell, exactly as in pretraining,
    so the feature space and normalisation carry over unchanged. Observations are
    added as ``obs_*`` variables on the same time axis, NaN wherever the tower
    has nothing to say -- which is most of it, and which the masked loss handles.
    """
    ds = xr.open_zarr(store).sel(time=slice(*years))
    if sites is None:
        sites = inventory()
        sites = match_grid(sites, ds.lat.values, ds.lon.values,
                           np.arange(ds.sizes["x"]))
        sites = sites[sites.grid_dist_km <= max_dist_km]
        sites = dedup_by_cell(sites)
        sites = split_sites(sites)

    idx = sites.grid_idx.to_numpy()
    sub = ds.isel(x=idx)
    times = sub.time.values

    obs = {v: np.full((len(times), len(sites)), np.nan, dtype="float32")
           for v in OBS_VARS}
    offsets, sources = [], []
    for j, (_, row) in enumerate(sites.iterrows()):
        try:
            o = site_observations(row, times, qc_max=qc_max)
        except Exception as e:                      # one bad file must not stop the build
            if verbose:
                print(f"  ! {row.site}: {type(e).__name__}: {e}")
            offsets.append(np.nan); sources.append("error")
            continue
        offsets.append(o.get("utc_offset_hours", np.nan))
        sources.append(o.get("slhf_source", ""))
        for v in OBS_VARS:
            if v in o:
                obs[v][:, j] = o[v]
        if verbose and (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(sites)} sites", flush=True)

    out_ds = sub.assign({f"obs_{v}": (("time", "x"), obs[v]) for v in OBS_VARS})
    out_ds = out_ds.assign_coords(
        site=("x", sites.site.to_numpy()),
        site_lat=("x", sites.lat.to_numpy().astype("float32")),
        site_lon=("x", sites.lon.to_numpy().astype("float32")),
        igbp=("x", sites.igbp.to_numpy()),
        split=("x", sites.split.to_numpy()),
        grid_dist_km=("x", sites.grid_dist_km.to_numpy().astype("float32")),
        utc_offset_hours=("x", np.array(offsets, dtype="float32")),
        flux_source=("x", np.array(sources, dtype=object)),
    )
    out_ds.attrs.update(
        source=ROOT, group=GROUP, qc_max=qc_max,
        note="tower observations converted from local standard time to UTC; "
             "fluxes are 6-hourly means over the PRECEDING window, sign-flipped "
             "to the store's downward-positive convention",
    )
    out_ds = out_ds.chunk({"time": 2000, "x": -1})
    out_ds.to_zarr(out, mode="w", consolidated=True)
    if verbose:
        n = {v: int(np.isfinite(obs[v]).any(axis=0).sum()) for v in OBS_VARS}
        print(f"Wrote {out}: {len(times)} times x {len(sites)} sites")
        print("  sites with any valid observation:", n)
    return out_ds
