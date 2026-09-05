"""Score an ai-land rollout against ecLand and plot the comparison.

Splits the scores at the train/test boundary, because the v0 notebook judged the
emulator by eye from a single figure and never computed a metric on the held-out
year.

Usage::

    python -m ailand.evaluate --preset v0+snow --point 5
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from . import config, data, infer


def score(pred_ds, truth, split="2022-01-01"):
    """RMSE, bias and R2 for each variable, before and after ``split``."""
    is_test = pred_ds.time.values >= np.datetime64(split)
    rows = []
    for name in pred_ds.data_vars:
        ref = truth[name].values
        arr = pred_ds[name].values
        for label, mask in (("train", ~is_test), ("test", is_test)):
            if mask.sum() == 0:
                continue
            e = arr[mask] - ref[mask]
            rows.append(
                dict(
                    variable=name,
                    period=label,
                    rmse=float(np.sqrt(np.mean(e**2))),
                    bias=float(np.mean(e)),
                    r2=float(r2_score(ref[mask], arr[mask])),
                )
            )
    return rows


def score_pooled(preds, truths, split="2022-01-01"):
    """Pool errors across grid points before scoring.

    A single point tells you almost nothing about a globally trained emulator;
    these scores are computed over the concatenated points.
    """
    if len(preds) == 1:
        return score(preds[0], truths[0], split)
    is_test = preds[0].time.values >= np.datetime64(split)
    rows = []
    for name in preds[0].data_vars:
        a = np.concatenate([p[name].values for p in preds])
        r = np.concatenate([t[name].values for t in truths])
        m = np.concatenate([is_test] * len(preds))
        ok = np.isfinite(a) & np.isfinite(r)
        for label, mask in (("train", ~m & ok), ("test", m & ok)):
            if mask.sum() == 0:
                continue
            e = a[mask] - r[mask]
            rows.append(dict(variable=name, period=label,
                             rmse=float(np.sqrt(np.mean(e**2))),
                             bias=float(np.mean(e)),
                             r2=float(r2_score(r[mask], a[mask]))))
    return rows


def print_table(rows, split):
    print(f"\n{'variable':10s} {'period':7s} {'RMSE':>12s} {'bias':>12s} {'R2':>9s}")
    print("-" * 54)
    for r in rows:
        print(f"{r['variable']:10s} {r['period']:7s} "
              f"{r['rmse']:12.5g} {r['bias']:12.5g} {r['r2']:9.3f}")
    test = [r for r in rows if r["period"] == "test"]
    if test:
        bad = [r["variable"] for r in test if r["r2"] < 0]
        print(f"\nHeld-out ({split} onwards): mean R2 = "
              f"{np.mean([r['r2'] for r in test]):.3f}")
        if bad:
            print(f"NEGATIVE R2 (worse than predicting the mean): {', '.join(bad)}")


def plot(pred_ds, truth, meta, split="2022-01-01", path=None):
    names = list(pred_ds.data_vars)
    ncol = 4
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        ax.plot(pred_ds.time, truth[name].values, label="ec-land")
        ax.plot(pred_ds.time, pred_ds[name].values, label="ai-land")
        ax.axvline(np.datetime64(split), color="k", linestyle="--")
        ax.set_xlim(pred_ds.time.values[[0, -1]])
        ax.set_ylabel(config.UNITS.get(name, name))
        ax.set_title(name)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    axes[0].legend()
    fig.suptitle(
        f"ec/ai-land comparison, preset '{meta['preset']}' "
        f"({meta.get('lat', float('nan')): .2f} N, {meta.get('lon', float('nan')): .2f} E)"
    )
    fig.tight_layout()
    path = path or config.FIGURES / f"ec-ai-land_{meta['preset'].replace('+', '_')}.png"
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default=config.DEFAULT_PRESET, )
    p.add_argument("--modeldir", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--profile", default="mock", choices=sorted(config.PROFILES),
                   help="dataset profile: 'mock' or 'o96'")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--geo", action="store_true")
    p.add_argument("--point", type=int, default=5)
    p.add_argument("--npoints", type=int, default=1,
                   help="score over this many grid points (evenly strided) and pool "
                        "the errors; single-point scores are misleading at global scale")
    p.add_argument("--split", default="2022-01-01")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    modeldir = args.modeldir or args.preset.replace("+", "_")
    model, model_diag, mmeta = infer.load(modeldir)

    if args.npoints > 1:
        npt = data.open_store(args.data, prof=args.profile).sizes["x"]
        points = list(range(0, npt, max(1, npt // args.npoints)))[:args.npoints]
    else:
        points = [args.point]

    preds, truths = [], []
    for i, pt in enumerate(points):
        feats_arr, times, truth, rmeta = data.rollout_inputs(
            preset=args.preset, point=pt, path=args.data,
            temporal=args.temporal, geo=args.geo, prof=args.profile,
        )
        meta = {**mmeta, **rmeta}
        feats_arr, diag_arr = infer.rollout(
            model, model_diag, meta, feats_arr,
            verbose=(not args.quiet) and len(points) == 1,
        )
        preds.append(infer.to_dataset(feats_arr, diag_arr, times, meta))
        truths.append(truth)
        if len(points) > 1 and (i + 1) % 10 == 0:
            print(f"  rolled out {i + 1}/{len(points)} points", end="\r", flush=True)
    if len(points) > 1:
        print(f"pooled over {len(points)} grid points" + " " * 20)

    rows = score_pooled(preds, truths, args.split)
    pred = preds[0]
    print_table(rows, args.split)
    if not args.no_plot:
        plot(pred, truth, meta, args.split)
    return rows


if __name__ == "__main__":
    main()
