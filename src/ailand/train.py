"""Train the aiLand v0 XGBoost emulator.

Usage::

    python -m ailand.train --preset v0+snow
    python -m ailand.train --preset v1 --scale-targets --n-estimators 2000
"""

import argparse
import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

from . import config, data


def r2_score_multi(y_pred, y_true):
    """R-squared over all targets flattened together (as in the v0 notebook)."""
    return r2_score(y_pred.flatten(), y_true.flatten())


def build_model(n_estimators, subsample, learning_rate, seed, multi_strategy=None):
    # The v0 notebook passed `objevtive=mean_absolute_error` -- a typo that
    # XGBoost silently accepts as an unused kwarg, so the objective stayed
    # reg:squarederror and the intended MAE was never applied. Set it explicitly.
    kwargs = dict(
        n_estimators=n_estimators,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric=r2_score_multi,
        subsample=subsample,
        random_state=seed,
    )
    if multi_strategy:
        # With the default "one_output_per_tree" each target is fitted by its own
        # trees, so per-target scaling is close to a no-op. "multi_output_tree"
        # shares one tree structure across all targets, which is where the loss
        # really is dominated by the largest-magnitude target and where scaling bites.
        kwargs["multi_strategy"] = multi_strategy
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    return xgb.XGBRegressor(**kwargs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default=config.DEFAULT_PRESET, )
    p.add_argument("--data", default=None, help="path to the ecLand Zarr store")
    p.add_argument("--train-years", nargs=2, default=("2020", "2021"), metavar=("START", "END"))
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--subsample", type=float, default=0.6)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument(
        "--scale-targets",
        action="store_true",
        help="divide prognostic increments by their standard deviation before "
             "fitting (aiLand v1's 'tendency scaler'), so variables with small "
             "increments are not swamped by soil temperature and snow",
    )
    p.add_argument("--profile", default="mock", choices=sorted(config.PROFILES),
                   help="dataset profile: 'mock' or 'o96'")
    p.add_argument("--temporal", action="store_true",
                   help="add time of day, day of year and TOA insolation to the inputs")
    p.add_argument("--geo", action="store_true",
                   help="add cos/sin latitude and longitude (point identifiers here)")
    p.add_argument("--multi-strategy", default=None,
                   choices=["one_output_per_tree", "multi_output_tree"],
                   help="XGBoost multi-output strategy; scaling only matters for "
                        "multi_output_tree, where all targets share a tree structure")
    p.add_argument("--outdir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    X, y_prog, y_diag, meta = data.training_arrays(
        preset=args.preset, years=tuple(args.train_years), path=args.data,
        temporal=args.temporal, geo=args.geo, prof=args.profile,
    )
    print(f"preset {args.preset}: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(meta['prognostic'])} prognostic + {len(meta['diagnostic'])} diagnostic targets")

    # v1 computes the loss only over valid (non-NaN) target elements. XGBoost has
    # no such masking, so drop the affected rows -- in the O96 extract this is a
    # handful of slhf values out of ~10^6.
    bad = np.isnan(y_prog).any(axis=1)
    if y_diag is not None:
        bad |= np.isnan(y_diag).any(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} of {len(bad)} samples with NaN targets "
              f"({bad.mean():.4%})")
        X, y_prog = X[~bad], y_prog[~bad]
        y_diag = y_diag[~bad] if y_diag is not None else None

    scalers = data.tendency_scalers(y_prog)
    # Diagnostics need standardising too, and for the same reason: in the O96 set
    # they span slhf ~1e7 J m-2 down to evaporation ~1e-3 m, so an unstandardised
    # sum-of-squares is entirely the turbulent fluxes and everything else gets
    # essentially no gradient.
    d_mean = d_std = None
    if y_diag is not None and args.scale_targets:
        d_mean = y_diag.mean(0).astype("float32")
        d_std = y_diag.std(0).astype("float32")
        d_std[d_std == 0] = 1.0
    if args.scale_targets:
        print("tendency scalers:",
              ", ".join(f"{v}={s:.4g}" for v, s in zip(meta["prognostic"], scalers)))
        y_fit = y_prog / scalers
    else:
        y_fit = y_prog

    tag = args.preset.replace("+", "_")
    if args.temporal:
        tag += "_temporal"
    if args.geo:
        tag += "_geo"
    if args.scale_targets:
        tag += "_scaled"
    if args.multi_strategy == "multi_output_tree":
        tag += "_mot"
    outdir = config.MODELS / (args.outdir or tag)
    outdir.mkdir(parents=True, exist_ok=True)

    verbose = False if args.quiet else 100
    print("Fitting XGB model for prognostic increments...")
    model = build_model(args.n_estimators, args.subsample, args.learning_rate,
                        args.seed, args.multi_strategy)
    model.fit(X, y_fit, eval_set=[(X, y_fit)], verbose=verbose)
    model.save_model(outdir / "prognostic.json")

    if meta["diagnostic"]:
        print("Fitting XGB model for diagnostic variables...")
        model_diag = build_model(args.n_estimators, args.subsample, args.learning_rate,
                                 args.seed, args.multi_strategy)
        y_diag_fit = (y_diag - d_mean) / d_std if d_mean is not None else y_diag
        model_diag.fit(X, y_diag_fit, eval_set=[(X, y_diag_fit)], verbose=verbose)
        model_diag.save_model(outdir / "diagnostic.json")

    meta = dict(meta)
    meta.update(
        scale_targets=bool(args.scale_targets),
        tendency_scalers=scalers.tolist(),
        diag_mean=d_mean.tolist() if d_mean is not None else None,
        diag_std=d_std.tolist() if d_std is not None else None,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        learning_rate=args.learning_rate,
        seed=args.seed,
        multi_strategy=args.multi_strategy,
        n_samples=int(X.shape[0]),
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {outdir}")
    print("NOTE: eval_set above is the TRAINING set, so those scores are in-sample. "
          "Run `python -m ailand.evaluate` for skill on the held-out year.")
    return outdir


if __name__ == "__main__":
    main()
