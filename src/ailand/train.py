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


def build_model(n_estimators, subsample, learning_rate, seed):
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
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    return xgb.XGBRegressor(**kwargs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default=config.DEFAULT_PRESET, choices=sorted(config.PRESETS))
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
    p.add_argument("--outdir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    X, y_prog, y_diag, meta = data.training_arrays(
        preset=args.preset, years=tuple(args.train_years), path=args.data
    )
    print(f"preset {args.preset}: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(meta['prognostic'])} prognostic + {len(meta['diagnostic'])} diagnostic targets")

    scalers = data.tendency_scalers(y_prog)
    if args.scale_targets:
        print("tendency scalers:",
              ", ".join(f"{v}={s:.4g}" for v, s in zip(meta["prognostic"], scalers)))
        y_fit = y_prog / scalers
    else:
        y_fit = y_prog

    outdir = config.MODELS / (args.outdir or args.preset.replace("+", "_"))
    outdir.mkdir(parents=True, exist_ok=True)

    verbose = False if args.quiet else 100
    print("Fitting XGB model for prognostic increments...")
    model = build_model(args.n_estimators, args.subsample, args.learning_rate, args.seed)
    model.fit(X, y_fit, eval_set=[(X, y_fit)], verbose=verbose)
    model.save_model(outdir / "prognostic.json")

    if meta["diagnostic"]:
        print("Fitting XGB model for diagnostic variables...")
        model_diag = build_model(args.n_estimators, args.subsample, args.learning_rate, args.seed)
        model_diag.fit(X, y_diag, eval_set=[(X, y_diag)], verbose=verbose)
        model_diag.save_model(outdir / "diagnostic.json")

    meta = dict(meta)
    meta.update(
        scale_targets=bool(args.scale_targets),
        tendency_scalers=scalers.tolist(),
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        learning_rate=args.learning_rate,
        seed=args.seed,
        n_samples=int(X.shape[0]),
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {outdir}")
    print("NOTE: eval_set above is the TRAINING set, so those scores are in-sample. "
          "Run `python -m ailand.evaluate` for skill on the held-out year.")
    return outdir


if __name__ == "__main__":
    main()
