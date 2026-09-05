"""Tests for the feature/target construction and rollout mechanics.

The upstream ec-land-db tests cover only the GRIB->Zarr ingest side; nothing
tested the ML path, which is where the v0 bugs were.
"""

import numpy as np
import pytest

from ailand import config, data, infer, train


@pytest.mark.parametrize("preset", sorted(config.PRESETS))
def test_presets_resolve(preset):
    prog, diag, feat = config.resolve(preset)
    assert prog, "every preset must have prognostic variables"
    assert set(prog).issubset(feat), "prognostic state must be part of the input vector"
    assert not set(diag) & set(feat), (
        "diagnostic variables must NOT be inputs: feeding them back from the "
        "truth data would leak information into the rollout"
    )
    assert set(prog).issubset(config.BOUNDS), "every prognostic needs physical bounds"


def test_snowc_bound_is_percent():
    # snowc is 0-100 in this store, not a fraction. A [0, 1] bound silently
    # destroys the entire snow signal.
    assert config.BOUNDS["snowc"] == (0.0, 100.0)
    ds = data.open_store()
    assert float(ds["snowc"].max().compute()) > 1.0


def test_targets_are_increments():
    X, y_prog, _y_diag, meta = data.training_arrays(preset="v1", years=("2020", "2020"))
    assert X.shape[0] == y_prog.shape[0]
    assert X.shape[1] == len(meta["features"])
    assert y_prog.shape[1] == len(meta["prognostic"])
    # Increments must be far smaller than the absolute state they came from.
    state = X[:, meta["prog_idx"]]
    assert np.abs(y_prog).mean() < np.abs(state).mean()


def test_prognostic_indices_are_name_based():
    # The v0 notebook assumed the targets were the last N features in order.
    # Reordering must not change which columns are updated.
    prog, _diag, feat = config.resolve("v1")
    idx = [feat.index(v) for v in prog]
    assert [feat[i] for i in idx] == prog


def test_bounds_arrays_align():
    prog, _, _ = config.resolve("v0+snow")
    lo, hi = data.bounds_arrays(prog)
    assert lo.shape == hi.shape == (len(prog),)
    assert np.all(lo <= hi)
    # Soil temperature must not be clipped at zero, as v0 did to every variable.
    assert lo[prog.index("stl1")] == -np.inf


def test_rollout_respects_bounds_and_shape():
    prog, diag, feat = config.resolve("v1")
    feats_arr, times, truth, meta = data.rollout_inputs(preset="v1", point=0)
    meta = {**meta, "tendency_scalers": None, "scale_targets": False}
    n = len(feats_arr)

    class ConstantModel:
        """Always predicts a large positive increment, to drive into the bounds."""

        def predict(self, x):
            return np.full((len(x), len(prog)), 1e6, dtype="float32")

    out, diag_arr = infer.rollout(ConstantModel(), None, meta, feats_arr, verbose=False)
    assert out.shape[0] == n
    lo, hi = data.bounds_arrays(prog)
    state = out[1:, meta["prog_idx"]]
    assert np.all(state <= hi + 1e-3), "upper bounds not enforced"
    assert np.isfinite(state[:, prog.index("snowc")]).all()
    assert state[:, prog.index("snowc")].max() <= 100.0


def test_objective_is_set_explicitly():
    # v0 passed `objevtive=` (a typo), which XGBoost silently accepted as an
    # unused kwarg, leaving the objective at the default.
    model = train.build_model(n_estimators=2, subsample=0.6, learning_rate=None, seed=0)
    assert model.get_xgb_params()["objective"] == "reg:squarederror"
    assert "objevtive" not in model.get_params()


def test_tendency_scalers_span_orders_of_magnitude():
    _X, y_prog, _y, meta = data.training_arrays(preset="v0+snow", years=("2020", "2020"))
    s = data.tendency_scalers(y_prog)
    assert np.all(s > 0)
    # This spread is the reason an unscaled summed-squared-error loss is
    # dominated by soil temperature and snow.
    assert s.max() / s.min() > 1e3


def test_temporal_forcings_are_physical():
    ds = data.open_store(temporal=True, years=("2020", "2020"))
    for name in config.TEMPORAL:
        assert ds[name].dims == ds["stl1"].dims, f"{name} must be a (time, x) field"
    ins = ds["insolation"].values
    assert ins.min() == 0.0, "insolation must be zero at night"
    # Peak TOA insolation at ~51.6 N is well under the solar constant.
    assert 900 < ins.max() < 1361
    for trig in ("cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time"):
        v = ds[trig].values
        assert -1.0001 <= v.min() and v.max() <= 1.0001


def test_runoff_is_diagnostic_not_prognostic():
    prog, diag, feat = config.resolve("v1+runoff")
    assert "sro" in diag and "ssro" in diag
    assert "sro" not in prog and "sro" not in feat, (
        "runoff is a flux, not a state: carrying it in the input vector propagates "
        "its noise into the soil column"
    )
    lo, hi = data.bounds_arrays(diag, config.DIAG_BOUNDS)
    assert lo[diag.index("sro")] == 0.0


def test_mlp_rollout_is_differentiable():
    import torch
    from ailand import mlp

    prog, diag, feat = config.resolve("v1", temporal=True)
    model = mlp.AiLandMLP(len(feat), len(prog), len(diag), width=32, depth=2)
    X, y_prog, y_diag, meta = data.training_arrays(
        preset="v1", years=("2020", "2020"), temporal=True
    )
    norm = mlp.Normaliser(X, y_prog, y_diag)
    lo, hi = mlp.bounds_tensors(prog)

    x = torch.as_tensor(X[:4], dtype=torch.float32)
    forcing = x.unsqueeze(1).repeat(1, 3, 1)
    states, diags = mlp.rollout_batch(model, norm, x, forcing, meta["prog_idx"],
                                      lo, hi, steps=3)
    assert states.shape == (4, 3, len(prog))
    # The whole point of the MLP: gradients flow back through every rollout step.
    states.sum().backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_normaliser_is_nan_aware():
    """A handful of missing values must not silently delete a whole variable.

    In the O96 extract, 114 NaNs out of 33.7 million in `slhf` made a plain
    mean/std NaN, which made every slhf prediction NaN, which made the scorer
    drop the variable from the results table without any error.
    """
    import numpy as np
    from ailand import mlp

    X = np.random.rand(100, 5).astype("float32")
    y_prog = np.random.rand(100, 3).astype("float32")
    y_diag = np.random.rand(100, 2).astype("float32")
    y_diag[7, 1] = np.nan  # one missing value in one column

    norm = mlp.Normaliser(X, y_prog, y_diag)
    assert np.isfinite(norm.d_mean).all(), "one NaN must not poison the column mean"
    assert np.isfinite(norm.d_std).all()
    assert np.isfinite(norm.tend).all()

    # A column that is entirely missing is a data problem and must be loud.
    y_diag[:, 0] = np.nan
    try:
        mlp.Normaliser(X, y_prog, y_diag)
    except ValueError as e:
        assert "non-finite" in str(e)
    else:
        raise AssertionError("an all-missing column must raise, not pass silently")


def test_batched_rollout_matches_single_point():
    """rollout_points over N points must equal N separate rollout() calls."""
    import numpy as np
    from ailand import infer

    prog, diag, feat = config.resolve("v1")
    meta_base = dict(prognostic=prog, diagnostic=[], profile="mock",
                     prog_idx=[feat.index(v) for v in prog],
                     scale_targets=False, tendency_scalers=None)

    class Toy:
        def predict(self, x):
            return (x[:, :len(prog)] * 0.0 + 1e-4).astype("float32")

    rng = np.random.default_rng(0)
    feats = rng.random((3, 12, len(feat))).astype("float32")

    batched, _ = infer.rollout_points(Toy(), None, dict(meta_base), feats.copy(),
                                      verbose=False)
    for i in range(3):
        single, _ = infer.rollout(Toy(), None, dict(meta_base), feats[i].copy(),
                                  verbose=False)
        np.testing.assert_allclose(batched[i], single, rtol=1e-6)
