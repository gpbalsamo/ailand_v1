"""A small MLP emulator in the shape of aiLand v1.

The XGBoost prototype cannot express two things v1 depends on:

* **differentiability** -- v1 chose an MLP specifically so gradients flow through
  the emulator, enabling gradient-based parameter estimation and data assimilation;
* **multi-step rollout loss** -- training on a single 6-hourly step is why errors
  compound; v1 accumulates loss over an R-step rollout, which requires
  backpropagating through the autoregressive update.

This module follows the v1 architecture described in Raoult et al. (2026, Sect. 2.2):
a shared prognostic backbone of Linear -> LayerNorm -> ReLU blocks projecting to
prognostic increments, plus a diagnostic branch predicting absolute values, with
increments applied residually and physical bounds enforced as post-processing.
It is deliberately small -- this repo trains on 10 grid points, not 171,039.
"""

import numpy as np
import torch
import torch.nn as nn

from . import config


class Block(nn.Module):
    """Linear -> LayerNorm -> ReLU, as in v1."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))


class AiLandMLP(nn.Module):
    """Shared prognostic backbone + diagnostic branch.

    :param n_features: size of the input vector
    :param n_prog: number of prognostic increments to predict
    :param n_diag: number of diagnostic variables to predict
    :param width: hidden width (v1 uses 512)
    :param depth: number of hidden blocks (v1 uses 6)
    :param diag_blocks: blocks in the diagnostic head (v1's base uses 1)
    """

    def __init__(self, n_features, n_prog, n_diag=0, width=256, depth=4, diag_blocks=1):
        super().__init__()
        blocks = [Block(n_features, width)]
        blocks += [Block(width, width) for _ in range(depth - 1)]
        self.backbone = nn.Sequential(*blocks)
        self.prognostic_head = nn.Linear(width, n_prog)

        self.n_diag = n_diag
        if n_diag:
            # The diagnostic branch sees the input alongside the latent state, so
            # diagnostics are re-diagnosed from forcing and the current state.
            diag = [Block(n_features + width, width)]
            diag += [Block(width, width) for _ in range(diag_blocks - 1)]
            self.diagnostic_head = nn.Sequential(*diag, nn.Linear(width, n_diag))

    def forward(self, x):
        """:returns: ``(prognostic_increments, diagnostics_or_None)`` in normalised space."""
        latent = self.backbone(x)
        prog = self.prognostic_head(latent)
        diag = None
        if self.n_diag:
            diag = self.diagnostic_head(torch.cat([x, latent], dim=-1))
        return prog, diag


class Normaliser:
    """Feature-wise standardisation, with tendency scalers for the increments.

    v1 normalises inputs with feature-wise statistics and divides prognostic
    increments by "tendency scalers" -- the standard deviation of the 6-hourly
    increments -- so that variables with very different evolution rates contribute
    comparably to the loss. Increment magnitudes here span four orders of magnitude,
    so without this the loss is almost entirely soil temperature and snow.
    """

    def __init__(self, X, y_prog, y_diag=None):
        # NaN-aware throughout. A single missing value anywhere in a column makes
        # a plain mean/std NaN, which then makes every prediction for that
        # variable NaN -- and because the scorer drops non-finite values, the
        # variable disappears from the results table silently rather than
        # failing. That is exactly what happened to slhf in the O96 extract,
        # which carries a handful of missing turbulent-flux values.
        self.x_mean = np.nanmean(X, axis=0).astype("float32")
        self.x_std = np.nanstd(X, axis=0).astype("float32")
        self.tend = np.nanstd(y_prog, axis=0).astype("float32")
        if y_diag is not None and y_diag.size:
            self.d_mean = np.nanmean(y_diag, axis=0).astype("float32")
            self.d_std = np.nanstd(y_diag, axis=0).astype("float32")
        else:
            self.d_mean = self.d_std = None
        self._sanitise()

    def _sanitise(self):
        """Replace degenerate or non-finite statistics, and say so."""
        for name in ("x_mean", "x_std", "tend", "d_mean", "d_std"):
            a = getattr(self, name)
            if a is None:
                continue
            bad = ~np.isfinite(a)
            if bad.any():
                raise ValueError(
                    f"{name} is non-finite in {int(bad.sum())} position(s); a column "
                    "is entirely missing. Fix the input data rather than masking it."
                )
            if name.endswith("std") or name == "tend":
                a[a == 0] = 1.0

    def to_dict(self):
        d = dict(x_mean=self.x_mean.tolist(), x_std=self.x_std.tolist(),
                 tend=self.tend.tolist())
        if self.d_mean is not None:
            d.update(d_mean=self.d_mean.tolist(), d_std=self.d_std.tolist())
        return d

    @classmethod
    def from_dict(cls, d):
        self = cls.__new__(cls)
        self.x_mean = np.asarray(d["x_mean"], dtype="float32")
        self.x_std = np.asarray(d["x_std"], dtype="float32")
        self.tend = np.asarray(d["tend"], dtype="float32")
        self.d_mean = np.asarray(d["d_mean"], dtype="float32") if "d_mean" in d else None
        self.d_std = np.asarray(d["d_std"], dtype="float32") if "d_std" in d else None
        return self


def rollout_batch(model, norm, x0, forcing, prog_idx, lo, hi, steps):
    """Differentiable R-step autoregressive rollout.

    At each step the prognostic columns of the input vector are replaced by the
    model's own previous prediction while the static and meteorological columns
    are taken from ``forcing``, exactly as in an offline forced run. Gradients
    flow through every step, which is what makes a multi-step loss possible.

    :param x0: ``(batch, n_features)`` initial input vectors, unnormalised
    :param forcing: ``(batch, steps, n_features)`` inputs at each future step,
        whose prognostic columns are ignored and overwritten
    :returns: ``(prog_states, diagnostics)`` each ``(batch, steps, n)``
    """
    x_mean = torch.as_tensor(norm.x_mean, device=x0.device)
    x_std = torch.as_tensor(norm.x_std, device=x0.device)
    tend = torch.as_tensor(norm.tend, device=x0.device)
    lo_t = torch.as_tensor(lo, device=x0.device)
    hi_t = torch.as_tensor(hi, device=x0.device)

    x = x0
    states, diags = [], []
    for t in range(steps):
        prog_n, diag_n = model((x - x_mean) / x_std)
        # Residual formulation: increments are added to the previous state.
        state = x[:, prog_idx] + prog_n * tend
        # Soft-free bounds: clamp is subgradient-safe and matches v1's
        # variable-specific post-processing.
        state = torch.clamp(state, min=lo_t, max=hi_t)
        states.append(state)
        if diag_n is not None:
            diags.append(diag_n)
        if t + 1 < steps:
            nxt = forcing[:, t + 1].clone()
            nxt[:, prog_idx] = state
            x = nxt
    out_diag = torch.stack(diags, 1) if diags else None
    return torch.stack(states, 1), out_diag


def bounds_tensors(prognostic, prof="mock"):
    table = config.profile(prof)["bounds"]
    lo, hi = [], []
    for v in prognostic:
        a, b = table.get(v, (None, None))
        lo.append(-np.inf if a is None else a)
        hi.append(np.inf if b is None else b)
    return np.array(lo, dtype="float32"), np.array(hi, dtype="float32")


class TorchPredictor:
    """Adapter giving a torch model the ``.predict(X) -> ndarray`` interface.

    Lets the MLP reuse :func:`ailand.infer.rollout` unchanged: it returns
    increments in *normalised* space, which the rollout then multiplies by the
    tendency scalers, exactly as for a scaled XGBoost model.
    """

    def __init__(self, model, norm, diagnostic=False, device="cpu"):
        self.model = model.eval()
        self.norm = norm
        self.diagnostic = diagnostic
        self.device = device

    @torch.no_grad()
    def predict(self, X):
        x = torch.as_tensor(np.asarray(X, dtype="float32"), device=self.device)
        x = (x - torch.as_tensor(self.norm.x_mean, device=self.device)) / \
            torch.as_tensor(self.norm.x_std, device=self.device)
        prog, diag = self.model(x)
        if not self.diagnostic:
            return prog.cpu().numpy()
        out = diag * torch.as_tensor(self.norm.d_std, device=self.device) + \
            torch.as_tensor(self.norm.d_mean, device=self.device)
        return out.cpu().numpy()


def load_trained(modeldir, device="cpu"):
    """Load a saved MLP and return ``(prognostic_predictor, diagnostic_predictor, meta)``."""
    import json

    meta = json.loads((modeldir / "meta.json").read_text())
    norm = Normaliser.from_dict(meta["norm"])
    model = AiLandMLP(
        len(meta["features"]), len(meta["prognostic"]), len(meta["diagnostic"]),
        width=meta["width"], depth=meta["depth"],
    )
    model.load_state_dict(torch.load(modeldir / "model.pt", map_location=device))
    model.to(device)
    # The rollout rescales by the tendency scalers, so declare that here.
    meta["scale_targets"] = True
    meta["tendency_scalers"] = norm.tend.tolist()
    prog = TorchPredictor(model, norm, diagnostic=False, device=device)
    diag = TorchPredictor(model, norm, diagnostic=True, device=device) if meta["diagnostic"] else None
    return prog, diag, meta
