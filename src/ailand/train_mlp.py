"""Train the MLP emulator with a multi-step rollout loss.

Follows aiLand v1's training strategy (Raoult et al. 2026, Sect. 2.2.3), scaled
down to this repo's 10-point mock dataset:

* smooth L1 (Huber, beta=1) loss, accumulated across the rollout and divided by
  the rollout length;
* prognostic increments normalised by tendency scalers before the loss;
* two phases of increasing rollout horizon (v1: R=4 then R=8);
* Adam with a cosine-decayed learning rate and linear warmup, gradient clipping.

Usage::

    python -m ailand.train_mlp --preset v1 --temporal
    python -m ailand.train_mlp --preset v1 --temporal --rollout 4 8 --epochs 60 10
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn

from . import config, data, mlp


def build_windows(ds, feat, prog, diag, steps, max_samples=None, time_block=100,
                  seed=0, quiet=False, n_starts=None, **_):
    """Build ``(sample, step, feature)`` rollout windows, blocked along TIME.

    The store is chunked ``(100, all points)``, so selecting a subset of points
    still reads every point in each chunk -- blocking along the point axis reads
    the whole dataset once per block. Blocking along time instead is
    chunk-aligned: each block reads exactly the chunks it needs, once.

    The output is preallocated rather than concatenated, so peak memory is the
    result itself and one block, not twice the result.
    """
    npoint, ntime = ds.sizes["x"], ds.sizes["time"]
    n = ntime - steps
    total = npoint * n
    rng = np.random.default_rng(seed)

    # Two ways to spend a sample budget: every point at few start times, or a
    # subset of points at many. The first saturates quickly for prognostic
    # variables, which are locally determined; the second gives the diagnostic
    # branch more distinct surface-atmosphere states to learn from.
    if max_samples is None or max_samples >= total:
        starts, ppp = np.arange(n), npoint
    elif n_starts:
        starts = np.unique(np.linspace(0, n - 1, min(n_starts, n)).astype(np.int64))
        ppp = max(1, min(npoint, max_samples // len(starts)))
    else:
        starts = np.unique(np.linspace(0, n - 1, max(1, max_samples // npoint))
                           .astype(np.int64))
        ppp = npoint

    nsample = len(starts) * ppp
    if not quiet:
        print(f"  {nsample:,} of {total:,} rollout windows "
              f"({len(starts):,} start times x {ppp:,} points)", flush=True)

    X = np.empty((nsample, steps, len(feat)), dtype="float32")
    Yp = np.empty((nsample, steps, len(prog)), dtype="float32")
    Yd = np.empty((nsample, steps, len(diag)), dtype="float32") if diag else None
    full = ppp >= npoint

    off = np.arange(steps)
    w = 0
    for a in range(0, n, time_block):
        b = min(a + time_block, n)
        sel = starts[(starts >= a) & (starts < b)]
        if not len(sel):
            continue
        lo, hi = int(sel.min()), int(sel.max()) + steps + 1
        sub = ds.isel(time=slice(lo, hi))

        def stack(names):
            # One variable at a time. to_array() over ~40 variables builds a
            # single dask graph and materialises a concatenated intermediate,
            # which is what makes the peak several times the result.
            out = np.empty((hi - lo, ds.sizes["x"], len(names)), dtype="float32")
            for j, v in enumerate(names):
                out[:, :, j] = np.asarray(sub[v].values, dtype="float32")
            return out

        x = stack(feat)
        yp = stack(prog)
        yd = stack(diag) if diag else None
        for t in sel:
            r = int(t) - lo
            # A different random subset per start time, so the union still covers
            # the whole grid while each sample is a distinct place and time.
            pi = slice(None) if full else rng.choice(npoint, ppp, replace=False)
            X[w:w + ppp] = x[r:r + steps][:, pi].transpose(1, 0, 2)
            Yp[w:w + ppp] = yp[r + 1:r + 1 + steps][:, pi].transpose(1, 0, 2)
            if yd is not None:
                Yd[w:w + ppp] = yd[r + 1:r + 1 + steps][:, pi].transpose(1, 0, 2)
            w += ppp
        del x, yp, yd
        if not quiet:
            print(f"    windows {w:,}/{nsample:,}", end="\r", flush=True)
    if not quiet:
        print(" " * 40, end="\r")
    return X, Yp, Yd


def cosine_lr(step, total, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(p, 1.0)))


def run_phase(model, norm, meta, ds, steps, epochs, peak_lr, batch_size, device,
              seed, clip=5.0, floor_lr=3e-7, warmup=1000, log_every=10,
              max_samples=None, diag_weight=1.0, var_weights=None,
              point_block=250, val_ds=None, val_max=200000, n_starts=None):
    feat, prog, diag = meta["features"], meta["prognostic"], meta["diagnostic"]
    prog_idx = meta["prog_idx"]
    lo, hi = mlp.bounds_tensors(prog, meta.get("profile", "mock"))

    X, Yp, Yd = build_windows(ds, feat, prog, diag, steps, max_samples=max_samples,
                              time_block=point_block, seed=seed, n_starts=n_starts)
    # Keep the window set in host memory and move only the batch. Putting all of
    # it on the device caps the usable dataset at GPU memory (40 GB on an A100),
    # which is far below what a faithful reproduction needs.
    X = torch.as_tensor(X)
    Yp = torch.as_tensor(Yp)
    Yd = torch.as_tensor(Yd) if Yd is not None else None

    tend = torch.as_tensor(norm.tend, device=device)
    d_mean = torch.as_tensor(norm.d_mean, device=device) if norm.d_mean is not None else None
    d_std = torch.as_tensor(norm.d_std, device=device) if norm.d_std is not None else None

    vw_t = (torch.as_tensor(var_weights, device=device)
            if var_weights is not None and not np.allclose(var_weights, 1.0) else None)
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    lossfn = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    g = torch.Generator(device="cpu").manual_seed(seed)

    # v1 reserves 2022 for validation and selects the best checkpoint on
    # validation loss. Taking the final weights instead is a real omission:
    # nothing detects a phase that overfits or diverges late.
    VX = VYp = VYd = None
    if val_ds is not None:
        VX, VYp, VYd = build_windows(val_ds, feat, prog, diag, steps,
                                     max_samples=val_max, time_block=point_block,
                                     quiet=True)
        VX = torch.as_tensor(VX)
        VYp = torch.as_tensor(VYp)
        VYd = torch.as_tensor(VYd) if VYd is not None else None
        print(f"  validation: {len(VX):,} windows")

    def loss_on(xb, ypb, ydb):
        states, diags = mlp.rollout_batch(model, norm, xb[:, 0], xb, prog_idx,
                                          lo, hi, steps)
        res = (states - ypb) / tend
        m = torch.isfinite(res)
        L = lossfn(torch.where(m, res, torch.zeros_like(res)), torch.zeros_like(res))
        if ydb is not None:
            dres = diags - (ydb - d_mean) / d_std
            if vw_t is not None:
                dres = dres * vw_t
            dm = torch.isfinite(dres)
            L = L + diag_weight * lossfn(
                torch.where(dm, dres, torch.zeros_like(dres)), torch.zeros_like(dres))
        return L

    @torch.no_grad()
    def validate():
        model.eval()
        tot = k = 0.0
        for b in range(0, len(VX), batch_size):
            sl = slice(b, b + batch_size)
            tot += float(loss_on(
                VX[sl].to(device), VYp[sl].to(device),
                VYd[sl].to(device) if VYd is not None else None))
            k += 1
        model.train()
        return tot / max(k, 1)

    best = (float("inf"), None)
    nsample = len(X)
    nbatch = max(1, nsample // batch_size)
    total = epochs * nbatch
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(nsample, generator=g)
        running = 0.0
        for b in range(nbatch):
            sel = perm[b * batch_size:(b + 1) * batch_size]
            xb = X[sel].to(device, non_blocking=True)
            ypb = Yp[sel].to(device, non_blocking=True)
            ydb = Yd[sel].to(device, non_blocking=True) if Yd is not None else None

            for grp in opt.param_groups:
                grp["lr"] = cosine_lr(step, total, peak_lr, floor_lr, warmup)

            states, diags = mlp.rollout_batch(
                model, norm, xb[:, 0], xb, prog_idx, lo, hi, steps
            )
            # Compare in normalised increment space: (predicted - truth) / tendency
            # scaler, so variables with tiny increments are not swamped.
            # v1 computes the loss only over valid (non-NaN) target elements,
            # which is what lets it train on sparse observational data too.
            res = (states - ypb) / tend
            m = torch.isfinite(res)
            loss = lossfn(torch.where(m, res, torch.zeros_like(res)),
                          torch.zeros_like(res))
            if ydb is not None:
                dres = diags - (ydb - d_mean) / d_std
                if vw_t is not None:
                    dres = dres * vw_t
                dm = torch.isfinite(dres)
                loss = loss + diag_weight * lossfn(
                    torch.where(dm, dres, torch.zeros_like(dres)),
                    torch.zeros_like(dres))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running += loss.item()
            step += 1
        vmsg = ""
        if VX is not None:
            vl = validate()
            if vl < best[0]:
                best = (vl, {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()})
                vmsg = f"  val {vl:.5f} *"
            else:
                vmsg = f"  val {vl:.5f}"
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  R={steps} epoch {epoch:3d}/{epochs}  loss {running / nbatch:.5f}"
                  f"  lr {opt.param_groups[0]['lr']:.2e}{vmsg}", flush=True)
    if best[1] is not None:
        print(f"  restoring best checkpoint (val {best[0]:.5f})")
        model.load_state_dict(best[1])
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="v1", )
    p.add_argument("--data", default=None)
    p.add_argument("--train-years", nargs=2, default=("2020", "2021"))
    p.add_argument("--profile", default="mock", choices=sorted(config.PROFILES),
                   help="dataset profile: 'mock' or 'o96'")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--geo", action="store_true")
    p.add_argument("--width", type=int, default=256, help="v1 uses 512")
    p.add_argument("--depth", type=int, default=4, help="v1 uses 6")
    p.add_argument("--diag-blocks", type=int, default=1,
                   help="blocks in the diagnostic head (v1's base uses 1; its S2 "
                        "fine-tuning strategy extends it to 3)")
    p.add_argument("--diag-weight", type=float, default=1.0,
                   help="global weight on the diagnostic loss term")
    p.add_argument("--var-weights", default=None,
                   help="per-diagnostic weights, e.g. 'slhf=3,sshf=3,e=3'. v1 allows "
                        "per-variable weighting but sets them all to 1.")
    p.add_argument("--rollout", type=int, nargs="+", default=[4, 8],
                   help="rollout length per phase (v1: 4 then 8)")
    p.add_argument("--epochs", type=int, nargs="+", default=[60, 10])
    p.add_argument("--lr", type=float, nargs="+", default=[5e-4, 3e-5])
    p.add_argument("--batch-size", type=int, default=512,
                   help="rollout windows per gradient update. v1's effective batch "
                        "is 4 windows each spanning ALL land points, i.e. 4 x 11,538 "
                        "= 46,152 point-windows at O96")
    p.add_argument("--warmup", type=int, default=1000, help="v1 uses 1000 steps")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-years", nargs=2, default=("2022", "2022"),
                   help="years held out for validation and best-checkpoint selection "
                        "(v1 reserves 2022)")
    p.add_argument("--no-validation", action="store_true")
    p.add_argument("--point-block", type=int, default=100,
                   help="timesteps loaded at once when building windows. The store "
                        "is chunked (100, all points), so blocking on time is "
                        "chunk-aligned; blocking on points would read every point "
                        "in every chunk regardless.")
    p.add_argument("--start-times", type=int, default=None,
                   help="spread the sample budget over this many distinct start "
                        "times, taking a random point subset at each, instead of "
                        "every point at few start times")
    p.add_argument("--stat-samples", type=int, default=2000000,
                   help="samples used to estimate normalisation statistics")
    p.add_argument("--val-max", type=int, default=200000,
                   help="validation windows used for checkpoint selection")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap the number of rollout windows per phase")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prog, diag, feat = config.resolve(args.preset, temporal=args.temporal, geo=args.geo, prof=args.profile)
    ds = data.open_store(args.data, tuple(args.train_years),
                         temporal=args.temporal or args.geo, prof=args.profile)
    val_ds = None if args.no_validation else data.open_store(
        args.data, tuple(args.val_years),
        temporal=args.temporal or args.geo, prof=args.profile)
    vw = np.ones(len(diag), dtype="float32")
    if args.var_weights:
        for item in args.var_weights.split(","):
            k, v = item.split("=")
            if k.strip() not in diag:
                raise KeyError(f"{k.strip()!r} is not a diagnostic of this preset")
            vw[diag.index(k.strip())] = float(v)
        print("diagnostic weights:",
              ", ".join(f"{v}={w:g}" for v, w in zip(diag, vw)))

    meta = {
        "preset": args.preset, "features": feat, "prognostic": prog,
        "diagnostic": diag, "prog_idx": [feat.index(v) for v in prog],
        "temporal": args.temporal, "geo": args.geo, "model": "mlp",
        "profile": args.profile,
        "width": args.width, "depth": args.depth,
        "diag_blocks": args.diag_blocks, "diag_weight": args.diag_weight,
        "var_weights": vw.tolist(),
        "rollout": args.rollout, "epochs": args.epochs, "seed": args.seed,
    }

    # Statistics from a subsample of single-step windows. Materialising the whole
    # training set to compute a mean and a standard deviation is what was
    # exhausting memory: 2,920 steps x 11,538 points through xarray's stack() is
    # several GB before any training starts, and at 22 years it is far worse.
    # A few million samples estimate these statistics perfectly well.
    Xs, Yps, Yds = build_windows(ds, feat, prog, diag, 1,
                                 max_samples=args.stat_samples,
                                 time_block=args.point_block, quiet=True)
    pidx = meta["prog_idx"]
    norm = mlp.Normaliser(Xs[:, 0], Yps[:, 0] - Xs[:, 0][:, pidx],
                          Yds[:, 0] if Yds is not None else None)
    del Xs, Yps, Yds
    print(f"preset {args.preset}: {len(feat)} features, {len(prog)} prognostic + "
          f"{len(diag)} diagnostic, device {args.device}")
    print("tendency scalers: " +
          ", ".join(f"{v}={s:.4g}" for v, s in zip(prog, norm.tend)))

    model = mlp.AiLandMLP(len(feat), len(prog), len(diag), width=args.width,
                          depth=args.depth, diag_blocks=args.diag_blocks).to(args.device)
    print(f"parameters: {sum(x.numel() for x in model.parameters()):,}")

    nphase = len(args.rollout)
    epochs = (args.epochs * nphase)[:nphase]
    lrs = (args.lr * nphase)[:nphase]
    for i, (steps, ep, lr) in enumerate(zip(args.rollout, epochs, lrs), 1):
        print(f"\nPhase {i}: rollout R={steps} ({steps * 6} h), {ep} epochs, peak lr {lr:g}")
        run_phase(model, norm, meta, ds, steps, ep, lr, args.batch_size,
                  args.device, args.seed + i, max_samples=args.max_samples,
                  diag_weight=args.diag_weight, var_weights=vw,
                  point_block=args.point_block, val_ds=val_ds, warmup=args.warmup,
                  val_max=args.val_max, n_starts=args.start_times)

    tag = args.outdir or f"mlp_{args.preset.replace('+', '_')}" + ("_temporal" if args.temporal else "")
    outdir = config.MODELS / tag
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    meta["norm"] = norm.to_dict()
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {outdir}")
    return outdir


if __name__ == "__main__":
    main()
