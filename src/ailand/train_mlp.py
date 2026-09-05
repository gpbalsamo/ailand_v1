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


def build_windows(ds, feat, prog, diag, steps, max_samples=None, point_block=250,
                  seed=0, quiet=False):
    """Build ``(sample, step, feature)`` rollout windows, a block of points at a time.

    Materialising every point for every time is not possible at the scale the
    paper trains on: 11,538 points x 36,525 timesteps x 44 features is 74 GB for
    the inputs alone. Instead each block of points is loaded, its quota of
    windows is drawn, and the block is released.
    """
    npoint, ntime = ds.sizes["x"], ds.sizes["time"]
    n = ntime - steps
    total = npoint * n
    if max_samples is None or max_samples >= total:
        per_point = n
    else:
        per_point = max(1, int(round(max_samples / npoint)))
    if not quiet:
        print(f"  {per_point * npoint:,} of {total:,} rollout windows "
              f"({per_point} per point x {npoint} points)")

    off = np.arange(steps)
    Xs, Yps, Yds = [], [], []
    for a in range(0, npoint, point_block):
        b = min(a + point_block, npoint)
        sub = ds.isel(x=slice(a, b))
        x = sub[feat].to_array().astype("float32").transpose("x", "time", "variable").values
        yp = sub[prog].to_array().astype("float32").transpose("x", "time", "variable").values
        yd = (sub[diag].to_array().astype("float32")
              .transpose("x", "time", "variable").values if diag else None)
        # Even stride in time keeps full seasonal coverage for every point.
        ti = np.linspace(0, n - 1, per_point).astype(np.int64) if per_point < n \
            else np.arange(n)
        cols = ti[:, None] + off
        for pi in range(x.shape[0]):
            Xs.append(x[pi][cols])
            Yps.append(yp[pi][cols + 1])
            if yd is not None:
                Yds.append(yd[pi][cols + 1])
        del x, yp, yd
    X = np.concatenate(Xs); Yp = np.concatenate(Yps)
    Yd = np.concatenate(Yds) if Yds else None
    return X, Yp, Yd


def cosine_lr(step, total, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(p, 1.0)))


def run_phase(model, norm, meta, ds, steps, epochs, peak_lr, batch_size, device,
              seed, clip=5.0, floor_lr=3e-7, warmup=1000, log_every=10,
              max_samples=None, diag_weight=1.0, var_weights=None,
              point_block=250, val_ds=None, val_max=200000):
    feat, prog, diag = meta["features"], meta["prognostic"], meta["diagnostic"]
    prog_idx = meta["prog_idx"]
    lo, hi = mlp.bounds_tensors(prog, meta.get("profile", "mock"))

    X, Yp, Yd = build_windows(ds, feat, prog, diag, steps, max_samples=max_samples,
                              point_block=point_block, seed=seed)
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
                                     max_samples=val_max, point_block=point_block,
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
    p.add_argument("--point-block", type=int, default=250,
                   help="grid points loaded at once when building windows")
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

    # Statistics come from the single-step arrays, matching the XGBoost path.
    X1, yp1, yd1, _ = data.training_arrays(
        preset=args.preset, years=tuple(args.train_years), path=args.data,
        temporal=args.temporal, geo=args.geo, prof=args.profile,
    )
    norm = mlp.Normaliser(X1, yp1, yd1)
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
                  point_block=args.point_block, val_ds=val_ds, warmup=args.warmup)

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
