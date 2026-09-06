"""Fine-tune a pretrained aiLand emulator on FLUXNET-Shuttle observations.

Reproduces aiLand v1's five strategies (Raoult et al. 2026, Table 2) and adds a
sixth that v1 could not run.

v1's S1-S5 are all answers to one question: how far may the prognostic backbone
move when the only observations are of *diagnostic* variables? S1 and S2 freeze
it so the pretrained dynamics survive exactly; S3-S5 let it drift at 1e-7 to
5e-5 and accept the risk of catastrophic forgetting. The paper says plainly that
separating those effects would need an ablation it defers.

That tension exists because the prognostic states were unobserved. The Shuttle
pool carries soil moisture and soil temperature at 154-229 sites per layer, so
the loss can act on the prognostic head directly:

* **S6 (Constrained)** -- unfreeze everything at a normal learning rate, with
  soil observations on the prognostic head and flux observations on the
  diagnostic head. The backbone now has evidence pulling it rather than needing
  protection from the absence of it.

Forgetting is guarded explicitly rather than by a tiny learning rate: an
**anchor** term penalises divergence from the pretrained model's own predictions
on global ecLand samples drawn alongside each observational batch. The model must
satisfy the towers *and* stay close to ecLand everywhere else. Structurally this
is a background term J_b, with the pretrained emulator as background and the
towers as observations.
"""

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from . import config, data, mlp

#: (backbone_lr, diagnostic_lr, epochs, deep_head, description)
STRATEGIES = {
    "S1": dict(backbone_lr=0.0, diag_lr=5e-5, epochs=20, deep_head=False,
               desc="Freeze: backbone and prognostic head frozen"),
    "S2": dict(backbone_lr=0.0, diag_lr=1e-4, epochs=20, deep_head=True,
               desc="Deep: frozen backbone, diagnostic head widened to 3 blocks"),
    "S3": dict(backbone_lr=1e-7, diag_lr=1e-4, epochs=20, deep_head=False,
               desc="DiffLR: full model, differential learning rates"),
    "S4": dict(backbone_lr=1e-6, diag_lr=5e-5, epochs=20, deep_head=False,
               two_phase=True, desc="TwoPhase: head first, then full model"),
    "S5": dict(backbone_lr=5e-5, diag_lr=5e-5, epochs=20, deep_head=False,
               desc="FullLR: full model at one rate"),
    "S6": dict(backbone_lr=5e-5, diag_lr=5e-5, epochs=20, deep_head=False,
               soil_obs=True, anchor=1.0,
               desc="Constrained: soil observations on the prognostic head, "
                    "plus an anchor term against the pretrained model"),
}

OBS_FLUX = ["slhf", "sshf"]
OBS_SOIL = ["swvl1", "swvl2", "swvl3", "stl1", "stl2", "stl3"]


def soil_offsets(path, prog, split=None):
    """Per-site, per-layer offset bringing tower soil observations onto the
    model's own scale.

    A point probe and a 125 km column mean are not the same quantity. Their
    difference is dominated by representativeness and sensor calibration, not by
    model error, so fitting absolute soil moisture imports that difference
    straight into the weights. Standard land-DA practice is to constrain
    *anomalies*; matching the tower's long-term mean to ecLand's own is the
    simplest form of that, and it leaves the temporal variability -- which is
    what the emulator can actually learn from -- untouched.
    """
    ds = xr.open_zarr(path)
    if split is not None:
        ds = ds.isel(x=np.flatnonzero(ds["split"].values == split))
    out = {}
    for v in OBS_SOIL:
        if v not in prog or f"obs_{v}" not in ds:
            continue
        o = ds[f"obs_{v}"].values
        m = ds[v].values
        valid = np.isfinite(o)
        off = np.full(o.shape[1], np.nan, dtype="float32")
        for j in range(o.shape[1]):
            k = valid[:, j]
            if k.sum() >= 200:                     # enough to define a mean
                off[j] = float(np.nanmean(m[k, j]) - np.nanmean(o[k, j]))
        out[v] = off
    return out


def load_windows(path, feat, prog, diag, steps, split, max_samples=None,
                 time_block=2000, quiet=False, soil_offset=None):
    """Rollout windows over the site cells, with observations alongside.

    Returns model inputs, the ecLand targets (which keep the dynamics anchored
    where no tower speaks) and the observational targets, which are NaN almost
    everywhere -- the masked loss is what makes that usable.
    """
    ds = xr.open_zarr(path)
    keep = np.flatnonzero(ds["split"].values == split)
    ds = ds.isel(x=keep)
    npoint, ntime = ds.sizes["x"], ds.sizes["time"]
    n = ntime - steps

    obs_names = [f"obs_{v}" for v in OBS_FLUX + OBS_SOIL]
    starts = np.arange(n)
    if max_samples and npoint * n > max_samples:
        starts = np.unique(np.linspace(0, n - 1, max(1, max_samples // npoint))
                           .astype(np.int64))
    ns = len(starts) * npoint
    if not quiet:
        print(f"  {split}: {ns:,} windows ({len(starts):,} starts x {npoint} sites)",
              flush=True)

    X = np.empty((ns, steps, len(feat)), dtype="float32")
    Yp = np.empty((ns, steps, len(prog)), dtype="float32")
    Yd = np.empty((ns, steps, len(diag)), dtype="float32") if diag else None
    O = np.empty((ns, steps, len(obs_names)), dtype="float32")

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
            out = np.empty((hi - lo, npoint, len(names)), dtype="float32")
            for j, v in enumerate(names):
                out[:, :, j] = np.asarray(sub[v].values, dtype="float32")
            return out

        x, yp = stack(feat), stack(prog)
        yd = stack(diag) if diag else None
        ob = stack(obs_names)
        if soil_offset:
            for v, off in soil_offset.items():
                ob[:, :, obs_names.index(f"obs_{v}")] += off[None, :]
        for t in sel:
            r = int(t) - lo
            X[w:w + npoint] = x[r:r + steps].transpose(1, 0, 2)
            Yp[w:w + npoint] = yp[r + 1:r + 1 + steps].transpose(1, 0, 2)
            if yd is not None:
                Yd[w:w + npoint] = yd[r + 1:r + 1 + steps].transpose(1, 0, 2)
            O[w:w + npoint] = ob[r + 1:r + 1 + steps].transpose(1, 0, 2)
            w += npoint
        del x, yp, yd, ob
    return X, Yp, Yd, O, obs_names


def build_optimizer(model, backbone_lr, diag_lr):
    """Differential learning rates: backbone and prognostic head, versus the
    diagnostic head. A rate of 0 freezes that group, as v1's S1 and S2 do."""
    back = list(model.backbone.parameters()) + list(model.prognostic_head.parameters())
    head = list(model.diagnostic_head.parameters()) if model.n_diag else []
    groups = []
    if backbone_lr > 0:
        groups.append({"params": back, "lr": backbone_lr})
    else:
        for p in back:
            p.requires_grad_(False)
    if head:
        groups.append({"params": head, "lr": diag_lr})
    return torch.optim.Adam(groups)


def widen_head(model, blocks=3, width=None):
    """Extend the diagnostic head, as v1's S2 does (two randomly initialised
    blocks in front of the existing projection)."""
    width = width or model.backbone[-1].fc.out_features
    n_in = model.diagnostic_head[0].fc.in_features
    old = list(model.diagnostic_head)
    new = [mlp.Block(n_in, width)] + [mlp.Block(width, width) for _ in range(blocks - 2)]
    model.diagnostic_head = nn.Sequential(*new, *old[-2:] if len(old) > 1 else old)
    return model


def masked_huber(residual, lossfn):
    """Huber over the finite elements only.

    v1 computes the loss over valid target elements only, which is what lets a
    model train on observations that are absent most of the time. Here most of
    the tensor is NaN: a tower speaks for one cell out of 362, and only when its
    instruments were working.
    """
    m = torch.isfinite(residual)
    if not bool(m.any()):
        return residual.new_zeros(())
    r = torch.where(m, residual, torch.zeros_like(residual))
    # Normalise by the number of valid elements, not the tensor size, so the
    # gradient does not shrink simply because a batch had few observations.
    return lossfn(r, torch.zeros_like(r)) * r.numel() / m.sum()


def finetune(model, base, norm, meta, tensors, strategy, device="cpu",
             batch_size=2048, seed=0, log_every=5, anchor_batch=None):
    """Run one fine-tuning strategy. Returns the model with the best checkpoint."""
    cfg = STRATEGIES[strategy]
    prog, diag, feat = meta["prognostic"], meta["diagnostic"], meta["features"]
    prog_idx = meta["prog_idx"]
    lo, hi = mlp.bounds_tensors(prog, meta.get("profile", "o96"))
    lossfn = nn.SmoothL1Loss(beta=1.0, reduction="mean")

    X, Yp, Yd, O, obs_names = tensors["train"]
    tend = torch.as_tensor(norm.tend, device=device)
    d_mean = torch.as_tensor(norm.d_mean, device=device)
    d_std = torch.as_tensor(norm.d_std, device=device)

    # Column positions of each observed variable within the model's outputs.
    flux_o = [obs_names.index(f"obs_{v}") for v in OBS_FLUX if v in diag]
    flux_d = [diag.index(v) for v in OBS_FLUX if v in diag]
    soil_o = [obs_names.index(f"obs_{v}") for v in OBS_SOIL if v in prog]
    soil_p = [prog.index(v) for v in OBS_SOIL if v in prog]
    use_soil = bool(cfg.get("soil_obs")) and len(soil_p) > 0
    w_anchor = float(cfg.get("anchor", 0.0))

    opt = build_optimizer(model, cfg["backbone_lr"], cfg["diag_lr"])
    g = torch.Generator().manual_seed(seed)
    steps = X.shape[1]

    def losses(xb, ypb, ydb, ob):
        states, diags = mlp.rollout_batch(model, norm, xb[:, 0], xb, prog_idx,
                                          lo, hi, steps)
        parts = {}
        # Observations of the diagnostic fluxes -- v1's fine-tuning target.
        if flux_d:
            parts["obs_flux"] = masked_huber(
                (diags[..., flux_d] - (ob[..., flux_o] - d_mean[flux_d]) / d_std[flux_d]),
                lossfn)
        # Observations of the prognostic soil state -- the new term.
        if use_soil:
            parts["obs_soil"] = masked_huber(
                (states[..., soil_p] - ob[..., soil_o]) / tend[soil_p], lossfn)
        # ecLand targets keep everything the towers do not constrain in place.
        parts["ecland_prog"] = masked_huber((states - ypb) / tend, lossfn)
        if ydb is not None:
            parts["ecland_diag"] = masked_huber(
                diags - (ydb - d_mean) / d_std, lossfn)
        return parts, states

    def anchor_loss(xb):
        """Stay close to the pretrained model on global ecLand samples.

        Replaces "use a 1e-7 learning rate and hope" with an explicit constraint,
        which is both stronger and inspectable.
        """
        with torch.no_grad():
            ref, _ = mlp.rollout_batch(base, norm, xb[:, 0], xb, prog_idx, lo, hi, steps)
        cur, _ = mlp.rollout_batch(model, norm, xb[:, 0], xb, prog_idx, lo, hi, steps)
        return masked_huber((cur - ref) / tend, lossfn)

    best = (float("inf"), None)
    n = len(X)
    nb = max(1, n // batch_size)
    for epoch in range(cfg["epochs"]):
        if cfg.get("two_phase") and epoch == cfg["epochs"] // 2:
            # v1's S4: head first, then unfreeze the whole model at a low rate.
            for p in model.parameters():
                p.requires_grad_(True)
            opt = build_optimizer(model, cfg["backbone_lr"], cfg["backbone_lr"])
            print("  phase 2: full model unfrozen", flush=True)
        perm = torch.randperm(n, generator=g)
        run = 0.0
        for b in range(nb):
            sl = perm[b * batch_size:(b + 1) * batch_size]
            parts, _ = losses(X[sl].to(device), Yp[sl].to(device),
                              Yd[sl].to(device) if Yd is not None else None,
                              O[sl].to(device))
            loss = sum(parts.values())
            if w_anchor and anchor_batch is not None:
                k = torch.randint(0, len(anchor_batch), (min(batch_size, len(anchor_batch)),),
                                  generator=g)
                loss = loss + w_anchor * anchor_loss(anchor_batch[k].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss)
        vl = validate(model, norm, meta, tensors["val"], device, batch_size,
                      lossfn, tend, d_mean, d_std, flux_o, flux_d, soil_o, soil_p,
                      use_soil, prog_idx, lo, hi, steps)
        star = ""
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            star = " *"
        if epoch % log_every == 0 or epoch == cfg["epochs"] - 1:
            print(f"  {strategy} epoch {epoch:3d}/{cfg['epochs']}  "
                  f"train {run / nb:.5f}  val {vl:.5f}{star}", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"  restored best checkpoint (val {best[0]:.5f})", flush=True)
    return model


@torch.no_grad()
def validate(model, norm, meta, tensors, device, batch_size, lossfn, tend,
             d_mean, d_std, flux_o, flux_d, soil_o, soil_p, use_soil,
             prog_idx, lo, hi, steps):
    """Validation loss on held-out *sites*, over observations only.

    Scored against the towers rather than ecLand: the point of fine-tuning is to
    correct the model towards observations, so selecting on an ecLand-based loss
    would reward exactly what we are trying to move away from.
    """
    model.eval()
    X, Yp, Yd, O, _ = tensors
    tot = k = 0.0
    for b in range(0, len(X), batch_size):
        sl = slice(b, b + batch_size)
        states, diags = mlp.rollout_batch(model, norm, X[sl][:, 0].to(device),
                                          X[sl].to(device), prog_idx, lo, hi, steps)
        ob = O[sl].to(device)
        L = states.new_zeros(())
        if flux_d:
            L = L + masked_huber(
                diags[..., flux_d] - (ob[..., flux_o] - d_mean[flux_d]) / d_std[flux_d],
                lossfn)
        if use_soil:
            L = L + masked_huber((states[..., soil_p] - ob[..., soil_o]) / tend[soil_p],
                                 lossfn)
        tot += float(L); k += 1
    model.train()
    return tot / max(k, 1)


W_PER_J = 1.0 / 21600.0  # 6-hourly accumulated J m-2 -> mean W m-2


@torch.no_grad()
def evaluate_sites(model, norm, meta, tensors, sites, device="cpu", batch_size=2048):
    """Score against the towers at held-out sites, in tower units.

    Beyond RMSE this reports the two physical diagnostics v1 uses, because a
    model can improve both flux RMSEs while getting the partition between them
    wrong:

    * **Bowen ratio** H/LE, per site, evaluated only where LE > 5 W m-2 and
      clipped to [-20, 20] as the benchmark scripts in the Shuttle repo do;
    * **energy-balance residual** H + LE against the tower's own H + LE, which
      is closed by construction in the corrected fluxes.
    """
    prog, diag = meta["prognostic"], meta["diagnostic"]
    prog_idx = meta["prog_idx"]
    lo, hi = mlp.bounds_tensors(prog, meta.get("profile", "o96"))
    X, Yp, Yd, O, obs_names = tensors
    model.eval()

    P, D = [], []
    for b in range(0, len(X), batch_size):
        sl = slice(b, b + batch_size)
        s, d = mlp.rollout_batch(model, norm, X[sl][:, 0].to(device),
                                 X[sl].to(device), prog_idx, lo, hi, X.shape[1])
        P.append(s[:, -1].cpu().numpy()); D.append(d[:, -1].cpu().numpy())
    P = np.concatenate(P); D = np.concatenate(D)
    Ob = O[:, -1].numpy()
    model.train()

    rows = {}
    for v in OBS_FLUX:
        if v not in diag:
            continue
        pred = D[:, diag.index(v)] * W_PER_J
        obs = Ob[:, obs_names.index(f"obs_{v}")] * W_PER_J
        m = np.isfinite(obs) & np.isfinite(pred)
        if m.sum() < 100:
            continue
        e = pred[m] - obs[m]
        rows[v] = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))),
                       bias=float(np.mean(e)),
                       r=float(np.corrcoef(pred[m], obs[m])[0, 1]))
    for v in OBS_SOIL:
        if v not in prog:
            continue
        pred = P[:, prog.index(v)]
        obs = Ob[:, obs_names.index(f"obs_{v}")]
        m = np.isfinite(obs) & np.isfinite(pred)
        if m.sum() < 100:
            continue
        e = pred[m] - obs[m]
        rows[v] = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))),
                       bias=float(np.mean(e)),
                       r=float(np.corrcoef(pred[m], obs[m])[0, 1]))

    # Bowen ratio and energy-balance residual, both in tower units.
    if all(v in diag for v in OBS_FLUX):
        le_p = D[:, diag.index("slhf")] * W_PER_J
        h_p = D[:, diag.index("sshf")] * W_PER_J
        le_o = Ob[:, obs_names.index("obs_slhf")] * W_PER_J
        h_o = Ob[:, obs_names.index("obs_sshf")] * W_PER_J
        m = np.isfinite(le_o) & np.isfinite(h_o) & (le_o > 5.0)
        if m.sum() > 100:
            br_p = np.clip(h_p[m] / le_p[m], -20, 20)
            br_o = np.clip(h_o[m] / le_o[m], -20, 20)
            rows["bowen"] = dict(n=int(m.sum()),
                                 mae=float(np.nanmean(np.abs(br_p - br_o))),
                                 median_err=float(np.nanmedian(br_p - br_o)))
        m2 = np.isfinite(le_o) & np.isfinite(h_o)
        if m2.sum() > 100:
            rows["energy_balance"] = dict(
                n=int(m2.sum()),
                residual=float(np.mean((le_p + h_p)[m2] - (le_o + h_o)[m2])),
                rmse=float(np.sqrt(np.mean(((le_p + h_p)[m2] - (le_o + h_o)[m2]) ** 2))))
    return rows


def print_scores(rows, label):
    print(f"\n--- {label} (held-out sites, vs towers) ---")
    print(f"{'variable':16s} {'n':>9s} {'RMSE':>10s} {'bias':>10s} {'r':>7s}")
    for k, v in rows.items():
        if k in ("bowen", "energy_balance"):
            continue
        u = " W/m2" if k in OBS_FLUX else ""
        print(f"{k:16s} {v['n']:9d} {v['rmse']:10.4g}{u:5s} {v['bias']:10.4g} {v['r']:7.3f}")
    if "bowen" in rows:
        print(f"{'Bowen ratio MAE':16s} {rows['bowen']['n']:9d} {rows['bowen']['mae']:10.4g}")
    if "energy_balance" in rows:
        eb = rows["energy_balance"]
        print(f"{'EB residual':16s} {eb['n']:9d} {eb['residual']:10.4g} W/m2  "
              f"(rmse {eb['rmse']:.3g})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="pretrained model directory")
    p.add_argument("--data", default="data/fluxnet_o96.zarr")
    p.add_argument("--strategies", nargs="+", default=["S1", "S6"],
                   choices=sorted(STRATEGIES))
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=2000000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--soil-match", choices=["none", "mean"], default="mean",
                   help="bring tower soil observations onto the model's scale by "
                        "matching per-site means, so the loss constrains anomalies "
                        "rather than absolute values dominated by representativeness")
    p.add_argument("--anchor-data", default=None,
                   help="global store for the anchor term (S6)")
    p.add_argument("--anchor-samples", type=int, default=200000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--outdir", default="finetune")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    base_prog, base_diag, base_meta = None, None, None
    from . import infer
    _, _, base_meta = infer.load(args.base, device=args.device)
    norm = mlp.Normaliser.from_dict(base_meta["norm"])
    feat, prog, diag = base_meta["features"], base_meta["prognostic"], base_meta["diagnostic"]
    print(f"base: {args.base} | {len(feat)} features, {len(prog)} prognostic, "
          f"{len(diag)} diagnostic")

    offs = None
    if args.soil_match == "mean":
        offs = soil_offsets(args.data, prog)
        for v, o in offs.items():
            good = np.isfinite(o)
            print(f"  soil offset {v}: {good.sum()} sites, "
                  f"median {np.nanmedian(o):+.4g}")

    tensors = {}
    for split in ("train", "val"):
        X, Yp, Yd, O, obs_names = load_windows(
            args.data, feat, prog, diag, args.rollout, split,
            max_samples=args.max_samples if split == "train" else args.max_samples // 4,
            soil_offset=offs)
        tensors[split] = (torch.as_tensor(X), torch.as_tensor(Yp),
                          torch.as_tensor(Yd) if Yd is not None else None,
                          torch.as_tensor(O), obs_names)

    anchor = None
    if args.anchor_data:
        from .train_mlp import build_windows
        ads = data.open_store(args.anchor_data, ("2020", "2021"), temporal=True, prof="o96")
        AX, _, _ = build_windows(ads, feat, prog, diag, args.rollout,
                                 max_samples=args.anchor_samples, quiet=True)
        anchor = torch.as_tensor(AX)
        print(f"anchor set: {len(anchor):,} global windows")

    ds = xr.open_zarr(args.data)
    sites = {s: str(g) for s, g in zip(ds.site.values, ds.split.values)}
    print(f"sites: {sum(v == 'train' for v in sites.values())} train, "
          f"{sum(v == 'val' for v in sites.values())} val")

    def fresh():
        m = mlp.AiLandMLP(len(feat), len(prog), len(diag), width=base_meta["width"],
                          depth=base_meta["depth"],
                          diag_blocks=base_meta.get("diag_blocks", 1))
        sd = torch.load(config.MODELS / args.base / "model.pt", map_location="cpu")
        m.load_state_dict(sd)
        return m.to(args.device)

    base_model = fresh().eval()
    for pm in base_model.parameters():
        pm.requires_grad_(False)

    results = {}
    results["base"] = evaluate_sites(base_model, norm, base_meta, tensors["val"],
                                     sites, args.device, args.batch_size)
    print_scores(results["base"], "aiLand-base (no fine-tuning)")

    outdir = config.MODELS / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for s in args.strategies:
        print(f"\n=== {s}: {STRATEGIES[s]['desc']} ===", flush=True)
        m = fresh()
        if STRATEGIES[s].get("deep_head"):
            m = widen_head(m).to(args.device)
        m = finetune(m, base_model, norm, base_meta, tensors, s, device=args.device,
                     batch_size=args.batch_size, seed=args.seed, anchor_batch=anchor)
        results[s] = evaluate_sites(m, norm, base_meta, tensors["val"], sites,
                                    args.device, args.batch_size)
        print_scores(results[s], f"{s} fine-tuned")
        torch.save(m.state_dict(), outdir / f"{s}.pt")
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {outdir}")
    return results


if __name__ == "__main__":
    main()
