# Full results and experiment log

The detailed numbers behind the headline results in [`README.md`](../README.md):
every ablation, rejected hypothesis, and lesson learnt getting here. Read
[`METHODS.md`](METHODS.md) first for what a script or term means, rather than
what it found.

---

## Lessons learnt

The most useful output of this exercise is the list of things that were quietly
wrong. Every one produced plausible-looking results.

1. **Physically obvious is not empirically right.** `snowc` cannot close a snow
   budget without the snow-mass prognostics `sd`/`rsn` — yet adding them made
   snow *worse* (`snowc` 0.55 → −1.49): they roll out badly themselves and feed
   that error back. v1 makes the opposite choice (promote `snowc`, drop snow
   mass from the state), and we found that from measurement before Table 1.
2. **Read the error signature, not just the error.** Turbulent fluxes scored R²
   −0.34 with near-zero bias — a *phase* error, since diagnostics train on t+1
   from t but the rollout wrote each prediction at t. Fixing the index alone,
   **no retraining**, moved `slhf` −0.34 → 0.96, `sshf` −0.67 → 0.96, `skt`
   9.39 K → 2.13 K.
3. **Silent NaN propagation deletes whole variables.** 114 missing values out of
   33.7 million made `mean`/`std` NaN, which made every `slhf` prediction NaN,
   which the scorer masked out — the variable vanished with no error anywhere.
   Statistics are now NaN-aware; an entirely-missing column raises.
4. **Scale the targets, or the loss is only about the biggest number.** Increments
   span four orders of magnitude (`stl1` ~2.5 K vs `ssro` ~1.5×10⁻⁴ m); unweighted
   least-squares is *entirely* soil temperature and snow. This is v1's tendency
   scaler. The diagnostics hit the same trap in physical units: `slhf` ~10⁷ J m⁻²
   swamps evaporation ~10⁻³ m.
5. **Judge by metric, never by figure.** v0's headline R² of 0.99 was `eval_set`
   pointed at training data — in-sample, and no metric was ever computed on the
   held-out year. Scored properly on 2022, v0's soil/snow state is mean R²
   **0.841**.
6. **Compare like with like before concluding anything is wrong.** The MLP looked
   far short of v1 until three fixes: the misalignment above, **excluding
   glacier/coastal points** (22.3% of O96 land, ~3× the soil-temperature error),
   and scoring a free 3-year rollout against the paper's *single-timestep* table.
   Corrected, the prognostic state is at parity — see below.
7. **Small silent bugs in your own tooling.** `land[::step][:npoints]` truncates
   rather than spans, so the first global sample silently had nothing south of
   30°S. Sampling is systematic, not stratified — a weakness given the rare
   biomes this work cares about.

---

## v1 reproduction: full detail

Full v1 recipe: O96, **11,538 land points, 1998–2019 training**, 2022 held out,
6×512, R=4→R=8, Adam 5e-4→3e-7 cosine with 1000-step warmup, batch 46,152. One
A100, 5 h.

**Metric and reference**, matching the paper's Table B1: RMSE (plus R², absent
from the paper's table) of **single-timestep** predictions against **ecLand
itself** — not ERA5, not FLUXNET, and not a free rollout (the separate Table 5
protocol below). Glacier/coastal excluded throughout; `H`/`LE` are the
time-averaged flux per 6-hourly step, W m⁻². Source:
`slurm/logs/ailand-repro.33370597.out`, held-out 2022.

| | paper RMSE | ours, 2 yr | **ours, 22 yr** | ratio | **ours R²** |
|---|---|---|---|---|---|
| `swvl1` | 0.01040 | 0.01050 | **0.010638** | 1.02× | 0.996 |
| `swvl2` | 0.003115 | 0.003250 | **0.003180** | 1.02× | 1.000 |
| `swvl3` | 0.000746 | 0.000791 | **0.000738** | **0.99×** | 1.000 |
| `stl1` | 1.056 K | 1.082 K | **1.117 K** | 1.06× | 0.993 |
| `stl2` | 0.138 K | 0.166 K | **0.154 K** | 1.12× | 1.000 |
| `stl3` | 0.032 K | 0.0128 K | **0.0109 K** | **0.34×** | 1.000 |
| `snowc` | 0.01462 | 0.05435 | **0.01717** | 1.17× | 0.997 |
| `2d` | 0.857 K | 2.099 K | 2.039 K | 2.38× | 0.977 |
| `2t` | 0.876 K | 1.608 K | 1.486 K | 1.70× | 0.990 |
| `skt` | 1.081 K | 1.986 K | 1.866 K | 1.73× | 0.988 |
| `H` | 11.57 W m⁻² | 17.62 | 17.24 | 1.49× | 0.960 |
| `LE` | 9.68 W m⁻² | 15.44 | 14.95 | 1.54× | 0.963 |

R² stays ≥0.96 even where the RMSE ratio is worst (`2t`, `skt`, `H`, `LE`); the
paper's table has no equivalent column. The RMSE ratio, against the paper's own
better-resourced run, is the more demanding comparison.

**The prognostic state (first seven rows) reproduces v1** — all at parity or
better, mean R² 0.989. 22 years of training fixed snow as expected (`snowc`
3.72× → 1.17×), confirming two years cannot represent snow variability.

**It did not close the diagnostic gap** (`2t` 1.61 → 1.49 K). Five hypotheses
tested and rejected:

| Hypothesis | Test | Verdict |
|---|---|---|
| Too little training data | 2 yr → 22 yr | **No** — fixed snow, not diagnostics |
| Diagnostic head too small | 1 → 3 blocks, 3× flux weights | **No** — no change |
| Missing albedo input | checked v1's Table 1 | **No** — v1 doesn't use it either; our inputs already match its 14 static + 7 dynamic + 3 temporal fields |
| RMSE aggregation convention | pooled vs per-gridpoint | Partly — `LE` 1.54× → 1.38×, rest ~5% |
| Too few distinct start times | 1,040 → 30,000, same budget | **No** — `2t` 1.4862 → 1.4836 K |

What remains is the gradient budget — ~5–10× short of the paper's, whose every
update spans all land points: a resource gap, not a method one.

Free autoregressive rollout through 2022, same points: mean R² 0.974, `stl1`
2.45 K, `swvl1` 0.0191, `2t` 1.55 K — stable, not just accurate one step out.

### Adding runoff, evaporation and GPP is free

`v1+fluxes` matches `v1` on v1's own variables (`stl1` 1.121 vs 1.117 K, `swvl1`
0.010642 vs 0.010638) while additionally predicting evaporation at R² **0.964**,
GPP at **0.984**, and runoff at 0.335/0.435 — at no cost.

---

## Which state vector

Presets in `ailand/config.py`, trained identically — 1000 trees, 2020–21, scored
on the 2022 rollout at point `x=5`:

* **`v0`** — soil/snow state: `swvl1-3`, `stl1-3`, `snowc`.
* **`v0+snow`** — adds `sd`/`rsn` and diagnostics `skt`/`aco2gpp`.
* **`v1`** — the same seven prognostic variables plus v1's diagnostic outputs.
* **`v0-asdistributed`** — the notebook as shipped, runoff in the state; kept
  only so the original remains runnable.

Held-out 2022 R²:

| variable | v0 | v0+snow |
|---|---|---|
| `swvl1` | **0.845** | 0.750 |
| `swvl2` | **0.741** | −0.071 |
| `swvl3` | **0.668** | −1.502 |
| `stl1` | **0.939** | 0.775 |
| `stl2` | **0.987** | 0.876 |
| `stl3` | **0.994** | 0.914 |
| `snowc` | **0.713** | −1.490 |
| mean | **0.841** | −1.197 |

**v0's state vector was already right** — 0.84 mean R² held-out; later gains
come from temporal forcings, target scaling and the MLP, not the state itself.

**Adding the snow prognostics makes snow worse** — `sd`/`rsn` roll out poorly and
feed that error into `snowc` (0.713 → −1.490). v1 makes the opposite choice.

**A note on runoff.** The notebook as distributed listed `sro`/`ssro` among its
targets — a flux in the state, which is also the input, feeding its own noise
back into soil moisture. A data-configuration slip, not part of v0's design.
Runoff is a useful *output* (below), just not a state.

---

## Temporal forcings and target scaling

`--temporal` adds v1's temporal/astronomical inputs — julian-day and local-time
sin/cos, plus TOA `insolation`. v0 had no notion of season or time of day at all.

`--scale-targets` divides each increment by its own std before fitting — v1's
tendency scaler. Held-out 2022 R², preset `v1`, 1000 trees:

| variable | baseline | +temporal | +scaled | +both |
|---|---|---|---|---|
| `swvl1` | 0.845 | 0.881 | 0.878 | **0.898** |
| `swvl2` | 0.741 | 0.886 | 0.871 | **0.949** |
| `swvl3` | 0.668 | 0.956 | 0.857 | **0.967** |
| `stl1` | 0.939 | **0.951** | 0.946 | 0.944 |
| `stl2` | 0.987 | 0.991 | 0.991 | 0.991 |
| `stl3` | 0.994 | 0.988 | **0.995** | 0.991 |
| `snowc` | **0.713** | 0.517 | 0.702 | 0.416 |
| `skt` | 0.704 | 0.653 | **0.705** | 0.652 |
| mean | 0.723 | 0.747 | 0.763 | 0.746 |

Both help and compound on soil moisture (`swvl3` 0.668 → **0.967**, the largest
gain in this repo), matching v1's claim that temporal forcings specifically
improve the soil column. Two caveats:

* **Snow drags the mean down** — these 10 points sit at ~51.6°N, mean snow cover
  1.8%, a rare signal easily degraded by anything sharpening the fit elsewhere.
  Judge snow on the paper's cold-biome results, not these points.
* **Scaling helps less than expected, structurally** — XGBoost's default
  `one_output_per_tree` fits each target with its own trees, so per-target
  scaling is nearly a no-op; it bites only when targets share structure (a
  neural network, or `multi_output_tree`, far worse here at mean R² −5.6). The
  tendency scalers do their real work in the MLP.

---

## Runoff as a diagnostic output

aiLand v1 does not output runoff. Surface/subsurface runoff (`sro`/`ssro`) are
soil-column fluxes, so they belong on the **diagnostic** branch — predicted each
step, bounded at zero, never fed back. `v1+runoff` does this; `v1+fluxes` adds
evaporation and GPP.

It's free: `sro` reaches R² +0.234, `ssro` +0.229, prognostic scores unchanged.
Carried as *state* instead, `sro` scores −14.4 and drags `swvl2`/`swvl3` to
−0.545/−1.235 — a flux in the state is also an input, so its noise propagates
into the soil column. That's the reason for the diagnostic branch.

---

## The MLP: ablation against XGBoost

`ailand/mlp.py` + `train_mlp.py` implement v1's architecture: shared prognostic
backbone (Linear → LayerNorm → ReLU) to increments, a diagnostic branch,
residual updates, bounds as post-processing. Training follows v1: smooth L1
over the rollout scaled by 1/R, tendency-scaled increments, Adam cosine+warmup,
grad clip 5.0, two phases (R=4 → R=8). Scaled to 4×256 (287k params) vs v1's
6×512 (1.3M), since this trains on 10 points not 171,039.

Held-out 2022, preset `v1+runoff --temporal`, same data for both:

| variable | XGBoost (+temporal +scaled) | **MLP** |
|---|---|---|
| `swvl1` | 0.898 | **0.964** |
| `swvl2` | 0.949 | **0.968** |
| `swvl3` | 0.967 | **0.992** |
| `stl1` | 0.944 | **0.986** |
| `stl2` | 0.991 | **0.996** |
| `stl3` | 0.991 | **0.999** |
| `snowc` | 0.416 | **0.860** |
| `skt` | 0.670 | 0.663 |
| `sro` | 0.116 | 0.127 |
| `ssro` | **0.242** | 0.160 |
| `aco2gpp` | −0.072 | −0.074 |

The MLP wins on every prognostic variable, largest gain `snowc` (0.416 → 0.860)
— exactly the compounding-error case the rollout loss should help. Held-out
`stl1` RMSE is 0.857 K. It's also **differentiable** end to end (v1's stated
reason for choosing an MLP), and `mlp.rollout_batch` backpropagates through the
autoregressive update — what makes the multi-step loss possible at all.

---

## Full O96 results (free rollout)

**11,538 land points**, 2020–2021 training, held-out 2022, continuous
autoregressive rollout pooled over 500 points. v1 network size (6×512, 1.64M
params), one A100, **26 minutes** end to end (`slurm/train_gpu.sh`, job 33311304).

| | RMSE | R² | v1 (paper) |
|---|---|---|---|
| `stl1` | 2.93 K | **0.980** | 1.47 K (1-yr, N320) |
| `stl2` | 2.20 K | **0.988** | — |
| `stl3` | 1.78 K | **0.992** | — |
| `swvl1` | 0.025 m³ m⁻³ | **0.973** | 0.014 (90-day) |
| `swvl2` | 0.018 m³ m⁻³ | **0.983** | — |
| `swvl3` | 0.020 m³ m⁻³ | **0.977** | — |
| `snowc` | 0.073 | **0.969** | — |
| `2t` | 4.91 K | 0.945 | **0.61–0.69 K** |
| `2d` | 1.92 K | 0.990 | — |
| `skt` | 9.39 K | 0.834 | **1.06 K** |
| `slhf` (LE) | — | **−0.337** | 10.1–11.1 W m⁻² |
| `sshf` (H) | — | **−0.670** | 11.7–12.5 W m⁻² |
| `e` (evaporation) | — | **−0.335** | — (v1 outputs LE, not E) |
| `sro` / `ssro` | — | 0.388 / 0.384 | not output by v1 |
| `aco2gpp` | — | −0.131 | not output by v1 |

**Prognostic state close to v1** — R² 0.97–0.99, `stl1` RMSE within a factor of
two, reasonable given O96 vs v1's N320, fewer epochs, and 3M of 33.7M available
rollout windows.

**Diagnostic branch is not.** `2t` 4.91 K vs 0.61–0.69 K, `skt` 9.39 K vs 1.06 K,
fluxes at negative R² — worse than climatology. Likely causes: a single-block
diagnostic head with no per-variable loss weighting; global rather than
per-point standardisation of spatially variable fluxes; simply less training
than v1. Two consistency checks did pass: `corr(slhf, e) = 0.9992`, and `e`/`slhf`
R² within 0.002 of each other.

LE and H are exactly the two variables v1 fine-tunes on FLUXNET (30%/20%
reported gains) — our being weakest there motivates `docs/STRATEGY.md`.

---

## FLUXNET-Shuttle fine-tuning: first results

**"aiLand-base" here is this repo's own reproduction** (`models/repro_v1_fluxes`
above, the default `--base` in `slurm/finetune.sh`) — **not** the paper's
published checkpoint from [Zenodo](https://doi.org/10.5281/zenodo.20764680). No
weights from the paper are loaded anywhere in this repo.

Fine-tuned on the Shuttle pool at O96 — 288 training, **74 held-out** sites,
scored against the towers in tower units: RMSE, bias, Pearson r, 6-hourly. S1/S5
are v1's; S6 is new.

| | `LE` RMSE | `LE` r | `H` RMSE | `H` r | `swvl1` | `stl1` | `stl2` | `stl3` | EB residual |
|---|---|---|---|---|---|---|---|---|---|
| aiLand-base | 61.98 | 0.694 | 63.33 | 0.781 | 0.0746 | 3.423 | 2.694 | 2.240 | 7.68 |
| S1 (freeze) | **56.02** | **0.746** | **57.65** | **0.823** | 0.0746 | 3.423 | 2.694 | 2.240 | **1.34** |
| S5 (full LR) | 56.24 | 0.745 | 58.81 | 0.817 | 0.0745 | 3.465 | 2.704 | 2.241 | 3.79 |
| **S6 (constrained)** | 56.24 | 0.743 | 58.27 | 0.818 | **0.0717** | **3.068** | **2.383** | **2.005** | 3.24 |

Fluxes W m⁻², soil moisture m³ m⁻³, soil temperature K. Source: `.rescore.log`.

**Every strategy improves the fluxes** 9–10%, EB residual 7.68 → 1.3–3.2 W m⁻².
v1 reports larger gains (30%/20%, closure 13.4 → <3 W m⁻²); ours are smaller, as
expected comparing a 125 km cell against a point tower.

**Only S6 improves the prognostic state**: `stl1` 3.423 → **3.068 K** (−10.4%),
`stl2`/`stl3` similar; `swvl1` 0.0746 → **0.0717** (−4.0%, r 0.844 → 0.854). S1
can't move it (backbone frozen); S5 unfreezes everything but has no soil term,
so it drifts and slightly *degrades* `stl1`. **This is what v1 could not do**:
soil observations on the prognostic head train the backbone on evidence, while
the anchor keeps fluxes from regressing.

Bowen ratio MAE is essentially flat (6.03 → 6.03 for S6) against v1's 42%
reduction — both flux magnitudes improve without their partition fixed, likely
because the two flux terms are weighted independently, with nothing
constraining their ratio.

---

## N320: what it changed, and what it did not

Worth it for the observational work, not for the emulator.

**Helped fine-tuning a lot** — collocation was the limiting factor:

| | O96 | N320 |
|---|---|---|
| tower cells retained | 362 | **529** |
| ... with soil observations | 335 | **475** |
| median site-to-cell distance | 46.2 km | **12.7 km** |
| base model `LE` RMSE at towers | 61.98 | **51.29** W m⁻² |

Same weights, finer grid — a 17% error reduction from comparing a tower against
a 31 km cell instead of a 125 km one.

**Did nothing for the emulator.** Native N320 base (Table B1 protocol):

| | ours, O96 base | ours, N320 base | paper N320 |
|---|---|---|---|
| `swvl1` | 0.01064 | 0.01123 | 0.01069 |
| `stl1` | 1.117 | 1.163 | 1.055 |
| `snowc` | 0.01717 | 0.01882 | 0.01644 |
| `2t` | 1.486 | 1.550 | **0.616** |
| `skt` | 1.866 | 1.980 | 1.038 |
| `LE` | 14.95 | 14.76 | 9.18 |

Prognostics stay at parity (1.04–1.10×); diagnostics unchanged or marginally
worse, while the paper's *improve* with resolution (`2t` 0.876 → 0.616), widening
the ratio 1.70× → 2.52×. v1 extracts something from the finer grid we don't —
attributed by the paper to "richer spatial heterogeneity" — making resolution
the sixth rejected hypothesis for the diagnostic gap. (Confound: the N320 base
trained on 2010–2022 vs O96's 1998–2019 — unlikely to matter, since 22 years
didn't help diagnostics either.)

---

## Fine-tuning against the towers, scored as v1 scores

Daily means at held-out sites, 2020–2022, against v1's Table 6 (17 held-out
sites, 2020–2023):

| | LE RMSE | LE r | H RMSE | H r |
|---|---|---|---|---|
| v1 base | 31.9 | 0.722 | 27.2 | 0.657 |
| **v1 S1** | **22.5** | **0.785** | **22.3** | **0.713** |
| ours base | 36.5 | 0.651 | 41.2 | 0.706 |
| ours S1 | 32.4 | 0.731 | 35.8 | **0.758** |
| ours S6 | 32.1 | 0.728 | 37.3 | 0.745 |

LE RMSE is 43% above v1's, H 61% above, but correlations are close — on `H`
ours is *better* than v1 both before and after fine-tuning. In improvement terms
we reach 11–13% against v1's 30%/18%, roughly half. Validation sets differ: v1
holds out 17 FluxDataKit sites, we hold out ~106 Shuttle-pool sites including
biomes PLUMBER2 under-represents — a harder test.

### The partition term

6-hourly scoring against the towers, N320 (source: `ailand-ft.34054555.out` for
base/S1/S6, `ailand-ft.34061535.out` for the corrected S7):

| | LE RMSE | LE r | H RMSE | H r | `swvl1` | Bowen MAE | EB residual |
|---|---|---|---|---|---|---|---|
| base | 51.29 | 0.714 | 59.21 | 0.787 | 0.0925 | 6.996 | 9.193 |
| S1 (freeze) | 45.41 | 0.784 | **54.29** | **0.819** | 0.0925 | 8.446 | 4.792 |
| S6 (constrained) | 44.32 | 0.793 | 56.34 | 0.804 | 0.0862 | 6.818 | 3.935 |
| **S7 (+ partition)** | **44.28** | **0.793** | 56.09 | 0.805 | **0.0861** | **6.728** | **−0.0001** |

S7 drives EB closure to essentially **zero** (v1: 13.4 → <3 W m⁻²) and is best
on LE and soil moisture. But Bowen MAE improves only 3.8% against v1's 42% —
not what constraining evaporative fraction should do, since EF *is* the ratio;
likely because Bowen MAE is dominated by heavy-tailed outliers the ±20 clip
doesn't tame, so the metric isn't measuring what the term fixes.

S1 alone *degrades* the Bowen ratio (6.996 → 8.446) while improving both flux
magnitudes — independent fitting can get both closer while the partition gets
worse. That's the pathology S7 exists to prevent, and does.

---

## aiLand v0 vs aiLand v1

| | v0 (this notebook) | v1 (Raoult et al., 2026) |
|---|---|---|
| Architecture | XGBoost, gradient-boosted trees | MLP: 6×512, LayerNorm + ReLU, ~1.3M params; shared prognostic backbone + diagnostic branch |
| Differentiable | no | **yes** — the stated reason for choosing an MLP |
| Training data | 10 land points, 2020–21, mock store | global N320, **171,039 land points**, 1998–2019, 6-hourly; 2022 held out |
| Data pipeline | hand-rolled `xarray` stacking | `anemoi-datasets`, YAML recipes, Zarr chunked in time |
| Prognostic targets | 9, incl. runoff | `stl1-3`, `swvl1-3`, `snowc` (increments) |
| Diagnostic targets | none | `2t`, `2d`, `skt`, `LE`, `H` (absolute) |
| Temporal forcing | none | time of day, day of year, TOA insolation |
| Normalisation | none | feature-wise stats; increments divided by **tendency scalers** |
| Loss | RMSE (typo'd objective) | smooth L1 (Huber, β=1), over rollout, scaled by 1/R |
| Rollout training | none (single step) | Phase 1: R=4 (24 h), 80 epochs; Phase 2: R=8 (48 h), 8 epochs |
| Optimiser | – | Adam, peak LR 5e−4 → 3e−7 cosine, 1000-step warmup, grad clip 5.0 |
| Hardware | 1 CPU | 4 GPUs, DDP, mixed precision |
| Bounds | `clip(x, 0, None)` | variable-specific bounds as post-processing |
| Observations | none | **fine-tuning on FLUXNET** (FluxDataKit): 199/42 sites, NaN-masked loss, 5 strategies S1–S5 |
| Evaluation | 1 point, by eye | RMSE/MAE/ACC vs climatology; 138×90-day integrations; 1-yr/4-yr rollouts; 6 k-means biomes; O96↔N320 transfer |

v1 headline: 90-day RMSE 1.19 K (`stl1`), 0.014 m³ m⁻³ (`swvl1`), stable over 4
years; fine-tuning cuts LE 30%, H 20%, Bowen error 42%, EB residual 13.4 → <3.

---

## What changed from the pristine v0 notebook

| Change | Why |
|---|---|
| Paths resolved from the repo root | `../tests/mock_data/...` didn't exist from this directory |
| `objevtive=` → `objective="reg:squarederror"` | v0's kwarg was a typo, silently accepted and ignored — the intended MAE objective never applied |
| `random_state=SEED` | `subsample=0.6` made v0 non-reproducible |
| Prognostic state indexed by name | v0 assumed targets were the last N features *in order* — breaks silently on any edit |
| Variable-specific physical bounds | v0 clipped everything to `[0, None]`, meaningless for soil temperature in K and missing snow's upper bound |
| Diagnostic branch (`skt`, `aco2gpp`) | Predicted as absolutes, *not* fed back, mirroring v1's diagnostic head |
| Scoring cell on the held-out year | v0 judged the rollout by eye; no metric was ever computed on 2022 |
| Split into `train`/`infer`/`evaluate` | A notebook can't be run per-stage, cached, or tested |
| Presets for v0/v0+snow/v1 state vectors | Makes the state-vector choice measurable rather than assumed |
| `tests/` for the ML path | Upstream tested only the GRIB→Zarr ingest |

**Kernel caveat.** The notebook's recorded kernel `ec_land_db` points at the
*system* interpreter, not a virtualenv, despite the upstream README — so
`requirements.txt` here is pinned from that interpreter instead, the one that
actually produced the stored outputs. The upstream `environment.yml` almost
certainly no longer solves; kept in `docs/upstream/` for reference only.

---

## Suggested next steps

1. **Hold out grid points, not just time** — only 2022 is genuinely independent;
   the weakest part of the evaluation.
2. **Move to the real data** — 10 points at one latitude can't show cross-biome
   or cross-resolution behaviour; O96 on `/lus` is next (`data/EXTERNAL.md`).
3. **Add the missing v1 diagnostics** — the mock store lacks `2t`, `2d`, `slhf`,
   `sshf`; the anemoi stores have all five.
4. **Fix `aco2gpp`** — negative skill everywhere; GPP isn't a memoryless function
   of current state, it needs phenology or a memory term.
5. **Scale the MLP toward v1** (6×512, longer rollouts, GPU) once trained on
   more than 10 points — `--device cuda` is already wired.
6. **Exploit the differentiability** — parameter sensitivity, then
   observation-constrained parameter estimation.
7. **Fine-tune on observations** — v1's FLUXNET stage corrects ecLand's own
   biases rather than just emulating them; needs (3) first.
