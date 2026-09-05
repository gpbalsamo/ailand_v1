# aiLand in this repo: what the code does, and where each idea came from

Written for an ecLand modeller rather than an ML practitioner. Every machine-learning
term is explained on first use, with the closest ecLand or data-assimilation analogue.
Section 6 lists which recipe came from the aiLand v1 paper, which from elsewhere, and
which are my own choices.

---

## 1. The scripts

Everything lives in `src/ailand/`. Each file does one job and can be run on its own.

| File | Role | Analogy |
|---|---|---|
| `config.py` | Variable lists, physical bounds, dataset "profiles" | The namelist |
| `data.py` | Opens the Zarr store, builds the input/target arrays, derives solar geometry | The I/O and setup layer |
| `extract.py` | Pulls a subset out of the big anemoi store on `/lus` into this repo's layout | A `mars` request |
| `train.py` | Fits the XGBoost emulator | — |
| `train_mlp.py` | Fits the neural-network emulator | — |
| `mlp.py` | The network itself, plus the differentiable rollout | The model's dynamical core |
| `infer.py` | Runs the trained emulator forward in time | An offline ecLand run |
| `evaluate.py` | Scores the run against ecLand and plots it | Verification |
| `slurm/*.sh` | Batch submission, including the GPU node | — |

Two entry points you will actually type:

```bash
python -m ailand.train_mlp --profile o96 --preset v1+fluxes --data data/o96_2000.zarr --temporal
python -m ailand.evaluate  --profile o96 --preset v1+fluxes --data data/o96_2000.zarr --temporal \
                           --modeldir o96_mlp --npoints 200
```

---

## 2. Vocabulary, in ecLand terms

**Feature / input vector.** What the emulator is given at the start of a timestep:
the static physiography, the meteorological forcing, and the current prognostic state.
Identical in content to what ecLand needs to take a step.

**Target.** What it must produce: the state one 6-hourly step later.

**Increment (or "tendency").** ΔX over 6 h. We train the emulator to predict the
*change*, not the absolute value, and then add it to the previous state. This is the
single most important design choice: it is the difference between integrating a
tendency and re-diagnosing the whole state from scratch each step. Because increments
are small relative to the state, the model is anchored to the previous timestep and
cannot drift far in one step.

**Prognostic vs diagnostic.** Exactly the ecLand meanings. Prognostic variables are
carried forward in the state vector and have memory; diagnostic variables are
recomputed from the state at each step and have none. In the emulator this
distinction is operational: a prognostic variable is fed back into the input vector
for the next step, a diagnostic one is not. Getting a variable on the wrong side of
this line is what broke the original notebook (Section 5).

**Autoregressive rollout.** Running the emulator forward using *its own* previous
state, while reading the meteorology from the forcing data. This is precisely an
offline forced ecLand run. It is also the only honest way to evaluate an emulator:
scoring one step at a time hides error accumulation completely.

**Loss function.** The single number the fitting procedure minimises — the direct
analogue of the cost function *J* in a variational assimilation.

**Gradient descent / Adam.** The minimiser. Adam is a variant that adapts its step
size per parameter. Same role as the minimiser inside 4D-Var.

**Learning rate.** The step length in that minimisation. **Warmup** means starting
small and ramping up (so early, badly-scaled steps do not wreck the initial guess);
**cosine decay** means annealing smoothly to near zero at the end. **Gradient
clipping** caps the step length — a limiter, to stop one bad batch blowing up.

**Epoch.** One complete pass through the training data. **Batch.** The subset of
samples used for one minimiser step.

**Overfitting.** Fitting the noise in the training data rather than the signal.
Detected by scoring on a period the model never saw — here, the year 2022.

**R².** Fraction of variance explained. 1 is perfect; **0 means no better than always
predicting the long-term mean**; negative means *worse* than that. A negative R² is
not "a bit poor", it is a variable that would be better replaced by its climatology.

**RMSE.** Root-mean-square error, in the variable's own units.

**Standardisation / normalisation.** Rescaling each variable so they have comparable
magnitudes. Necessary because a least-squares cost function is dominated by whichever
variable carries the largest numbers. In this dataset soil temperature increments are
~2.5 K while subsurface runoff increments are ~1.5 × 10⁻⁴ m — four orders of magnitude
apart, so an unweighted cost is *entirely* soil temperature and snow, and runoff
receives essentially no gradient. Same problem as weighting observation types in an
assimilation cost function without dividing by their error variances.

**Tendency scaler.** The v1 term for the standard deviation of the 6-hourly increment,
used as the normalising factor above.

**MLP (multi-layer perceptron).** A stack of matrix multiplications separated by a
simple nonlinear function. With enough width it can approximate any smooth mapping.
**ReLU** is that nonlinearity: `max(0, x)`. **LayerNorm** rescales the intermediate
values within each layer to keep the optimisation well-conditioned — a preconditioner.

**Backbone and head.** A shared trunk that builds an internal representation, plus
small output branches. Our backbone produces the prognostic increments; a separate
diagnostic branch produces the diagnostic variables. Structurally this mirrors ecLand:
one column physics, several diagnosed outputs hanging off it.

**Backpropagation.** The chain rule applied through the network to get the derivative
of the loss with respect to every parameter. Mathematically the same construct as an
**adjoint model**. This is why "differentiable" matters: an MLP emulator comes with
its adjoint for free, which is what makes gradient-based parameter estimation and
variational assimilation possible. Gradient-boosted trees have no such thing.

**XGBoost / gradient-boosted trees.** A sum of many small decision trees, each fitted
to the residual left by the previous ones. Fast and strong on tabular data, but the
output is piecewise constant and non-differentiable — no adjoint, and no way to train
through a multi-step forecast.

**Multi-step (rollout) loss.** Instead of scoring one 6-hourly step, run the emulator
R steps forward and score the whole trajectory, then differentiate through all of it.
The analogy is close to 3D-Var versus 4D-Var: fitting to a single time versus fitting
to a trajectory over a window. This is what teaches the model not to accumulate error.

---

## 3. Why the point counts changed (10 → 200 → 2000 → 11,538)

First, a correction of a likely misreading: **it was never 5 points.** The original
notebook did `isel(x=5)`, which selects *point index 5* — a single grid point. The
mock store contains 10 land points in total:

```
x      = 60870 ... 60879        (10 points)
lat    = 51.61 for all ten
lon    = 4.63, 5.14, ... 9.26   (a single east-west line across NL/Germany)
```

So v0 **trained on 10 points and evaluated on 1**, all at the same latitude, spanning
about 300 km. That is a transect, not a sample. It cannot show anything about cold
climates, the tropics, or arid regions, and with 1.8% mean snow cover the snow scores
from it are close to meaningless.

The step to O96 is deliberately a ladder, not a jump:

| Stage | Points | Why that number |
|---|---|---|
| mock | 10 | What upstream shipped. One latitude. |
| O96 test | 200 | First real extraction; enough to confirm the pipeline works end to end globally. |
| O96 working | 2,000 | Current. ~2 GB, spans −89° to +83°, trains on CPU in minutes. |
| O96 full | 11,538 | Every land point at O96, `lsm > 0.5`. `slurm/extract.sh`. |
| N320 (the paper) | 171,039 | What aiLand v1 actually used. 1.2 TB store. |

The 200 and 2,000 figures are **my choices, not the paper's**, driven by one practical
fact: the anemoi store is chunked one timestep at a time (`chunks = (1, 76, 1, 40320)`),
so reading *any* number of points costs the same ~63 ms per 6-hourly step. Extracting
2,000 points costs no more read time than 200 — only more output disk. So the ladder is
really about what trains comfortably, not what reads comfortably. Going to all 11,538
land points is a change of output size (~12 GB), not of extraction time.

A caught bug worth recording: my first 2,000-point extract used
`land[::step][:npoints]`, which *truncates* the strided list rather than spanning it.
On a north-to-south ordered grid that silently deleted everything south of 30°S. The
fix is `np.linspace` over the land index; coverage went from −30.4° to −89.3°.

---

## 4. The procedure, step by step, with what each step bought

All numbers are held-out 2022 R², mock store, rollout at point 5. Each row is a single
change from the row above.

| # | Change | Mean R² | The variable it moved most |
|---|---|---|---|
| 0 | v0's soil and snow state | **0.841** | `stl3` 0.994, `swvl3` 0.668 |
| 1 | Add snow prognostics `sd`, `rsn` | −1.197 | `snowc` 0.713 → **−1.490** (worse) |
| 2 | Add temporal/astronomical forcing | 0.747 | `swvl3` 0.668 → **0.956** |
| 3 | Scale the increments by tendency scalers | 0.763 | `swvl2` 0.741 → 0.871 |
| 4 | Both 2 and 3 | 0.746 | `swvl3` → **0.967** |
| 5 | Add runoff as a *diagnostic* output | — | `sro` +0.234, soil states unchanged |
| 6 | Replace XGBoost with the MLP + rollout loss | — | `snowc` 0.416 → **0.860**, `stl1` → 0.986 |

v0's state vector was already the right one: its soil and snow emulation scores 0.84 on
a held-out year. The gains here come from the inputs, the loss and the architecture.

The result that matters physically:

**Adding the snow prognostics made snow worse.** Carrying `sd` and `rsn` looks
physically obligatory — `snowc` cannot close a snow budget without them. But they roll
out badly themselves and feed that error straight into `snowc`, which collapsed from
0.551 to −1.490. v1 makes the opposite choice: promote `snowc` to prognostic (it is
diagnostic in ecLand) and leave the snow mass variables out of the state entirely. We
reproduced that decision from measurement before reading it in Table 1.

**A note on runoff.** The notebook as distributed also listed `sro`/`ssro` among its
targets. A flux in the state vector is also an input, so its noise propagates into the
soil column — carried as state it scores R² −14.4 and drags `swvl2` to −0.545. That is a
data-configuration slip rather than a property of the v0 design, so the `v0` preset
states the design as intended and `v0-asdistributed` preserves the original for
reproducibility. Runoff is a useful *output* on the diagnostic branch (Section 6), and
one v1 does not provide.

---

## 5. Bugs found in the v0 notebook

| Bug | Effect |
|---|---|
| `objevtive=mean_absolute_error` | A typo. XGBoost silently accepts unknown keywords, so the objective stayed at the default and the intended MAE was never used. |
| Targets indexed by position (`[-len(targ):]`) | Assumed the targets were the last N inputs *in order*. Breaks silently on any edit to either list. |
| `np.clip(x, 0, None)` applied to everything | Meaningless for soil temperature in K; misses the upper bound on snow cover. |
| `snowc` treated as a fraction | It is a **percentage (0–99.9)** in the mock store. A `[0,1]` bound destroys the signal. In the O96 store it *is* a fraction — hence per-profile bounds. |
| No held-out metric at all | Skill was judged by eye from one figure whose range is dominated by the seasonal cycle. Measured, v0 has negative R² on two of three soil moisture layers. |
| `eval_set` = the training set | The reported R² ≈ 0.99 is in-sample and means nothing. |
| No seed | `subsample=0.6` is stochastic; results were not reproducible. |

---

## 6. Where each recipe came from

### From the aiLand v1 paper — Raoult et al. (2026), *aiLand v1: Physics-Based Land Surface Emulator with Observational Fine-Tuning*, EGUsphere preprint [egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620)

| Recipe | Where in the paper |
|---|---|
| MLP over LSTM/XGBoost, chosen for accuracy/efficiency **and differentiability** | Abstract, Sect. 2.2 |
| Architecture: Linear → LayerNorm → ReLU, 6 hidden layers × 512, ~1.3M params | Sect. 2.2.1 |
| Shared prognostic backbone + separate diagnostic branch | Sect. 2.2.1, Fig. 1 |
| Residual formulation: predicted increments added to the previous state | Sect. 2.2.1 |
| Variable-specific physical bounds as post-processing (not as loss penalties) | Sect. 2.2.1 |
| Smooth L1 (Huber, β=1) loss | Sect. 2.2.2 |
| Loss accumulated across the rollout, scaled by 1/R | Sect. 2.2.2 |
| Increments normalised by **tendency scalers** before the loss | Sect. 2.2.2 |
| Loss computed only over valid (non-NaN) target elements | Sect. 2.2.2 |
| Two-phase training, rollout R=4 (24 h) then R=8 (48 h) | Sect. 2.2.3 |
| Adam, peak LR 5×10⁻⁴ → 3×10⁻⁷ cosine, 1000-step linear warmup | Sect. 2.2.3 |
| Gradient clipping at global norm 5.0 | Sect. 2.2.3 |
| State vector: `stl1-3`, `swvl1-3`, `snowc` prognostic; `2t`, `2d`, `skt`, LE, H diagnostic | Table 1 |
| `snowc` promoted to prognostic although diagnostic in ecLand | Sect. 2.1.1 |
| Temporal/astronomical forcing: time of day, day of year, TOA insolation | Table 1 |
| `anemoi-datasets` Zarr layout, chunked in time | Sect. 2.1.5 |
| Training on 4 GPUs, DDP, mixed precision | Sect. 2.2.3 |
| Held-out year 2022 | Sect. 2.2.3 |

### From the wider literature (cited by the paper, or standard method)

| Recipe | Source |
|---|---|
| Solar declination and equation of time (for TOA insolation) | Spencer (1971) Fourier fits, as used in the NOAA solar position algorithm. **Not from the paper** — the paper's data already carries `insolation` precomputed; I had to derive it for the mock store, which does not. |
| Huber / smooth L1 loss | Huber (1964) |
| Adam optimiser | Kingma & Ba (2015) |
| Layer normalisation | Ba et al. (2016), cited in the paper |
| XGBoost | Chen & Guestrin (2016) |
| The MLP/LSTM/XGB comparison that preceded v1 | Wesselkamp et al. (2025), cited by the paper as the prototype v1 builds on |

### My own choices, not from any paper — each one measured, not assumed

| Choice | Rationale / result |
|---|---|
| **Runoff as a diagnostic output** | v1 does not output runoff at all. Adding Qs/Qsb to the diagnostic branch gives `sro` R² +0.234 at no cost to the soil states. This is an addition beyond the paper. |
| **Standardising the diagnostic targets** | The paper says diagnostics are *not* normalised before the loss — but its data is already normalised at the dataset level by anemoi. Our extracted store is in physical units, where `slhf` ~10⁷ J m⁻² and `e` ~10⁻³ m; without standardising, the diagnostic loss is 100% turbulent fluxes and evaporation gets no gradient. |
| Point counts 200 / 2,000 | Practical; see Section 3. The paper uses all 171,039 N320 land points. |
| Network 256 × 4 rather than 512 × 6 | We train on 2,000 points, not 171,039. `slurm/train_gpu.sh` uses the paper's 512 × 6. |
| The `v0` / `v0+snow` / `v1` / `v1+runoff` preset comparison | Built to *measure* the state-vector question rather than assume it. |
| Separating `v0` (design as intended) from `v0-asdistributed` | So a data-configuration slip is not reported as a property of the prototype. |
| Pooled multi-point scoring | Single-point scores are misleading for a globally trained emulator. |
| Even-`linspace` land sampling | See the truncation bug in Section 3. |
| All the v0 bug fixes in Section 5 | — |

---

## 7. Hardware

This login node (`ac6-102`) has no GPU, so everything so far is CPU-only, which is
what forced the reduced network and the sample cap. The Atos partitions do have them:

```
gpu        30 nodes   gpu:ga100:4     (4x NVIDIA A100 per node)
gpu_debug  32 nodes   gpu:ga100:4
```

That is the same hardware aiLand v1 used. `slurm/train_gpu.sh` submits a single-GPU
job at the paper's network size; the comment block at the end of it shows what needs
adding for all four (a `DistributedDataParallel` wrapper and a distributed sampler,
which is v1's "4 GPUs with distributed data parallelism").
