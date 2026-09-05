# Enhancing aiLand training beyond v1, using the FLUXNET Shuttle site pool

Source datasets: `/perm/pad/fluxnet-shuttle-ecland` (775 sites, ecLand-run, benchmarked)
and the anemoi O96/N320 stores on `/lus`. Reference: Raoult et al. (2026), aiLand v1,
[egusphere-2026-3620](https://doi.org/10.5194/egusphere-2026-3620).

---

## 1. What is new relative to v1

| | aiLand v1 | Shuttle pool |
|---|---|---|
| Source | FluxDataKit (PLUMBER-2 / AmeriFlux / ICOS) | FLUXNET Shuttle live (AmeriFlux 381, ICOS 342, TERN 52) |
| Sites after QC | 241 | 775 with forcing; 703 with soil files |
| Fine-tuning split | 199 train / 42 validation | to be defined |
| Site-years | not stated | 5,397 forcing; 5,035 with both SWC and TS |
| Fine-tuned variables | **LE and H only** | **LE, H, plus SWC and TS** |
| Constrains | diagnostic branch | diagnostic branch **and prognostic state** |

Measured coverage of the soil observations (`soil/shuttle-all775-era5/`, 703 files):

* 608 sites with usable SWC (5,156 site-years), 674 with TS (5,626), 586 with both.
* 2,000 SWC and 2,754 TS sensors carry a real `depth_m`, read from tower BADM
  `VAR_INFO_HEIGHT` by `extract_soil_ancillary.py`.
* Depth distribution across ecLand's four soil layers —
  SWC: L1 20%, L2 36%, L3 35%, L4 8%; TS: L1 31%, L2 33%, L3 31%, L4 6%.
  **The sensors constrain the whole column, not just the surface.**
* QC=0 (measured, not gap-filled): median 88% of valid SWC samples, 93% of TS.
* IGBP spread includes 228 sites in the fire/vegetation-stress classes
  (SAV/WSA/OSH/CSH/GRA) against PLUMBER2's 170 sites in total.

### Why this changes the method, not just the sample size

Every one of v1's five fine-tuning strategies (Table 2 of the paper) is an answer to a
single question: *how far may the prognostic backbone be allowed to move when the only
observational signal is on the diagnostics?* S1 and S2 freeze it outright so the
pretrained dynamics survive exactly; S3–S5 let it move at 10⁻⁷–5×10⁻⁵ and accept the
risk of catastrophic forgetting. The paper states plainly that separating batch-size
effects from data-distribution effects here would need a dedicated ablation it defers.

That whole tension exists **because the prognostic states were unobserved**. Soil
moisture and soil temperature observations remove it: the loss can act on the
prognostic head directly, so the backbone is trained on evidence rather than protected
from the absence of it.

There is a second, subtler gain. Because SWC and TS *are* prognostic, v1's multi-step
rollout loss can for the first time be applied against observations — score a 48-hour
autoregressive rollout against a tower's soil moisture trace, not just a single step.
v1 could only apply the rollout loss against ecLand, because only ecLand knew the state.

---

## 2. Stage 0 — fix the sampling first (prerequisite)

Current `ailand.extract` takes an even `np.linspace` stride over the land-point index.
That is systematic and reproducible, but it is *area-proportional*: rare biomes appear
in proportion to their extent, which is exactly the bias the Shuttle pool was built to
counter. Two changes:

1. **Stratify by biome.** Reproduce the paper's k-means biomes (Sect. 2.3.2: monthly
   mean and seasonal amplitude of 2 m temperature and precipitation, aridity index,
   top-layer soil moisture, snow cover, LAI, Bowen ratio; k=7, glacier cluster masked,
   local majority filter). Sample a fixed quota per biome rather than per unit area.
   Cheaper interim proxy: stratify on IGBP class from the tower metadata.
2. **Force-include the tower cells.** Whatever the global sample, the grid cells
   containing the 775 sites must be in it, so pretraining and fine-tuning share a
   geometry and a normalisation.

## 3. Stage 1 — pretrain unchanged

Global ecLand, v1's recipe exactly. This is the physics prior and should not be
touched: it is what gives the emulator globally consistent behaviour where no tower
exists. Our repo already reproduces the architecture, loss, tendency scalers, two-phase
R=4→R=8 rollout and optimiser schedule.

**Do this at N320, not O96.** At O96 (~125 km) a large fraction of the 775 towers will
collide in the same grid cell — the paper had to drop only 5 co-located sites at N320
(~31 km), and cell area is ~16× larger at O96. Fine-tuning against towers that share a
cell means fitting several conflicting observations to one model column. O96 is the
right grid for cheap method development; N320 is the right grid for the science.

## 4. Stage 2 — build the site-collocated dataset

For each of the 775 sites: nearest N320 land point, then dedup by cell keeping the
highest-quality record, as the paper does. Assemble a Zarr in the same layout as the
global set, with the observations carried as extra variables aligned on the model time
axis:

| target | source | conversion |
|---|---|---|
| LE | `flux/…_Flux.nc` `Qle_cor` | already W m⁻²; v1 uses the energy-balance-closed variant |
| H | `Qh_cor` | as above |
| SWC L1–L4 | `soil/…` `SWC_n` nearest `depth_m` | volumetric % ↔ model kg m⁻² / (thickness × 1000) × 100 |
| TS L1–L4 | `soil/…` `TS_n` nearest `depth_m` | °C → K |

Reuse `benchmark.py:pick_depth_var` rather than reimplementing the depth matching —
it already handles the case where index 1 is not the shallowest sensor.

Two filters to apply at build time, not later:

* **QC.** Keep QC=0 (measured) for the loss; QC 1–3 are gap-filled and should be
  either excluded or down-weighted. Do not mix them in silently.
* **Forcing provenance.** `reference/qc_report_shuttle-all775-era5.csv` classifies each
  (file, variable) as mild/medium/heavy/complete gap-fill — 1,169 heavy and 1,223
  complete out of 7,750. Where the *forcing* is largely ERA5, the tower is no longer an
  independent constraint, and the paper's own argument against training on reanalysis
  applies. Prefer `mild`/`medium` sites for the fine-tuning set and hold the heavily
  filled ones out, or at minimum record which is which.

## 5. Stage 3 — extend the loss

v1's loss already masks invalid target elements, which is what makes sparse
observational data usable at all. The extension is mechanical:

```
L = w_prog · Huber( (rollout_state − obs_soil) / tendency_scaler )     # NEW
  + w_diag · Huber(  diagnostics − obs_flux )
  + w_anchor · Huber( state − aiLand-base(state) )                      # NEW, see below
```

* Weight per variable. SWC/TS samples will vastly outnumber LE/H once four layers are
  in play; without weights the flux constraint that v1 demonstrated will be drowned.
* Apply the soil term **through the rollout**, not per step — this is the new capability.
* Normalise the soil residual by the same tendency scalers used in pretraining, so the
  observational and synthetic losses live in the same units.

## 6. Stage 4 — a fine-tuning strategy v1 could not run

v1's S1–S5 span "freeze the backbone" to "unfreeze at a tiny learning rate". Add:

**S6 — prognostic-constrained.** Unfreeze the full model. Put the soil observations on
the prognostic head and the flux observations on the diagnostic head, both at a normal
learning rate. The backbone now has evidence pulling it, so it need not be protected by
an artificially small step.

**Guard against forgetting explicitly, not implicitly.** Replace "use a 10⁻⁷ learning
rate and hope" with an anchor term: penalise divergence from the pretrained
`aiLand-base` predictions on a batch of *global ecLand* samples drawn alongside each
observational batch. The model then has to satisfy the towers **and** stay close to
ecLand everywhere else, which is the actual requirement. This is a standard
regularisation-toward-a-prior construction and is the same idea as a background term
*J*<sub>b</sub> in variational assimilation — the pretrained emulator is the background,
the towers are the observations.

**Bias handling.** A point sensor and a 31 km column mean are not the same quantity.
Fitting absolute soil moisture will import the representativeness error straight into
the model. Follow land-DA practice (de Rosnay et al.) and constrain **anomalies** —
CDF-matching or seasonal-anomaly matching per site — rather than absolute values, at
least for SWC. TS is far better behaved and can probably be used directly.

## 7. Stage 5 — validation design

* **Site-held-out**, stratified by IGBP, as v1 does (199/42 → scale to ~620/155).
* **Biome-held-out**: train on five biomes, test on the sixth. v1 did this against
  ecLand; doing it against *observations* is new and directly tests whether
  observational fine-tuning transfers or merely memorises.
* **Period-held-out**: v1 fine-tunes on 2000–2019 and evaluates on 2020–2023. The
  Shuttle pool has records to 2025, so keep the same structure with a longer tail.
* Keep the pretraining held-out year (2022) untouched by fine-tuning so the two stages
  remain separately auditable.

## 8. What to score

Beyond RMSE / R² / ACC, keep the paper's physical diagnostics and add soil ones:

* Bowen ratio error and energy-balance closure residual (v1: 13.4 → <3 W m⁻²).
* Soil moisture **drydown timescale** — the decay constant after rain events. A model
  can have good RMSE and completely wrong drainage; this is the diagnostic that catches it.
* Anomaly correlation of SWC per site, which is what the anomaly-space training targets.
* Snow-season soil temperature, since v1's residual errors concentrate in
  snow-insulated cold biomes and the Shuttle pool adds boreal/tundra sites.

## 9. Honest risks

| Risk | Note |
|---|---|
| Representativeness | Point sensor vs grid cell; worse for SWC than for fluxes. Mitigated by anomaly-space training, not eliminated. |
| Forcing is partly ERA5 | ~30% of records are heavily or completely gap-filled. Those towers are not independent of the reanalysis. |
| Sensor calibration | SWC absolute values vary between probe types and installations; another argument for anomalies. |
| Depth mismatch | Sensor depths rarely coincide with ecLand layer mid-points; `pick_depth_var` picks nearest, which is approximate. |
| Overfitting to towers | 5,035 site-years is large but spatially clustered in North America and Europe. The anchor term in Stage 4 is the main defence. |
| Gap-filled soil data | ~10% of soil samples are QC>0. Filter, don't blend. |
