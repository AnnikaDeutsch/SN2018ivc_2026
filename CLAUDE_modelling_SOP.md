# CSM-Interaction Modeling SOP (`redback-csm`) — SN 2018ivc

Standard operating procedure for CSM-interaction light-curve modeling of SN 2018ivc using
`redback-csm` (Sarin & Hirai 2026, arXiv:2605.19571) on top of `redback` (Sarin et al.
2024, arXiv:2308.12806). This is a living document, built up step by step as the
modeling proceeds — each stage gets documented here once it's actually been done, not in
advance. Read this before picking the CSM modeling work back up in a new session.
Modeled on `CLAUDE_radio_SOP.md` / `CLAUDE_xray_SOP.md`, which document the parallel
data-reduction workflows this modeling work consumes.

**Never run `git commit` (or `git push`) in this repo, on any file, for any reason,
even if asked to "commit" as part of a broader task.** The user always commits
directly themselves. Editing/creating/regenerating files, running analysis, and
staging (`git add`) are all fine — just never create the commit itself. If in doubt,
stop short of committing and say so rather than asking for one-time confirmation.

Host galaxy NGC 1068, z = 0.003793 (see [[project_sn2018ivc]] memory) — same
redshift/distance context as the radio and X-ray work.

## Environments

- **`18ivc_csm`** (conda env, Python 3.11.15): the environment for all CSM modeling
  work. Confirmed 2026-09-14: `redback` 1.16.0 and `redback-csm` 0.1.0, both editable
  installs from the git-cloned `redback/` and `redback-csm/` source trees in this repo
  root; `bilby` 2.8.0 and `dynesty` 3.0.0 present for sampling; `gfortran` (Homebrew
  15.2.0) available so the Fortran extension is a real compiled build, not a stub.
  `wind_bpl_bolometric` confirmed present in `redback.model_library.all_models_dict`,
  i.e. the plugin registration works end-to-end. Activate with
  `conda activate 18ivc_csm`.
  - **Correction to earlier notes:** `CLAUDE_radio_SOP.md`'s Environments section and
    `CLAUDE_xray_SOP.md`'s Step 6 both said `redback-csm` was not installed in
    `18ivc_clean`, dated as of 2026-09-03/09-10. That's still true for `18ivc_clean`
    specifically, but a **separate, already-configured env (`18ivc_csm`) exists and
    has both packages working** — use `18ivc_csm` for all modeling in this file, not
    `18ivc_clean`.
  - One harmless warning on `import redback`: `Plugin model '_nickelcobalt_engine'
    from 'csm_models' conflicts with a built-in model. Skipping plugin model.` —
    doesn't block anything; just means that one specific ⁵⁶Ni-engine plugin isn't
    double-registered under its own name. Not investigated further; revisit only if
    a nickel-powered CSM model variant is needed and doesn't appear in
    `all_models_dict`.
- **`18ivc_clean`**: not used for this SOP — `redback-csm` isn't installed there (see
  [[feedback_notebook_kernel]] memory; that memory's "always use `18ivc_clean`"
  guidance is for plotting notebooks, not CSM model fitting).

## Model-independent parameters

`model_indep_params.yml` (repo root) holds physical constants shared across the
radio/X-ray/modeling workflows. Relevant to CSM fitting, added 2026-09-14:

| Parameter | Value | Source |
|---|---|---|
| `mexp` (ejecta mass) | 3 M☉ | Maeda et al. 2023b |
| `eexp` (explosion energy) | 1.2 foe | Maeda et al. 2023b |
| `vwind` | 20 km/s | Maeda et al. 2023b (matches the pre-existing `v_wind` entry, previously attributed to Maeda+2023a for a different formula) |
| `D` | 10.1 ± 1.8 Mpc | Maeda et al. 2023 / Boestrom et al. 2020 |

**Decision (user, 2026-09-14): `mexp`, `eexp`, and `vwind` are fixed to these literature
values in every `redback-csm` fit below, not left free.** The fits are meant to
constrain CSM properties (Ṁ, density profile), not re-derive already-published ejecta
parameters. `delta`/`nn` (the BPL ejecta density-profile inner/outer slopes) are
**not** covered by this decision — no literature value found for these for SN 2018ivc,
so they remain free unless/until a source is found (flag if one turns up).

## Data inputs

- **Radio:** `data/radio_18ivc_data.csv` — 63 rows, `phase,freq,flux,flux_err,telescope`,
  4.1–2668 days, 1.52–250 GHz. See `CLAUDE_radio_SOP.md` for full provenance. For
  `redback`, frequency must be converted GHz → Hz (`freq * 1e9`), matching the existing
  (never-run) `redback_fit_example.py` pattern.
- **X-ray:** `data/Chandra/spectral_fitting/fits/xray_flux_luminosity_pyxspec_0p3_10kev.csv`
  — **only 3 rows** (one per unique epoch: 20306 at ~12.7 d, 29071+29072 merged at
  ~1868.7 d, 31211+31996 merged at ~2550.3 d), 0.3–10 keV unabsorbed flux and
  luminosity with asymmetric statistical errors (`flux_errlo/hi`, `lum_errlo/hi`) and a
  separate distance-systematic bracket (`luminosity_dist_lower/upper`), D=10 Mpc
  central (Maeda et al. 2023b). See `CLAUDE_xray_SOP.md` Step 5 for the full derivation
  and caveats (e.g. 29071+29072's stat error is comparatively well-behaved as of the
  2026-09-08 powerlaw-only revision, but is still the faintest/least-constrained epoch).
  **Only 3 epochs is sparse for a multi-parameter light-curve fit** — see the Step 3
  caveat below.

## Step 1 — Sanity checks before fitting, done 2026-09-15

Script: `csm_modeling/step1_sanity_checks.py` (new `csm_modeling/` dir, first script for
this SOP's step-by-step work; diagnostic plots go to `csm_modeling/diagnostics/`, kept
separate from `figures/`, which is for final publication figures per `CLAUDE_plotting.md`
— this is a quick-look check, not a paper figure).

- **Env confirmed:** `18ivc_csm` resolves at `/opt/anaconda3/envs/18ivc_csm` (not under
  `~/miniconda3` — use `source /opt/anaconda3/etc/profile.d/conda.sh` if `conda
  activate` can't find it). `redback` 1.16.0, `redback-csm` 0.1.0, `wind_bpl_radio` /
  `wind_bpl_xray` both present in `redback.model_library.all_models_dict`. Only the
  already-known harmless warnings (`_nickelcobalt_engine` plugin conflict, missing
  `lalsimulation`).
- **Transient objects built**, following the `redback_fit_example.py` pattern:
  - Radio: `redback.transient.Transient(data_mode='flux_density', ...)`, all 63 points,
    freq converted GHz→Hz.
  - X-ray: `redback.transient.Supernova(data_mode='luminosity', ...)`, all 3 points.
    `Lum50_err` here is a **provisional** quadrature sum of the statistical error
    (mean of `lum_errlo/hi`) and the distance-systematic half-width (mean of the
    `luminosity_dist_lower/upper` offsets from central) — good enough for this
    sanity-check plot, but **still not the final Step 3 decision** on how to combine
    these two error sources (or whether to fit them asymmetrically).
- **Quick non-Bayesian look:** `wind_bpl_radio`/`wind_bpl_xray` called directly (not
  `redback_csm.explore.csm_lightcurve_from_density`, which wasn't needed for this) at
  fixed `mexp=3, eexp=1.2, vwind=20`, three `mdot` guesses (1e-5, 1e-4, 1e-3 M☉/yr),
  representative ejecta/microphysics values (`delta=1, nn=10, eff=0.5, logepsb=-2,
  logepse=-1, p=3` for radio; `logepsx=-1` for X-ray), plotted against real data at
  4 representative radio frequencies (6/15/33/100 GHz) and all 3 X-ray epochs.
  **Result: right ballpark.** Radio data at all four frequencies fall within/near the
  `mdot=1e-4`–`1e-3` envelope; X-ray luminosities are consistent with similar `mdot`
  values. No order-of-magnitude mismatch or wildly wrong timescale — `wind_bpl` is a
  reasonable model family to proceed to Step 2's actual fit with.
- **Styling decision (user, 2026-09-15), applies to future CSM diagnostic/light-curve
  plots in `csm_modeling/`, not just this one:**
  - Radio panel: color follows `CLAUDE_plotting.md`'s "Frequency/band → turbo color"
    rule as-is — plain `turbo` colormap through `LogNorm(vmin=1, vmax=250)` on
    frequency, **not darkened**. Darkening (via the `darken_color` helper from
    `CLAUDE_plotting.md`) was tried and explicitly rejected by the user; don't
    reintroduce it here without asking again.
  - mdot (linestyle: dashed/solid/dotted for 1e-5/1e-4/1e-3) gets its own legend
    entries with **gray** proxy `Line2D` handles, kept separate from the
    frequency-colored data-point legend entries — putting an mdot label on one
    specific frequency's colored line (the original approach) read as if that
    linestyle only applied to that frequency, which was confusing.
  - X-ray model curves use a fixed **purple** (`darkviolet`), deliberately chosen to
    sit off the radio panel's turbo scale (which runs blue→cyan→green→yellow→
    orange→red) so the two panels' curves don't read as sharing one color axis when
    viewed side by side. This is specific to the X-ray panel not being a
    multi-frequency light curve — the turbo/frequency rule doesn't apply to it in the
    first place (see the "Quick non-Bayesian look" bullet above).

## Step 2 — Baseline radio-only fit: `wind_bpl_radio`, not started

Steady wind + broken-power-law ejecta (`wind_bpl`), radio wrapper. Simplest physically
motivated starting model, per the user's explicit decision to establish this baseline
before trying an eruptive/shell model.

- **Fixed:** `mexp=3`, `eexp=1.2`, `vwind=20`, `redshift=0.003793`.
- **Free:** `mdot`, `delta`, `nn`, `eff`, `logepsb`, `logepse`, `p`.
- **Data:** `data/radio_18ivc_data.csv`, all 63 points, multi-frequency (redback fits
  the frequency-dependence directly via the `frequency` array passed in
  `model_kwargs`, same pattern as `redback_fit_example.py`).
- **Priors:** start from `redback.priors.get_priors('wind_bpl_radio')` (auto-generated
  from the model signature per the `redback-csm` README), tighten `mdot`/`vwind`-scale
  ranges if the Step 1 sanity check suggests the auto priors are far from the data.
- **Sampler:** `dynesty` via `redback.fit_model`, `nlive` to be chosen once a first
  quick run's runtime is known (start at `nlive=500` per the package's own examples,
  adjust if convergence is slow given 7 free parameters).
- **Output:** Ṁ posterior (the primary science target), corner plot, multiband
  light-curve fit plot (`result.plot_multiband_lightcurve()` /
  `result.plot_corner()`).

## Step 3 — Baseline X-ray-only fit: `wind_bpl_xray`, not started

Same `wind_bpl` CSM/ejecta family, X-ray wrapper.

- **Fixed:** `mexp=3`, `eexp=1.2`, `vwind=20`, `redshift=0.003793`.
- **Free (tentative — see caveat below):** `mdot`, `delta`, `nn`, `eff`, `logepsx`
  (+ possibly `n_h_host` if absorption looks relevant at the 20306 epoch, which
  `CLAUDE_xray_SOP.md` Step 4 already flags as having a large free intrinsic N_H in
  the spectral fit).
- **Sparse-data caveat (important, unresolved):** only 3 X-ray epochs exist. A
  5-parameter fit against 3 data points is badly underconstrained on its own. Two
  options to consider once Step 2 is done, not yet decided:
  1. Fix `delta`/`nn` to the posterior median from the Step 2 radio-only fit (both
     wrappers share the same ejecta-profile parameters physically, so this is a
     legitimate informative-prior transfer, not just a computational shortcut) and
     fit only `mdot`, `eff`, `logepsx` against the 3 X-ray points.
  2. Treat the X-ray-only fit explicitly as a qualitative/exploratory step (e.g. wide
     priors, report that the posterior is prior-dominated) rather than a fully
     independent constraint, and lean on the Step 5 joint fit for the real
     constraining power.
  Revisit and pick one once Step 2's results exist.
- **Error handling:** resolve the `Lum50_err` question flagged in Step 1 (combining
  statistical + distance-systematic error, or fitting with asymmetric errors) before
  running this fit.
- **Output:** Ṁ posterior from X-ray alone, to compare against Step 2's radio-only Ṁ
  (Step 4).

## Step 4 — Compare radio-only vs. X-ray-only Ṁ (sanity check), not started

Per the user's explicit "separate fits first, then combine once each behaves sensibly"
plan: before building a joint fit, check that the independent radio and X-ray Ṁ
posteriors from Steps 2–3 are at least order-of-magnitude consistent. If they disagree
substantially, investigate (wrong band-specific microphysics parameter, a bad prior, an
error-bar problem in one dataset) before combining — a joint fit built on top of two
inconsistent per-band results would just average away a real discrepancy rather than
resolve it.

## Step 5 — Joint radio+X-ray fit: `wind_bpl`, not started

Depends on Step 4 showing the two bands are broadly consistent. The physical rationale
(from the `redback-csm` README): `_radio` and `_xray` wrappers for the same base CSM
scenario share the same underlying shock solution (`rshock`, shock velocity, upstream
CSM density) — so a joint fit ties Ṁ/CSM density to *both* datasets simultaneously
through one physical model, rather than reconciling two independently-fit numbers by
eye.

- **Fixed:** same `mexp`/`eexp`/`vwind`/`redshift` as Steps 2–3.
- **Free:** the union of Steps 2–3's free parameters — `mdot`, `delta`, `nn`, `eff`
  (shared, since both bands trace the same shock/CSM), plus `logepsb`/`logepse`/`p`
  (radio-specific) and `logepsx` (X-ray-specific).
- **Implementation — not yet worked out:** `redback.fit_model` is built around a
  single `Transient`/single-band likelihood. A combined radio+X-ray likelihood needs
  either (a) a `bilby.core.likelihood.JointLikelihood`-style sum of two per-band
  Gaussian likelihoods built directly from `redback`'s internal likelihood classes
  (one on the flux-density transient calling `wind_bpl_radio`, one on the luminosity
  transient calling `wind_bpl_xray`, both taking the shared free parameters), or
  (b) checking whether a newer `redback` version has multi-band/multi-instrument
  joint-fit support built in already. **Needs investigation before this step can
  start** — not just a parameter/prior choice like Steps 2–3.
- **Priors:** seed from Steps 2–3's marginal posteriors (as a cross-check that the
  joint posterior is consistent with, not just a compromise between, the two
  single-band fits) rather than starting from the raw auto-generated priors again.

## Step 6 — CSM mass from the `wind_bpl` fit, not started

Per the user's explicit interest in CSM mass (not just Ṁ), using
`redback_csm.analysis` once Step 5 (or, as a fallback, Step 2's radio-only posterior)
gives a CSM density/Ṁ posterior:

- Translate the fitted `mdot`/`vwind` into a CSM density profile (steady wind:
  ρ(r) = Ṁ / (4π r² v_wind)) over the radius range actually probed by the shock
  during the fitted epochs.
- `csm_mass_from_density_grid` (or the direct wind-mass integral, since a steady wind's
  enclosed mass has a closed form — check whether `analysis.py` has a dedicated
  steady-wind helper or whether this needs `cumulative_csm_mass_profile` on a
  manually-built grid) for the spherical-equivalent mass.
- `sample_geometry_corrected_mass` with covering-fraction and volume-filling-factor
  priors — **no covering-fraction/filling-factor prior has been chosen yet**, this
  needs a decision (uniform on [some range], informed by any disk/torus geometry
  evidence for this system, or a conservative wide default) before running.

## Step 7 — Eruptive/shell model: `gausswind_bpl`, not started

Per the user's explicit follow-up plan, after Steps 2–6 establish the steady-wind
baseline: refit with `gausswind_bpl` (Gaussian wind-history CSM + BPL ejecta) to test
whether a discrete mass-loss episode is preferred over a smooth wind. Motivated by
existing evidence already in `CLAUDE_xray_SOP.md` (Step 5.5: HR decreases
monotonically with phase, i.e. the X-ray spectrum softens over time — not on its own
proof of an eruption, but a spectral-evolution signature worth testing against a
shell-encounter model) and any bump/plateau structure in the radio light curve.

- Same fixed `mexp`/`eexp`/`vwind`/`redshift` as the wind_bpl fits.
- New free parameters replace `mdot`: `t_peak`, `t_width`, `mdot_baseline`,
  `mdot_peak`, plus `vwind` stays fixed (it's still a wind-history model, just
  time-variable in `mdot`, not `vwind`).
- Repeat the same sequence as `wind_bpl` — radio-only, X-ray-only, consistency check,
  joint — once the `wind_bpl` baseline (Steps 2–5) is actually done; not worth
  parallelizing before the baseline model's behavior on this data is understood.
- **Model comparison:** use the nested-sampling log-evidence (`log Z`) from both fits
  (`wind_bpl` vs. `gausswind_bpl`, same dataset) to compute a formal Bayes factor —
  don't just eyeball which fits "look better," per standard nested-sampling practice
  and consistent with how `05_simulation_and_inference.py` reports `log_evidence`.

## Step 8 — CSM mass under the eruptive model, not started

Same as Step 6, but for whichever of `wind_bpl`/`gausswind_bpl` the Step 7 Bayes factor
actually favors — a shell/eruption CSM mass is a genuinely different physical quantity
(mass ejected in one discrete event) from a steady wind's Ṁ integrated over an
assumed duration, so don't report both as if they were the same "CSM mass" without
being clear which model they came from.

## Figures

Once fits produce posterior/light-curve figures worth keeping, follow
`CLAUDE_plotting.md` / the `18ivc-plotting` skill's project-wide conventions (canonical
epoch→color mapping, one-notebook-per-figure under `figure_notebooks/`, `.png`+`.pdf`
output to `figures/`) rather than the raw `matplotlib` styling in `redback-csm`'s own
example scripts — those examples are for the package's own documentation, not this
project's figure conventions.

## Open items / TODO

- Step 1 done 2026-09-15 (see above); Steps 2–8 are still "not started."
- Step 3's sparse-X-ray-data handling (fix `delta`/`nn` from radio, vs. treat as
  exploratory) is an open decision, to be made once Step 2 has results.
- Step 5's joint-likelihood implementation mechanism (custom `bilby` joint likelihood
  vs. built-in `redback` support) needs investigation before that step can start.
- Step 6/8's covering-fraction and volume-filling-factor priors for
  `sample_geometry_corrected_mass` are undecided.
- No literature value found yet for `delta`/`nn` (BPL ejecta profile slopes) — these
  stay free in every fit above; flag if a source is found so they can be fixed too,
  consistent with how `mexp`/`eexp`/`vwind` were fixed.
- `CLAUDE_radio_SOP.md` and `CLAUDE_xray_SOP.md` both still describe `redback-csm` as
  "not installed" (true only for `18ivc_clean`) — worth a small correcting edit in
  both pointing here, so a future session doesn't re-derive the same stale conclusion
  (done here 2026-09-14, see this file's Environments section for the correction
  itself).
