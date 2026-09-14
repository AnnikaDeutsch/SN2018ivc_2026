# Radio (VLA/ALMA) Analysis SOP — SN 2018ivc

Standard operating procedure for the radio analysis of SN 2018ivc. This is a living
document, built up step by step as the analysis proceeds — each stage gets documented
here once it's actually been done, not in advance. Read this before picking the radio
analysis back up in a new session. Modeled on `CLAUDE_xray_SOP.md`, which documents the
parallel Chandra/X-ray workflow for this same object.

**Never run `git commit` (or `git push`) in this repo, on any file, for any reason,
even if asked to "commit" as part of a broader task.** The user always commits
directly themselves. Editing/creating/regenerating files, running analysis, and
staging (`git add`) are all fine — just never create the commit itself. If in doubt,
stop short of committing and say so rather than asking for one-time confirmation.

Data lives under `data/` (not a dedicated `data/Radio/` subdirectory the way Chandra
data does — see inventory below). Host galaxy NGC 1068, z = 0.003793 (see
[[project_sn2018ivc]] memory) — same redshift/distance context as the X-ray work.

For plotting conventions specific to this project (color scheme, canonical epoch list,
marker shapes, file organization), see `CLAUDE_plotting.md` / the `18ivc-plotting`
skill — it already has a detailed, actively-maintained section on radio SED/light-curve
conventions (see "Existing plotting conventions" below for a pointer into it). Don't
duplicate that content here; this document is about data provenance and modeling code.

## Environments

- **`18ivc_clean`** (conda env): the environment used for all radio work — plotting
  notebooks, `functions.py`-based fitting scripts, and `redback`. Has `redback` 1.16.0
  installed (`pip show`/`import redback` confirmed working). `redback-csm` (the CSM
  Fortran model-extension package cloned into `redback-csm/`, see below) is **not**
  installed in this env — `import redback_csm` fails. Activate with
  `conda activate 18ivc_clean`. Same env used for X-ray notebooks — see
  [[feedback_notebook_kernel]] memory, notebooks in this repo run under `18ivc_clean`,
  not `18ivc`.
- No CIAO/Sherpa-equivalent special environment is needed for radio — everything here
  is plain Python (numpy/scipy/pandas/astropy/matplotlib) plus `redback`.

## Radio data inventory

### Primary dataset: `data/radio_18ivc_data.csv`

The master flux-density table for SN 2018ivc, 63 rows, columns `phase` (days
post-explosion), `freq` (GHz), `flux` (mJy), `flux_err` (mJy), `telescope`. Built up by
hand in `SN2018ivc-data-readin.ipynb` (hardcoded arrays per epoch/source, concatenated
and written out) — that notebook is the provenance record for where each point came
from; not something to re-run casually since it's hand-transcribed literature +
proposal data, not a live query.

- **Telescopes:** VLA (41 rows), ALMA (21 rows), VLASS (1 row).
- **Phase coverage:** 4.1 to 2668.0 days post-explosion (~7.3 years).
- **Frequency coverage:** 1.52–250 GHz (L band through ALMA mm-wave). VLA rows span
  the classic bands (S/C/X/Ku/K/Ka, ~1.5–33 GHz); ALMA rows are ~92–250 GHz.
- **Data sources by epoch** (per `sed_epoch_grid.ipynb`'s `SOURCE_LABEL` mapping):
  ALMA points are from Maeda et al. 2023a,b; early/mid VLA points (through ~200 days)
  are archival/literature; the **five late-time epochs (~1300, 1700, 2100, 2500, 2650
  days) are this project's own VLA monitoring campaign** (proposal 19A-219 era, W.
  Cotton) and get special marker treatment (see plotting conventions below) and are
  the epochs with actual SED model fits (see below).

### Comparison sample: `data/SN*_radio_data.csv`

Literature radio light curves for other stripped-envelope (mostly Type IIb/Ib) SNe,
used for comparison plots (e.g. `radio_lc_IIb_comparison_Xband.ipynb`). Same
`phase,freq,flux,flux_err,telescope` schema as the 2018ivc file. Covers: SN 1993J,
SN 2001gd, SN 2001ig, SN 2003bg, SN 2008ax, SN 2010P, SN 2011dh, SN 2011hs, SN 2013df,
SN 2016bas, SN 2016gkg — a fairly standard IIb/Ib radio comparison sample, various
telescopes (VLA/EVLA/JVLA, ATCA, MERLIN, Ryle).

### `data/vla_imgs/` — VLA imaging products

CASA-style `.image` directories (not flat FITS), organized by epoch phase and band:
`phase-1300/{C,Ku,Ka}-band/`, `phase-1700/{C,Ku,S}-band/`, `phase-2100/{C,Ku,K,S,X}-band/`,
`phase-2400/*.image` (flat, e.g. `2018ivc_S_2529d_final.image`,
`2018ivc_X_2476d_final.image`), plus `18ivc_circular_region` (a saved CASA region file,
presumably the photometry aperture used to extract the flux densities in
`radio_18ivc_data.csv`). These are the underlying VLA images the late-time flux
densities were measured from — not themselves inputs to the SED-fitting/plotting code
below, which all works from the already-extracted `radio_18ivc_data.csv` table.

## Existing plotting conventions (pointer, not duplicated here)

`CLAUDE_plotting.md` already documents, in detail, the project-wide radio plotting
conventions — read it before making or editing any radio figure:

- **Canonical epoch → viridis color mapping** (11-epoch list spanning 4–2650 days,
  radio epochs interleaved with the 3 X-ray epochs on one shared color axis).
  `sed_epoch_grid.ipynb` hardcodes each epoch's `x` position from this table via a
  `CANONICAL_X` dict — don't recompute `linspace` locally over a figure's own epoch
  subset (that's what produced the "known inconsistency" the doc describes).
- **Frequency → turbo color mapping** for light curves (`LogNorm(vmin=1, vmax=250)`,
  matching this dataset's 1.52–250 GHz range).
- **Marker shape by data source**: circle = ALMA (Maeda+2023a,b), square = VLA (Bill
  Cotton / literature), **star = "this work"** — specifically the five late canonical
  epochs (1300/1700/2100/2500/2650 days), regardless of which `telescope` value the
  row has.
- **Known, deliberate inconsistency**: `figures/vlba_proposal_sed.png/pdf`,
  `sed_all_epochs_one_plot.png`, and `sed_subplots.png` use an older 5-epoch-only
  viridis mapping and were intentionally left unregenerated when the epoch list grew
  — their colors don't match newer figures. Don't "fix" this without asking first,
  since one of those feeds an existing proposal document.
- **One notebook per figure**, saved under `figure_notebooks/`, self-contained,
  outputs both `.png` and `.pdf` to `figures/`. Several root-level notebooks
  (`all_sed_plot.ipynb`, `vlba_proposal_figures.ipynb`, etc. — see below) predate this
  convention and are not where new figures should be added.

## SED / light-curve modeling code

### `functions.py` — shared model + fitting library

The central module nearly everything else imports. Radio-relevant contents:

- **Absorption models** (frequency-domain, single epoch):
  - `F_SSA(freq, K1, K2, p, freq_scale=10)` — synchrotron self-absorption model,
    `tau = K2*(freq/freq_scale)^-((p+4)/2)`, `F = K1*(freq/freq_scale)^(5/2)*(1-exp(-tau))`
    (based on Chandra 2018 formalism).
  - `F_FFA(freq, K1, K2, alpha, freq_scale=10)` — free-free absorption model, same
    Chandra 2018-based formalism, absorption index `alpha` in place of SSA's `p`.
  - `F_SSA_time(...)` / `F_FFA_time(...)` — time-dependent generalizations of the
    above (add power-law time evolution indices `a`, `b`/`beta`, `delta`), for
    lightcurve-style (freq, time) joint fitting rather than one epoch at a time.
  - `find_peak_SSA` / `find_peak_FFA` — grid-search the peak frequency/flux of a
    single-component curve given its K1/K2/index parameters.
- **Single/multi-component SSA fitting (Nayana et al. 2022 parameterization)** — the
  formulation actually used by the current per-epoch fitting workflow (see
  `chi2_comp.py` below), parameterized directly by peak flux/frequency rather than the
  K1/K2 constants above:
  - `F_SSA_Nayana(nu, F_p, nu_p, p)`, plus `_2comp`/`_3comp` variants that sum 2 or 3
    independent `F_SSA_Nayana` components.
  - `one_comp_ls_fit` / `two_comp_ls_fit` / `three_comp_ls_fit` — `scipy.optimize.curve_fit`
    wrappers that fit 1/2/3-component SSA models to a phase-windowed slice of a radio
    data table, with support for fixing the electron index `p` per component
    (`fix_p`), returning best-fit params and (optionally) chi².
- **MCMC scaffolding**: `lnlike`/`lnprob` (emcee-style log-likelihood/log-posterior)
  built around `F_SSA_time` — present in `functions.py` but no driver script currently
  calls them; appears to be infrastructure for a not-yet-built time-dependent MCMC fit,
  not part of the active per-epoch workflow.
- **Physical parameter conversions** (from best-fit SSA peak params to physical
  quantities), eqns following Chandra 2018:
  - `B_peak_SSA(p, F_p, D, nu_p, ...)` — magnetic field at the SSA photosphere.
  - `R_peak_SSA(p, F_p, D, nu_p, ...)` — shock/emission-region radius.
  - `Mdot_peak_SSA(m_H, B, t, v_wind, ...)` — mass-loss rate from B and radius.
- `functions.set_plot_style()` — shared matplotlib rcParams setup used by every figure
  notebook (per `CLAUDE_plotting.md`).

### `chi2_comp.py` — current, active per-epoch SED fitting script

The primary tool actually used to fit the late-time SEDs. CLI script: reads a
`phase,freq,flux,flux_err` CSV, restricts to a phase window (`--phase_lower`/
`--phase_upper`), fits 1-, 2-, and 3-component `F_SSA_Nayana` models via
`functions.{one,two,three}_comp_ls_fit`, and **picks whichever component-count has χ²
closest to 1** as the best-supported model for that epoch. Supports fixing the
electron index `p` per component (`--fix_p1/2/3`). Prints best-fit peak flux/frequency/
electron-index per component plus the derived **spectral index α = (p−1)/2**, and
writes:
- `model_params/best_fit_params_<avg_phase>_days.csv` — component-by-component
  F_p/nu_p/p/alpha table for that epoch.
- (with `--plot`) `figures/best_fit_model_<avg_phase>_days.png` — data + best-fit
  curve(s) SED plot for that single epoch (not styled to the project's
  `CLAUDE_plotting.md` conventions — uses fixed `blue`/`pink` colors, not viridis).

**`sed_fit_commands.txt`** is the log of the exact `chi2_comp.py` invocations run for
the five canonical late-time epochs (1300/1700/2100/2400/2700-day phase windows,
matching the ~2500/2650-day bins in the canonical epoch table once you account for
window vs. bin-center rounding), including the specific initial-guess and fixed-index
choices used per epoch — **this file is the reproducibility record for how the current
`model_params/best_fit_params_*.csv` files were generated**, not just a scratch note.

Current `model_params/` contents (all from `chi2_comp.py`, dated Apr 29–May 21 2026):
`best_fit_params_{1300,1700,2100,2500,2650}.0_days.csv` — one file per canonical
late-time epoch, each holding the fitted component parameters (including `alpha`) for
whichever component-count `chi2_comp.py` selected as best for that epoch. **No
early-epoch (~4–1000 day) SED fits exist** — those epochs are data-points-only in
`sed_epoch_grid.ipynb` (per `CLAUDE_plotting.md`'s open-items note).

### `mass_loss_rate.py` — Ṁ from fitted SSA peak parameters

Takes the per-epoch/per-component `nu_p`/`p` values (from the `model_params/` fits
above) and derives progenitor mass-loss rate via **Weiler et al. (1986) eq. 16**,
converting SSA peak frequency to an assumed free-free optical depth at 5 GHz
(`tau_5GHz = (nu_p/5)^2.1`) and then to Ṁ. Two `tau` derivation methods implemented
(`tau_at_5GHz(..., method="ff"|"ssa")` — free-free vs. synchrotron-self-absorption
optical-depth scaling). Output: `mass_loss_rates.csv` — one row per
(phase, component), with `nu_p`, `p`, `tau_5GHz`, and `Mdot (Msun/yr)`. Currently
populated for the five canonical late-time epochs' 2nd/3rd components (the epochs
where a distinct FFA-turnover component was identified), Ṁ values order
~6×10⁻⁴–4×10⁻³ M☉/yr. `plot_mass_loss.py` presumably visualizes this table (not
inspected in depth here — flag for a closer look if mass-loss trends become relevant).

### `rvb_calc.py` / `tffa_calc.py` — standalone physical-parameter calculators

Small argparse CLI utilities built on `functions.py`, not epoch-fitting pipelines:
- `rvb_calc.py` — given an SSA peak frequency/flux (or K1/K2/p to derive them via
  `find_peak_SSA`), electron index `p`, and observation time, computes shock radius
  R, velocity v (via `model_indep_params.yml`'s adopted distance `D`), and magnetic
  field B (`R_peak_SSA`/`B_peak_SSA`).
- `tffa_calc.py` — given ejecta temperature, mass, velocity, and a target frequency,
  estimates the time for the ejecta to become optically thin to free-free absorption
  (`functions.t_ffa_optically_thin`).

Both read shared physical constants/scale factors from `model_indep_params.yml`
(distance `D`, unit-conversion scales) rather than hardcoding them — the same config
file `rvb_calc.py` depends on for `D`/scale values.

### Earlier exploratory notebooks — superseded by the `chi2_comp.py` workflow

`SN2018ivc_FFA_fitting.ipynb` and `SN2018ivc_SSA_fitting.ipynb` (both dated Feb 2026,
i.e. before `chi2_comp.py`/`functions.py`'s current fitting helpers were built):
hand-tuned, "by-eye" single/two-epoch fits with hardcoded K1/K2 guesses, notes like
"Determine peak flux and frequency" via a manual max-finding loop, and at least one
`savefig` call to an absolute path outside this repo
(`/Users/adeutsch/Desktop/UVA/.../SN2018ivc_lightcurve.pdf`). These correspond to the
`figures/sed-18ivc-fitbyeye-{1300,1700,2100,2400}.png` outputs (dated Mar 18 2026).
**Treat these as historical/exploratory, not the current fitting method** — the
active, reproducible workflow is `chi2_comp.py` + `sed_fit_commands.txt` +
`model_params/`. Don't extend these notebooks; if by-eye fitting is ever needed again,
it should go through `functions.py`'s existing fit helpers in a new notebook instead.

### `redback/` and `redback-csm/` — Bayesian transient-fitting packages (present, mostly unused so far)

Both are full git-cloned source trees (not just pip installs) living in the repo root:

- **`redback/`** — the general-purpose `redback` package (Sarin et al. 2024,
  arXiv:2308.12806) for Bayesian inference (via `bilby`) on transient light
  curves/SEDs across many event types. **Installed and importable in `18ivc_clean`**
  (`redback` 1.16.0, confirmed via `import redback`).
- **`redback-csm/`** — a companion package (Sarin & Hirai 2026, arXiv:2605.19571)
  adding Fortran-based circumstellar-medium (CSM) interaction models as plug-ins to
  redback's model library ("once installed, all CSM models are automatically
  available in redback's model library"). **Not currently installed** in `18ivc_clean`
  — `import redback_csm` fails (`ModuleNotFoundError`). Its models are therefore
  **not** in `redback.model_library.all_models_dict` right now.
- **`redback_fit_example.py`** — a short example script (not yet run, no output
  files/logs from it found in the repo) showing the intended usage: load
  `data/radio_18ivc_data.csv` into a `redback.transient.Transient` (flux-density mode,
  frequency in Hz), fix `redshift=0.003793`, and run `redback.fit_model(...)` with the
  **`synchrotron_massloss`** model (confirmed present in
  `redback.model_library.all_models_dict` even without `redback-csm`) via `dynesty`
  nested sampling. This is the most promising path toward a **physically-motivated,
  uncertainty-quantified fit across all epochs simultaneously** (vs. `chi2_comp.py`'s
  per-epoch independent least-squares fits), but **has not actually been run/fit yet**
  — no corner plot, posterior, or fit-result files exist for it anywhere in the repo.

## Radio figure notebooks (`figure_notebooks/`)

- **`full_radio_lightcurve.ipynb`** (2026-08-27) — full multi-band light curve, all
  data in `radio_18ivc_data.csv`, frequency-as-color (turbo) legend by band. Outputs
  `figures/full_radio_lightcurve.png/.pdf`.
- **`late_time_lightcurve_zoom.ipynb`** (2026-08-27) — same style, zoomed to the
  late-time (this-work) epochs. Outputs `figures/late_time_lightcurve_zoom.png/.pdf`.
- **`radio_lc_IIb_comparison_Xband.ipynb`** (2026-08-27) — 3-panel comparison of
  SN 2018ivc's X-band (8–12 GHz) light curve against the IIb/Ib comparison sample
  (`data/SN*_radio_data.csv`), plus a combined single-panel version. Outputs
  `figures/radio_lc_IIb_comparison_Xband.png/.pdf`.
- **`sed_epoch_grid.ipynb`** (2026-09-01, most recently touched) — **the current
  reference SED figure**: 3×3 grid, one panel per canonical epoch (4/20/200/1300/
  1700/2100/2500/2650 days) plus an "all epochs" panel, log-log flux vs. frequency,
  colored by the canonical 11-epoch viridis mapping, marker shape by data source
  (circle=ALMA, square=literature VLA, star=this-work VLA per the five late epochs).
  **Data points only — no model curves overlaid** (the `chi2_comp.py` fits aren't
  plotted here). Reads `data/radio_18ivc_data.csv` directly (not the `model_params/`
  fit files). Outputs `figures/sed_epoch_grid.png/.pdf`. This is the notebook
  `CLAUDE_plotting.md` treats as canonical for the epoch→color mapping and hardcodes
  `CANONICAL_X` from — likely the right starting point/template for a new JWST-proposal
  SED figure that does need model curves and a spectral-index annotation.

## Other radio-relevant notebooks (repo root, pre-`figure_notebooks/` convention)

These predate the "one notebook per figure in `figure_notebooks/`" convention and are
not where new figures should be added (per `CLAUDE_plotting.md`) — listed here for
awareness, not as active workflow:

- **`SN2018ivc-data-readin.ipynb`** — builds `data/radio_18ivc_data.csv` from
  hand-transcribed per-epoch arrays (see "Primary dataset" above). Provenance record,
  not a live pipeline.
- **`vlba_proposal_figures.ipynb`** (2026-08-07) — SED + light curve for a past VLBA
  proposal, marking requested U-band coverage/phases. Outputs
  `figures/vlba_proposal_{sed,lightcurve}.png/.pdf` — these use the **old 5-epoch
  viridis mapping**, one of the "known inconsistency" figures flagged in
  `CLAUDE_plotting.md` (don't regenerate without asking, feeds an existing proposal).
- **`all_sed_plot.ipynb`** — another catch-all SED plotting notebook (large, 3.3 MB);
  outputs include `figures/sed_all_epochs_one_plot.png` and `figures/sed_subplots.png`
  — also on the old 5-epoch mapping, also flagged as deliberately un-regenerated.
- **`vla_IIb_prop.ipynb`**, **`sn_comp_18ivc.ipynb`** — data loading + comparison-sample
  plotting, likely earlier drafts of what became `radio_lc_IIb_comparison_Xband.ipynb`.
- **`earlier_epochs_comp.ipynb`** — comparison focused on the early epochs
  (~4–1000 days); not closely inspected, worth a look if early-epoch modeling starts.
- **`SNeIIb-radio-Poonam-plotting.ipynb`** — plotting using the comparison-sample data,
  likely tied to the same collaborator (Poonam Chandra) whose Chandra proposal work
  appears in the X-ray SOP.
- **`vla_imaging_parameter_calc.ipynb`** — VLA observation/imaging parameter
  calculations (sensitivity, integration time, etc.) — imaging-setup planning, not SED
  analysis.

## Investigation — why component count differs between 2100 and 2500/2650 days (2026-09-10)

Prompted by wanting to compare fitted spectral index α across the last 3 late-time
epochs (2100/2500/2650 days) for the planned JWST-proposal figure: 2100 days'
`model_params/best_fit_params_2100.0_days.csv` has 2 components, while 2500 and 2650
days each have only 1. Reproduced all three fits from `sed_fit_commands.txt` exactly
(bit-for-bit identical best-fit params to the existing CSVs — done in an isolated
scratch copy of `chi2_comp.py`/`functions.py`, not by rerunning in place, to avoid
touching `model_params/`) and inspected the chi2 values `chi2_comp.py` doesn't
normally print for the non-selected models:

| Epoch | 1-comp reduced χ² | 2-comp reduced χ² | 3-comp reduced χ² | Selected |
|---|---|---|---|---|
| 2100 days | 7.92 (bad) | **1.26** | 2.54 | 2-component |
| 2500 days | **0.99** | 1.39 | 4.12 | 1-component |
| 2650 days | **1.14** | 1.71 | 9.99 | 1-component |

**This is not a fitting artifact or bug — it's the "closest to χ²=1" selection
criterion correctly responding to a real change in SED shape.** At 2100 days, a
single SSA component fits terribly (reduced χ²=7.92): the 8 data points (3–22 GHz)
have a shape that genuinely needs two absorbed components (turnovers at
nu_p≈4.2 GHz and ≈21 GHz) to describe. By 2500 and 2650 days, a single component
already fits close to perfectly (reduced χ²≈1) — adding a second component doesn't
capture any new structure, it just overfits: in both epochs the fitted "component 2"
collapses to a near-duplicate of component 1 (same nu_p, amplitude pinned at the
`--amp_lower` floor of 1.0 mJy), which is why its χ² gets *worse*, not better, than
the 1-component fit. So the physical read is that the higher-frequency
(~9–21 GHz-turnover) absorbed component visible at 2100 days is no longer
distinguishable from the data by 2500/2650 days — consistent with, and supportive
of, the spectral-shallowing story (the SED is simplifying to a single, less
absorbed/optically-thinner component at the same time α is dropping).

**Reproducibility gotcha found along the way:** `sed_fit_commands.txt`'s commands
for the 1300/1700/2100-day epochs use a bare `--fix_p` flag, which **no longer runs
on the current `chi2_comp.py`** (`error: ambiguous option: --fix_p could match
--fix_p1, --fix_p2, --fix_p3`) — that single flag was split into per-component
`--fix_p1/2/3` in commit `38d2e5f` (2026-05-12), fixing *all* components' p when
set (there was no way to fix only some). The 1300/1700/2100 fits (`model_params/`
dated May 4–5) predate that commit; the 2500/2650 fits (dated May 21) postdate it
and already use the new `--fix_p1 --fix_p2` form. To rerun the three older commands
verbatim today, substitute `--fix_p1 --fix_p2 --fix_p3` for the bare `--fix_p` —
confirmed this reproduces the existing 2100-day CSV exactly. Consider updating
`sed_fit_commands.txt` itself to the runnable form next time it's touched, so it
stays a true reproducibility record.

## `sed_spectral_shallowing.ipynb` — JWST proposal figure (2026-09-10)

`figure_notebooks/sed_spectral_shallowing.ipynb` -> `figures/sed_spectral_shallowing.png/.pdf`.
Single-panel SED overlaying data + best-fit `chi2_comp.py` curve for the 3 most
recent epochs (2100/2500/2650 days), all on one plot in their canonical viridis
epoch colors (data star markers + darkened-shade fit curves, per
`CLAUDE_plotting.md`'s model-curve-color convention), with an inset showing fitted
α vs. phase across the same 3 epochs.

Notable build issues found and fixed (all by rendering and inspecting the actual
PNG each time, not guessed):
- **Inset background must be fully opaque (`alpha=1.0`)**, not `0.9` — at 0.9 the
  main SED curves visibly bled through the inset, corrupting it into what looked
  like a stray glyph and ghost curves.
- **Inset placement must avoid real data, not just "look empty."** An
  upper-right inset covered several real 10-24 GHz data points; an upper-left
  legend covered the one 1.52 GHz point. Checked numerically (not by eye): every
  real data point across all 3 epochs sits above y-axis-fraction ~0.78 (because
  the curves' extrapolated low/high-frequency tails stretch the log y-range far
  below the data), so the bottom ~75% of the panel is safe for both the inset
  (bottom-left) and legend (bottom-right), as long as they're kept apart in x and
  each inset/legend's own tick labels are given clearance from the main panel's
  axis labels at the panel edges.
- **`freq_range` extended to 45 GHz** (past the ~24 GHz data max) so the 2100-day
  epoch's 2-component fit fully shows its second SSA turnover (nu_p≈21 GHz)
  instead of being cut off mid-rise, which otherwise reads as a broken curve.
- Explicit x-ticks (`[2,3,5,10,20,40]`) needed — default log-axis tick locator
  only draws one major tick over this narrow (~1.5 decade) range.

The double-peaked shape of the 2100-day curve (visible in the figure) is real,
not a rendering artifact — see the "Investigation" section above: it's the
2-component fit's second, higher-frequency SSA component, which the χ² selection
found necessary at 2100 days but not at 2500/2650 days.

### Free-*p* sanity check (2026-09-10) and figure revision to 4 epochs

Prompted by the fact that every one of the 5 late-time `model_params/` fits has
`p` **fixed by hand** (3.24 for 1300/1700/2100 days; 1.8 for 2500/2650 days, per
`sed_fit_commands.txt`), not fit freely — so the "shallowing" in the original
3-epoch figure was partly an assumed input, not purely a measurement. Checked by
re-running the fits with `fix_p=None` (bounds p∈[1.5, 5.0]), same phase windows
and initial guesses as `sed_fit_commands.txt`, for 2100/2500/2650 days:

| Epoch | Best free-*p* model | reduced χ² | Free-fit α | Fixed-*p* α |
|---|---|---|---|---|
| 2100 | 1-comp | 3.98 (bad fit) | 0.35 (unreliable — bad fit) | 1.12 |
| 2100 | 2-comp | 0.31 | **both p pegged at the upper bound (5.0)** — dof=2, underconstrained, not trustworthy | 1.12 |
| 2500 | 1-comp | 0.63 (good, no pegging) | **0.57** | 0.40 |
| 2650 | 1-comp | 0.26 (good, no pegging) | **0.55** | 0.40 |

**Conclusion: the shallowing survives a free fit and is not an artifact of the
fixed-*p* choice** — 2500 and 2650 days each independently converge, via a
well-behaved single-component fit with no parameters pegged at bounds, on
α≈0.55–0.57, distinctly shallower than the early epochs. 2100 days can't be
checked the same way: a single component doesn't fit at all (confirming the
2-component finding above), and the free 2-component fit pegs both components'
`p` at the imposed upper bound with only 8 data points and 6 free parameters
(dof=2) — genuinely underconstrained, not a number to trust, though if anything
it points toward 2100 being *at least* as steep as the fixed value, not
shallower.

**Decision (user, 2026-09-10): update `sed_spectral_shallowing.ipynb` to use
the free-*p* α at 2500/2650 days (not the fixed-*p*=1.8 `model_params/`
values), keep the fixed-*p* value at 2100 days (and add 1700 days, also
fixed-*p*, per the same "one epoch further back" discussion) since neither has
a trustworthy free-fit alternative.** Implementation:
- The free-*p* fit is computed **inline in the notebook** (calling
  `functions.one_comp_ls_fit(..., fix_p=None)` directly), not pasted numbers,
  so the notebook stays self-contained/reproducible.
- Results are also saved to `model_params/best_fit_params_{2500,2650}.0_days_freep.csv`
  — a **new file per epoch, alongside** (not overwriting) the original
  fixed-*p* `best_fit_params_{2500,2650}.0_days.csv`, which remains the
  `chi2_comp.py`/`sed_fit_commands.txt` record.
- The figure now has 4 epochs (1700/2100/2500/2650 days) instead of 3. The
  1700-day epoch's data pulled the real-data flux floor down to ~3 mJy (a
  33 GHz VLA point) and ~5.9 mJy within the left half of the panel, versus
  ~6.2 mJy before adding it — the legend/inset placement logic (data-driven,
  not eyeballed, see the earlier build-issues note above) was recomputed
  against the new floor, not just carried over.
- Inset originally marked the two free-*p* epochs with a small `*` next to
  their marker plus a one-line in-panel footnote — **removed at the user's
  request 2026-09-10** (an unexplained asterisk with no footnote would be
  confusing, so both were dropped together, not just the footnote text). The
  mixed fixed-*p*/free-*p* methodology is documented here and in the notebook
  markdown/comments, but is no longer visible on the figure itself — the user
  is handling that explanation in the proposal's figure caption instead.

Updated shallowing sequence shown in the figure: α = 1.12 → 1.12 → 0.57 → 0.55
(1700 → 2100 → 2500 → 2650 days).

## Open items / not yet done

- No SED model fits exist for the three early epochs (4/20/200 days) or any epoch
  beyond ~2650 days.
- `redback`/`synchrotron_massloss` Bayesian fit (`redback_fit_example.py`) has never
  actually been run — no posterior/corner-plot output exists yet. Could supersede the
  per-epoch `chi2_comp.py` approach with a single global fit, but that's a larger
  undertaking than the current per-epoch method.
- `redback-csm` is cloned into the repo but not installed in `18ivc_clean` — its CSM
  models aren't available until it's installed.
- No figure currently overlays `chi2_comp.py`'s fitted SED curves on the
  `sed_epoch_grid.ipynb`-style multi-epoch panel layout, and no figure computes or
  displays how the fitted spectral index changes across epochs — this is the gap the
  planned JWST-proposal spectral-shallowing figure will need to fill.
