# X-ray (Chandra) Analysis SOP — SN 2018ivc

Standard operating procedure for the Chandra/X-ray analysis of SN 2018ivc. This is a
living document, built up step by step as the analysis proceeds — each stage gets
documented here once it's actually been done, not in advance. Read this before
picking the X-ray analysis back up in a new session.

**Never run `git commit` (or `git push`) in this repo, on any file, for any reason,
even if asked to "commit" as part of a broader task.** The user always commits
directly themselves. Editing/creating/regenerating files, running analysis, and
staging (`git add`) are all fine — just never create the commit itself. If in doubt,
stop short of committing and say so rather than asking for one-time confirmation.

Data lives under `data/Chandra/`. The `heasoft-6.36/` subdirectory there is the
HEASoft software install, not data — ignore it for inventory purposes.

## Environments

- **`ciao-4.17`** (conda env): CIAO/Sherpa tools — `download_chandra_obsid`,
  `chandra_repro`, `specextract`, `dmcopy`, region tools, Sherpa fitting. Activate
  with `conda activate ciao-4.17`.
- **`18ivc_clean`** (conda env): general Python/astropy env used for the rest of the
  project (plotting, notebooks). Has `astropy` for FITS header/data inspection but
  not the CIAO tools. Activate with `conda activate 18ivc_clean`. See
  [[feedback_notebook_kernel]] memory — notebooks in this repo run under
  `18ivc_clean`, not `18ivc`.

## ObsID inventory

| ObsID | Target | Date-obs | Exposure | PI / SEQ_NUM | Notes |
|---|---|---|---|---|---|
| 31211 | SN 2018ivc | 2025-11-16 08:06–14:19 | 19.8 ks | Poonam Chandra / 503689 | Fully repro'd + spectrum extracted (see below). Same visit as 31996 — see note. |
| 31996 | SN 2018ivc | 2025-11-16 17:44–23:05 | 16.7 ks | Poonam Chandra / 503689 | **Same epoch as 31211**, not a separate one — same SEQ_NUM, same day, ~3.5 hr gap consistent with a Chandra visit split across two ObsIDs (e.g. radiation-belt interrupt). Reprocessed, extracted, and combined with 31211 into one merged spectrum 2026-09-01 (Steps 2–3.5) — see `data/Chandra/31211_31996_merged/`. |
| 20306 | SN 2018ivc | 2018-12-05 16:39–20:00 | 10.0 ks | David Pooley / 503002 | Genuinely distinct, early epoch (~1 month post-explosion). Different program than 31211/31996. Reprocessed + extracted 2026-09-01. Uses a **20306-specific source region** (not the shared canonical one) — see Step 3 note. Final: 237 src / 24 bkg counts, strong detection. |
| 29071 | NGC 1068 | 2024-01-04 17:29–21:14 | 10.3 ks | Andrea Marinucci / 705139 | **Post-explosion** (SN went off Nov 2018), ~5 yr after explosion. Serendipitous — from an AGN-monitoring program targeting the host galaxy, not a SN 2018ivc-targeted program. SN position confirmed on active chip (chip_id 7 / ACIS-S3, off-axis 0.33′, well clear of edges/gaps). Reprocessed + extracted 2026-09-01 — 64 src / 4 bkg counts, faint but clearly detected. **Combined with 29072 into one merged spectrum 2026-09-01** (Step 3.5b) — same phase to within 1.3%, rates statistically consistent (1.4σ) — see `data/Chandra/29071_29072_merged/`. |
| 29072 | NGC 1068 | 2024-01-28 17:36–21:01 | 9.3 ks | Andrea Marinucci / 705140 | Same program as 29071, ~3.5 weeks later. SN position confirmed on active chip (chip_id 7, off-axis 0.25′). Reprocessed + extracted 2026-09-01 — 45 src / 8 bkg counts, faint but detected. **Combined with 29071** — see Step 3.5b. |

So: **3 genuinely independent epochs** (20306; the merged 31211+31996 visit; the
merged 29071+29072 visit pair), spanning ~1 month to ~7 years post-explosion.
(29071/29072 are 24 days apart but both ~5 yr post-explosion — merged per the
justification in Step 3.5b, not treated as separate epochs despite being different
ObsIDs from different visits, unlike 31211/31996 which really were one interrupted
visit.)

## Step 1 — Data acquisition

Download full archive packages with CIAO's `download_chandra_obsid` (writes each
ObsID into its own `<ObsID>/` subdirectory, standard CDA layout):

```bash
conda activate ciao-4.17
cd data/Chandra/
download_chandra_obsid <obsid1>,<obsid2>,...
```

Omit file-type filters to get the *full* package (needed for `chandra_repro` — it
requires `evt1`/`flt1`/`mtl1`/`stat1`/aspect-solution files from `secondary/`, not
just the pipeline `evt2` in `primary/`). A filtered call like
`download_chandra_obsid <obsid> evt2,asol,bpix,fov,msk,pbk` only gets you pipeline
products and is not enough to reprocess.

To find ObsIDs for a target: Chandra Data Archive ChaSeR
(https://cda.harvard.edu/chaser/), searchable by target name, position, or PI/sequence
number. Header keywords `OBSERVER`, `TITLE`, `SEQ_NUM` in an existing `evt2` file (or
`oif.fits`) identify which GO program an ObsID belongs to, which is useful for finding
sibling observations in the same program.

**Status:** done for all 5 ObsIDs above.

## Step 2 — Reprocessing (`chandra_repro`)

Done for **31211 and 31996**. Not yet run for 20306, 29071, or 29072.

Command used (run from `data/Chandra/`, `ciao-4.17` env):
```bash
chandra_repro indir=./<obsid> outdir=./<obsid>/repro cleanup=no clobber=no verbose=1
```

Workflow used for 31211 (`data/Chandra/31211/repro/`):
1. `chandra_repro` on the ObsID directory → reprocessed evt2, bpix, fov, mask, mtl,
   stat files in `repro/`.
2. Background flare screening → `bg.lc`, `bg.gti`, `bg_deflare.png`,
   `evt_noflares.fits`.
3. Region definition (DS9) → `src.reg` (3.3″ circle on the source) and `bkg.reg`
   (source circle + a 9.3″ background circle). **These regions are in `physical`
   (pixel) coordinates, valid only for 31211's own WCS** — do not reuse the raw
   file against another ObsID's event list.

For 31996 (`data/Chandra/31996/repro/`, done 2026-09-01): full repro + flare
screening done, mirroring 31211's step 2 exactly:
```bash
dmextract infile="acisf<obsid>_repro_evt2.fits[bin time=::259.28]" outfile=bg.lc opt=ltc1
deflare infile=bg.lc outfile=bg.gti method=clean nsigma=3 plot=no save=bg_deflare.png
dmcopy infile="acisf<obsid>_repro_evt2.fits[@bg.gti]" outfile=evt_noflares.fits
```
No flares found in 31996 — full 16.72 ks exposure retained. Note `deflare` (unlike
most CIAO tools) has no `clobber` param — rerunning it requires removing the old
`bg.gti`/`bg_deflare.png` first.

`chandra_repro` also run for 20306, 29071, 29072 (2026-09-01) — all three exit 0,
clean, same standard bpix-session-use notice as the others. Flare screening and
extraction for these three done same day — see Step 3.

Flare screening for 20306/29071/29072 used the identical recipe as 31996 above.
Filtered exposures: 20306 9.98 ks (out of 10.0 ks — negligible flare time removed),
29071 10.01 ks (out of 10.3 ks), 29072 9.33 ks (out of 9.3 ks, i.e. no flares). No
warnings or errors in any of the three.

**Chip-coverage check (resolves earlier open item):** confirmed via `dmcoords`
(`option=cel` with the SN's RA/Dec, using each ObsID's own reprocessed evt2 + asol)
that the SN 2018ivc position lands on `chip_id` 7 (ACIS-S3) in all three, well inside
the chip (chipx/chipy mid-chip, not near edges) and at small off-axis angle
(20306: 0.11′; 29071: 0.33′; 29072: 0.25′). So all 3 are usable epochs — no chip-gap
or off-chip issue.

## Step 3 — Spectral extraction (`specextract`)

Done for **all 5 ObsIDs** (2026-09-01).

### Shared sky-coordinate regions (`data/Chandra/regions/`)

31211's original `src.reg`/`bkg.reg` were in `physical` (pixel) coordinates, valid
only for 31211's own WCS. To use the same on-sky aperture for 31996 (and any future
epoch), converted the region centers to RA/Dec via `dmcoords` (`option=sky` on
31211's repro evt2) and radii from pixels to arcsec (ACIS pixel scale 0.492"/pixel):
source circle 1.6274" radius, background circle 4.5813" radius. Canonical fk5 region
files now live in `data/Chandra/regions/`: `src_fk5.reg`, `bkg_fk5.reg` — copy these
into a new ObsID's `repro/` dir (as bare filenames, not a relative path — see gotcha
below) before extracting.

**Gotcha 1 — fk5 RA is parsed as hours, not degrees, unless suffixed `d`.** An
unqualified decimal RA in a ds9/CIAO `fk5` region file is read as *hours*
(`40.672` → 610°, wraps to nowhere near the source → silently gives **zero counts**,
no error). Always write `circle(40.672d,-0.009d,1.63")` — explicit `d` suffix on
both RA and Dec.

**Gotcha 2 — `specextract`'s auto-detection of asol/bpix/mask files breaks if the
region path passed to `infile=`/`bkgfile=` contains a relative path with `../`** (it
mis-parses the bracket-filter string and looks for ancillary files in the wrong
place, e.g. `ASOLFILE=...from regions/src_fk5.reg)][#row=0] not found`). Fix: copy
the region file into the same directory as the event file and reference it by bare
filename, and pass `asp=<asol file>` explicitly rather than relying on
auto-detection.

**Bug found and fixed 2026-09-01 — background region included the source aperture.**
31211's original `bkg.reg` (and the fk5 conversion initially made from it) was the
*union* of the small source circle and the large offset circle, not the offset circle
alone — confirmed by `BACKSCAL` (background area matched the sum of both circles'
areas) and visually in the DS9 QA snapshots (`data/Chandra/regions/qa/*.png`), which
prompted the check. This meant ~all of the source's own counts were double-counted
into the "background," making the background estimate ~20x too high. Fixed by
rewriting `bkg_fk5.reg` to contain only the offset circle, then **re-ran
`specextract` for both 31211 and 31996** with the corrected background region (source
region and all other parameters unchanged). The old union-region file is preserved as
`data/Chandra/regions/bkg_union_fk5.reg.bak` for reference.

Command used (per ObsID, from its `repro/` dir):
```bash
specextract infile="acisf<obsid>_repro_evt2.fits[sky=region(src_fk5.reg)]" \
  outroot=SN2018ivc_specextract \
  bkgfile="acisf<obsid>_repro_evt2.fits[sky=region(bkg_fk5.reg)]" \
  asp=<asol file> weight=no weight_rmf=no correctpsf=yes \
  grouptype=NUM_CTS binspec=15 bkg_grouptype=NONE bkg_binspec=""
```

The same command (with the corrected `bkg_fk5.reg`) was then run for 20306, 29071,
and 29072, copying `src_fk5.reg`/`bkg_fk5.reg` into each ObsID's `repro/` dir first
(bare filenames, per gotcha 2).

All 5 ObsIDs, final counts:

| ObsID | source region counts | background region counts | exposure |
|---|---|---|---|
| 31211 | 276 | 13 | 19.8 ks |
| 31996 | 230 | 8 | 16.7 ks |
| 20306 | 237 | 24 | 10.0 ks |
| 29071 | 64 | 4 | 10.3 ks |
| 29072 | 45 | 8 | 9.3 ks |

**20306-specific region correction (2026-09-01):** user-visual QA on
`20306_regioncheck.png` showed the shared canonical `src_fk5.reg` clipped part of the
source in 20306 specifically — its aperture centroid sits ~0.3–0.5″ north of the
shared nominal position, consistent across aperture radii 3.3–6 px (computed via a
simple event centroid in `20306/repro/acisf20306_repro_evt2.fits` around the nominal
position, energy-filtered 0.3–8 keV). This is normal inter-observation astrometric
scatter (20306 is a completely different program/PI from the one the shared regions
were built from, so its own aspect solution has an independent small zero-point
offset) — not a bug in the shared-region approach, and **not applied to the other 4
ObsIDs**, whose regions were already visually confirmed well-centered. Fix: built a
**20306-only** source region (`20306/repro/src_fk5.reg`, overwritten in place, not
`data/Chandra/regions/src_fk5.reg`) recentered on the local centroid
(RA 40.6720635944516, Dec −0.0088708255532581 — notably close to the catalog
`RA_TARG`/`DEC_TARG`) and enlarged from 1.6274″ to 2.0″ radius. Re-ran `specextract`
for 20306 only with this region (background region unchanged); counts went from 224
→ 237. User-confirmed via regenerated `20306_regioncheck.png`.

If a similar clipped-source issue shows up in a future epoch, repeat this recipe
(local centroid check within a modest aperture, then a per-ObsID region override)
rather than editing the shared canonical region files.

(Background BACKSCAL is ~7.9× the source area throughout, so background contributes
only a few counts to each source aperture — never dominant. 20306 is a strong
detection comparable to the 2025 epoch; 29071/29072 are much fainter, consistent with
the SN having faded substantially by ~5 yr post-explosion, but still clearly
detected above background.)

Products per ObsID in `<obsid>/repro/`: `SN2018ivc_specextract.pi/.arf/.rmf`
(source), `SN2018ivc_specextract_bkg.pi/.arf/.rmf` (background),
`SN2018ivc_specextract.corr.arf` (aperture-corrected ARF variant),
`SN2018ivc_specextract_grp.pi` (grouped source spectrum).

QA: `data/Chandra/regions/qa/<obsid>_regioncheck.png` for all 5 ObsIDs — ds9
snapshots (generated via `ds9 ... -saveimage png ... -exit` batch mode, ds9 runs
natively on this Mac, no virtual display needed) showing the source region centered
on the same compact source in every epoch, offset from the bright NGC 1068 nuclear
emission, with the background region in a source-free area. User-confirmed 2026-09-01
for all 5.

## Step 3.5 — Combine 31211 + 31996 into one spectrum

Done 2026-09-01. `combine_spectra` auto-detects ARF/RMF/background files from the
`ANCRFILE`/`RESPFILE`/`BACKFILE` header keywords of the input source PHAs, so only
the two corrected source `.pi` files need to be passed in:

```bash
combine_spectra \
  src_spectra=31211/repro/SN2018ivc_specextract.pi,31996/repro/SN2018ivc_specextract.pi \
  outroot=31211_31996_merged/SN2018ivc_merged \
  method=sum bscale_method=asca exp_origin=pha clobber=yes verbose=2
```

It correctly picked up the aperture-corrected ARF (`.corr.arf`, not the plain `.arf`)
for both ObsIDs since that's what `ANCRFILE` pointed to in each source PHA.

Output in `data/Chandra/31211_31996_merged/`: `SN2018ivc_merged_src.pi/.arf/.rmf`
(combined source) and `SN2018ivc_merged_bkg.pi/.arf/.rmf` (combined background,
referenced via `BACKFILE` in the source PHA).

Verified: 506 source counts (276+230 ✓), 21 background counts (13+8 ✓), exposure
36533.9 s (19815.0+16718.9 ✓, i.e. the full 36.5 ks combined). Source `BACKSCAL`
normalized to 1.0 with background `BACKSCAL` = 7.92 (the source/background area
ratio), per the `asca`-style background scaling `combine_spectra` uses.

**Not yet done:** the merged source spectrum (`SN2018ivc_merged_src.pi`) is
ungrouped — 31211/31996's individual `_grp.pi` grouping (`NUM_CTS`, 15 counts/bin)
was not reapplied to the merged file. Group before fitting (Step 4) if binning is
wanted, since grouping choices affect the fit statistic.

Alternative considered but not used: a joint/simultaneous Sherpa fit of the two
unmerged spectra instead of physically combining them (more rigorous, avoids
exposure-weighting responses together, but more setup) — revisit at Step 4 if the
combined-spectrum approach turns out to be insufficient.

## Step 3.5b — Combine 29071 + 29072 into one spectrum

Done 2026-09-01, same rationale and method as Step 3.5, but for a different reason:
29071 and 29072 are *not* one interrupted visit (different dates, 24 days apart,
same AGN-monitoring program) — they're merged because, at their shared phase
(~5 yr post-explosion), that 24-day gap is only a 1.3% fractional change in phase,
and the measured rates are statistically indistinguishable: 29071 net rate
(6.19 ± 0.78)×10⁻³ cts/s vs. 29072 (4.72 ± 0.72)×10⁻³ cts/s, a 1.4σ difference. At
this late a phase CSM-interaction X-ray emission evolves over months–years, not
weeks, so there's no physical basis to expect a real flux change between the two,
and combining trades a much better-constrained single spectrum for that epoch
against negligible averaging risk.

```bash
combine_spectra \
  src_spectra=29071/repro/SN2018ivc_specextract.pi,29072/repro/SN2018ivc_specextract.pi \
  outroot=29071_29072_merged/SN2018ivc_merged \
  method=sum bscale_method=asca exp_origin=pha clobber=yes verbose=2
```

Output in `data/Chandra/29071_29072_merged/`: `SN2018ivc_merged_src.pi/.arf/.rmf` +
`SN2018ivc_merged_bkg.pi/.arf/.rmf`. Verified: 109 source counts (64+45 ✓), 12
background counts (4+8 ✓), exposure 19581.4 s (10252.9+9328.5 ✓). `DATE-OBS` in the
merged file header is inherited from the first-listed spectrum (29071, 2024-01-04) —
used as the nominal date for this merged epoch's phase calculation.

Same caveat as Step 3.5: merged spectrum is ungrouped (`group_counts(1)` applied at
plot/fit time per the Step 4 decision, not baked into the file).

## Step 4 — Spectral fitting

Binning/statistic choice decided 2026-09-01 (see below); first round of fits
(3 models × 3 epochs) run 2026-09-01, see "Fit results" below.

### Grouping-choice QA (done 2026-09-01, before the 29071+29072 merge decision)

Compared `group_counts` at 5/10/20/50 counts/bin for the (then-4) unique epochs
before deciding on a fitting approach. Data extraction
(`data/Chandra/spectral_fitting/extract_grouping_qa.py`, run under `ciao-4.17` —
Sherpa isn't in `18ivc_clean`) loads each epoch's source PHA, `subtract()`s the
background, restricts to 0.3–8 keV, applies `group_counts` at each binning, and
dumps `get_data_plot()`'s grouped rate/energy/errors to CSV
(`data/Chandra/spectral_fitting/grouping_qa/<epoch>_group<N>.csv`).

Resulting bin counts (29071/29072 shown separately, as extracted — pre-merge):

| Epoch (total src counts) | 5 cts/bin | 10 cts/bin | 20 cts/bin | 50 cts/bin |
|---|---|---|---|---|
| 20306 (237) | 44 | 23 | 12 | 5 |
| 31211+31996 merged (506) | 87 | 47 | 24 | 10 |
| 29071 (64) | 13 | 7 | 4 | 2 |
| 29072 (45) | 9 | 5 | 3 | **1** |

After the 29071+29072 merge (Step 3.5b), `extract_grouping_qa.py`'s `EPOCHS` dict
was updated to the 3 final epochs and rerun (old `29071_group*.csv`/`29072_group*.csv`
deleted, replaced by `29071_29072_group*.csv`; e.g. `group_counts(20)` now gives 6
bins for the merged pair instead of 4 and 3 separately) — the table above is a
point-in-time record of the pre-merge comparison, not the current file layout.

### Per-epoch spectrum figure (`figure_notebooks/xray_spectra_by_epoch.ipynb`)

Repurposed from an earlier `xray_grouping_comparison` notebook (deleted, along with
its `figures/xray_grouping_comparison.png/.pdf`) at the user's request: rather than
comparing binnings, this figure shows one spectrum per unique epoch, one panel per
epoch, ordered chronologically by phase post-explosion. Phase computed from each
merged/unmerged source PHA's `DATE-OBS` header vs. explosion epoch MJD 58445.0
(Maeda et al. 2023a); title format
`"<obsid(s)> (<date>, ~<N> days post-explosion)"` (standardized to days across all
panels — an earlier months/years version was replaced at the user's request).
Output: `figures/xray_spectra_by_epoch.png/.pdf`. Currently 3 panels (1×3 layout,
chronological left to right): 20306 (~13 days), 29071+29072 merged (~1869 days),
31211+31996 merged (~2550 days). Grouping was `group_counts(1)` initially, then
updated to **`group_counts(15)`** (2026-09-01) to match the final fitting grouping
decided above — much cleaner spectral shape visible per panel than the
single-count-per-bin version. Update this notebook (not a new one) if another epoch
is added/merged, or the grouping choice changes again — reread
`CLAUDE_plotting.md`/the `18ivc-plotting` skill first, per the "revising an existing
figure" workflow.

### Fitting statistic decision (2026-09-01)

At these count levels, coarse grouping for chi2 fitting is the wrong tool — chi2
needs roughly Gaussian per-bin errors (rule of thumb ≳20–25 counts/bin), which the
50 cts/bin column above makes obviously untenable for 29071/29072 (collapses to 1–2
bins, destroying essentially all spectral shape information for exactly the epochs
that need it most). **Decision: use a Poisson-likelihood statistic (Cash-family) for
all epochs**, not chi2. C-stat/wstat stay unbiased down to very low counts, and
using one consistently across all epochs (rather than switching statistic by epoch
brightness) keeps the fitting method consistent project-wide. (Decided when there
were 4 unique epochs; still applies unchanged now that 29071+29072 are merged
into 3.)

**Practical correction found when actually fitting (2026-09-01): use `wstat`, not
`cstat`.** Sherpa refuses `cstat` on background-subtracted data
(`FitErr: cstat statistics cannot be used with background subtracted data`) — cstat
assumes the data being fit are themselves Poisson-distributed counts, which
background-subtracted values aren't. The fix isn't to go back to chi2: `wstat` is
the standard Cash-family statistic built for exactly this case (Poisson source +
Poisson background, fit jointly without subtracting), so the fits load the PHA
*without* `subtract()` and use `set_stat("wstat")` instead. This is still the same
"unbiased Poisson likelihood, no chi2 Gaussian-approximation bias" approach the
original decision was about — just the specific Sherpa statistic name.

**Grouping revisited 2026-09-01, after the 29071+29072 merge raised that epoch to
109 counts.** With more counts, the user asked about chi2 with 10-20 cts/bin.
Key clarification that resolved this without abandoning C-stat: **grouping level and
fit statistic are independent choices.** The chi2 bias concern above is specifically
about approximating Poisson counts as Gaussian — it doesn't apply to C-stat
regardless of how coarsely the data is grouped, since C-stat computes an exact
Poisson likelihood on whatever counts land in each bin (grouped or not). So a
courtesy grouping in the 10-20 range is safe under C-stat purely as a
resolution/plotting convenience, without reintroducing the bias that made chi2 risky
in the first place.

**Final decision: `group_counts(15)` + wstat**, for all 3 epochs — splits the
10-20 range, gives comfortable bin counts (20306: 16 bins; 29071+29072: 8 bins;
31211+31996: 32 bins) without discarding much resolution. This grouping is what's
used both for the `xray_spectra_by_epoch` figure (above) and the fits below.

### Fit results (first round, 2026-09-01)

At the user's request, fit **3 spectral models × 3 epochs = 9 fits**, to compare
across models rather than pre-committing to one. All 9 use: `group_counts(15)`,
0.3–8 keV, `wstat`, N_H frozen at the Galactic value toward SN 2018ivc —
**2.6×10²⁰ cm⁻² (0.026 in `tbabs` units of 10²² cm⁻²)**, looked up via HEASoft's
`nh` tool (HI4PI survey) at RA 40.672 / Dec −0.0089:
```bash
export HEADAS=<repo>/data/Chandra/heasoft-6.36/aarch64-apple-darwin24.6.0
source $HEADAS/headas-init.sh
printf "2000\n40.672\n-0.0089\n" | nh
```
N_H free was not attempted — with 2–3 free parameters already and only 109–506
counts per epoch, adding a 3rd/4th free parameter (N_H) was judged unlikely to be
meaningfully constrained; revisit if a future epoch has enough counts to support it.

Models (all via Sherpa/XSPEC components, `data/Chandra/spectral_fitting/fit_models.py`):
- **`tbabs*apec`** — absorbed single-T thermal plasma. Abundance frozen at solar,
  redshift frozen at 0.003793. Free: kT, norm.
- **`tbabs*powerlaw`** — absorbed simple power law. Free: PhoIndex (Γ), norm.
- **`tbabs*bremss`** — absorbed thermal free-free continuum. Free: kT, norm.
  (`bremss`/`powerlaw` have no redshift parameter in XSPEC — fine, z=0.0038 is a
  negligible continuum-shape correction at this resolution, not worth a
  `zbremss`/`zpowerlw` swap.)

Full results (`data/Chandra/spectral_fitting/fits/fit_summary.csv`; per-model folded
model curves in `fits/<model>/<epoch>_model.csv`):

| Model | Epoch | Free param 1 | Free param 2 (norm) | W-stat/dof | Fit quality |
|---|---|---|---|---|---|
| apec | 20306 | kT = 64.0 keV (**pegged at hard max**) | 2.58e-4 | 107.3/14 | Poor — `conf()` refused (rstat 7.7 > Sherpa's guard of 3) |
| apec | 29071+29072 | kT = 28.2 keV (essentially unconstrained: −18.2/+∞) | 6.90e-5 (+2.06e-5/−1.15e-5) | 1.1/6 | Formally fine but uninformative on kT |
| apec | 31211+31996 | kT = 7.51 (+3.00/−1.27) keV | 1.567e-4 (±7e-6) | 36.4/30 | Reasonable, physically plausible shock temperature |
| powerlaw | 20306 | Γ = 0.19 (±0.13) | 1.49e-5 | 31.8/14 | Better than thermal models, still not great (very hard index) |
| powerlaw | 29071+29072 | Γ = 1.33 (+0.20/−0.19) | 1.40e-5 | 1.0/6 | Fine, unremarkable |
| powerlaw | 31211+31996 | Γ = 1.59 (+0.10/−0.08) | 4.54e-5 | 38.2/30 | Good fit |
| bremss | 20306 | kT = 200 keV (**pegged at hard max**) | 1.22e-4 | 98.9/14 | Poor, same pattern as apec |
| bremss | 29071+29072 | kT = 39.3 keV (essentially unconstrained: −26.6/+∞) | 2.57e-5 | 1.2/6 | Formally fine but uninformative on kT |
| bremss | 31211+31996 | kT = 8.61 (+2.88/−1.80) keV | 5.89e-5 | 34.4/30 | Good fit, slightly best rstat of the 3 models for this epoch |

Comparison figures (data points + best-fit curve, 1×3 chronological panels, same
data/style as `xray_spectra_by_epoch`): `figure_notebooks/xray_spectra_apec_fit.ipynb`,
`xray_spectra_powerlaw_fit.ipynb`, `xray_spectra_bremss_fit.ipynb` →
`figures/xray_spectra_{apec,powerlaw,bremss}_fit.png/.pdf`.

**Advisor guidance 2026-09-02 — add residual panels: done 2026-09-02.** Each
per-epoch panel in the three fit-comparison figures now has a data−model residuals
sub-panel plotted directly below the fitted spectrum (same per-epoch column, shared
x-axis). Implementation:

- `data/Chandra/spectral_fitting/fit_models.py` (run under `ciao-4.17`, needs Sherpa)
  extended to also call `get_resid_plot(1)` after each fit and dump
  `fits/<model>/<epoch>_resid.csv` (`energy_kev`, `energy_err_kev`, `resid`,
  `resid_err`), at the same `group_counts(15)` bin resolution as the data. This is
  **not** the same as differencing the existing `_model.csv` curve against the data —
  that CSV is a fine unbinned model curve (~528 points from `get_model_plot`), while
  residuals need the model folded at the actual fit bins; `get_resid_plot` does that
  correctly. `ResidPHAPlot` exposes bin edges as `xlo`/`xhi`, not a symmetric `xerr`
  — half-width computed as `(xhi-xlo)/2` to match the data errorbar convention.
  Rerunning the script reproduced the exact same `statval`/`dof` as the original fit
  round (fully deterministic re-fit), so `fit_summary.csv` and the `_model.csv`
  curves are unchanged, just regenerated alongside the new `_resid.csv` files.
- All three notebooks (`xray_spectra_{apec,powerlaw,bremss}_fit.ipynb`) edited in
  place: plotting cell switched from `plt.subplots(1,3,...)` to a `GridSpec(2,3,
  height_ratios=[3,1])`, spectrum row on top (sharex/sharey across the 3 epoch
  columns, as before) and a residuals row below (sharex per column with its
  spectrum panel, **not** sharey across epochs — residual amplitude varies ~30x
  between the faint 29071+29072 epoch and the others, so a shared y-axis would flatten
  it unreadably). Residual points use the same per-epoch viridis color as the data
  points; a `axhline(0)` marks the zero line. Re-executed under `18ivc_clean` via
  `jupyter nbconvert --execute --inplace`.
- Visual confirmation: the 31211+31996 residual panel in the power-law figure clearly
  shows the ~2.9 keV and ~6.7 keV residual bumps that motivate the "improve final
  epoch fit" advisor guidance below — consistent with that plan.

Output: `figures/xray_spectra_{apec,powerlaw,bremss}_fit.png/.pdf` regenerated.

**Interpretation / open questions for next session:**
- **20306 (~13 days, brightest per-exposure epoch, 16 bins): both thermal models
  fail outright** — kT runs to its hard parameter-space boundary in both apec (64
  keV) and bremss (200 keV), i.e. the fit wants a temperature hotter than the model
  grid supports, and W/dof ≈ 7 is a bad fit either way. Power law fits
  noticeably better (W/dof ≈ 2.3) but with an unusually flat/hard index
  (Γ ≈ 0.19 — most astrophysical hard-continuum sources are Γ ≳ 1). A hard index
  like this is often a sign of **intrinsic (host/CSM) absorption being
  under-modeled** (fixed N_H here is Galactic-only) suppressing the soft end and
  mimicking hardness, rather than the source truly being that hard. Worth
  revisiting with intrinsic N_H free (or a 2nd absorber) specifically for this
  epoch once there's appetite to add a 3rd free parameter, and/or a two-temperature
  thermal model.
- **29071+29072 (~1869 days, 8 bins, 109 counts): all 3 models fit "fine" (W/dof
  ≲ 1) but none meaningfully constrain their shape parameter** — kT/Γ uncertainties
  are huge or one-sided-unbounded. Can't yet discriminate emission mechanism at
  this epoch; more counts (a future epoch, or reconsidering exposure) would help
  more than model choice does.
- **31211+31996 (~2550 days, 32 bins, 506 counts, best-constrained epoch): all 3
  models fit comparably well** (W/dof 1.15–1.27) — apec kT ≈ 7.5 keV and bremss
  kT ≈ 8.6 keV agree well with each other (as expected, similar physics), powerlaw
  Γ ≈ 1.59 is unremarkable. Not possible to statistically prefer one model over
  another from goodness-of-fit alone at this S/N; model choice for this epoch
  should probably be driven by physical expectation (CSM-interaction shocked
  plasma → thermal is the more motivated choice) rather than fit statistics.
- No formal model-comparison statistic (e.g. AIC/BIC, or an F-test analog) has been
  computed yet — the table above is goodness-of-fit per model, not a ranking.

**Not yet done:** flux/luminosity conversion (Step 5) from any of these fits —
explicitly deferred, do not start without being asked.

### Collaborator guidance 2026-09-03 — free N_H and a bremss+powerlaw combined fit

Done 2026-09-03. Discussion with a collaborator raised two follow-ups to the first
fitting round above:

- **Let N_H vary, but only for the 20306 epoch (~13 days post-explosion).** At the
  user's direction, N_H (tbabs) was thawed for **20306 only**, for all 4 models
  (apec, powerlaw, bremss, and the new bremss+powerlaw below); the other two epochs
  (29071+29072, 31211+31996) keep N_H frozen at the Galactic value
  (2.6×10²⁰ cm⁻²), matching the first round. Implemented in `fit_models.py` via a
  per-epoch `_set_nh()` helper (`FREE_NH_EPOCH = "20306"`) shared by all 4 model
  builders.
- **Combined bremss + powerlaw model, `tbabs*(bremss+powerlaw)`.** Added as a 4th
  model — a thermal free-free continuum plus a nonthermal power-law continuum fit
  together, rather than the 3 single-continuum models from the first round. The
  powerlaw component represents nonthermal synchrotron emission, which could
  originate from either a PWN (pulsar wind nebula, if the SN left behind a young
  pulsar) or CSM shock interaction (particle acceleration at the forward/reverse
  shock) — the fit itself doesn't distinguish which, but it's the physical
  motivation for the nonthermal component. Free: `brem1.kT`, `brem1.norm`,
  `pl1.PhoIndex`, `pl1.norm` (+ `abs1.nH` for 20306 only, per above).

**Fitting complication found and fixed: bremss+powerlaw is prone to local minima.**
A single-start `levmar` fit for 31211+31996 converged to W-stat=53.2/28 with the
power-law norm driven to ~0 (effectively degenerating to a poor bremss-only fit) —
but Sherpa's `conf()` call (run afterward for error bars) stumbled onto a
genuinely better minimum, W-stat=34.4/28, while searching, without the script
capturing it (the row was built from `get_fit_results()` right after the initial
`fit()`, before `conf()` ran and silently relocated the model to the better point —
so the recorded params and the recorded confidence intervals were briefly out of
sync with each other). Fixed by adding a **multi-start search** to
`fit_models.py`: for `bremss_powerlaw` only, `levmar` is run from 9 starting points
(`kT0` ∈ {1, 5, 15} keV × `PhoIndex0` ∈ {−1, 1, 3}), the lowest-statistic result is
kept, and *that* is what gets refit-and-recorded (so `get_fit_results()`/`conf()`
downstream operate on the true minimum). After the fix, 31211+31996 converges
cleanly to W-stat=34.4/28 with no "New minimum statistic found" warning, and the
other two epochs' bremss_powerlaw fits also improved slightly (20306: 38.5→31.9;
29071+29072: 0.89→0.85), i.e. the single-start version had likely been missing the
true minimum there too, just less dramatically.

**Full results, all 12 fits (4 models × 3 epochs):**

| Model | Epoch | W-stat/dof | rstat |
|---|---|---|---|
| apec | 20306 | 39.9/13 | 3.07 |
| apec | 29071+29072 | 1.1/6 | 0.19 |
| apec | 31211+31996 | 36.4/30 | 1.21 |
| powerlaw | 20306 | 31.7/13 | 2.44 |
| powerlaw | 29071+29072 | 1.0/6 | 0.17 |
| powerlaw | 31211+31996 | 38.2/30 | 1.27 |
| bremss | 20306 | 39.8/13 | 3.06 |
| bremss | 29071+29072 | 1.2/6 | 0.20 |
| bremss | 31211+31996 | 34.4/30 | 1.15 |
| bremss_powerlaw | 20306 | 31.9/11 | 2.90 |
| bremss_powerlaw | 29071+29072 | 0.8/4 | 0.21 |
| bremss_powerlaw | 31211+31996 | 34.4/28 | 1.23 |

**Interpretation:**
- **20306, free N_H:** substantially improves both thermal fits versus the
  frozen-N_H first round (apec rstat 7.7→3.07, bremss rstat ~7→3.06) — both still
  peg at their hard kT bound (apec 64 keV, bremss 200 keV) and both still fail
  Sherpa's rstat>3 guard for `conf()`, so still not formally acceptable fits, but
  clearly less bad. Both prefer a large N_H (apec 1.82×10²² cm⁻², bremss
  1.71×10²² cm⁻² — ~70× Galactic) to help fit the shape. **Powerlaw behaves
  oppositely:** its free N_H is pushed to the pegged minimum (0), i.e. the
  power-law fit wants *less* absorption than Galactic, not more — and remains the
  best single-continuum fit for this epoch (rstat 2.44, up slightly from 2.27 in
  the frozen-N_H round — statval itself barely moved, 31.8→31.7, so the rstat
  increase is just the extra free parameter cutting dof by 1; freeing N_H bought
  essentially no fit-quality improvement for powerlaw here, it just relocated N_H
  to 0 without changing the fit). **bremss_powerlaw for 20306** finds a genuinely
  different, non-pegged solution (kT=8.27 keV, Γ=0.10, N_H≈0.001×10²² ≈ 0, i.e.
  also pushed toward zero absorption) — its thermal component no longer runs to
  the hard boundary the way apec/bremss alone do, but its rstat (2.90) is still
  worse than powerlaw alone (2.44), so **powerlaw alone remains the best-fitting
  model for 20306** even after these changes.
- **29071+29072:** all 4 models fit "fine" (rstat ≲ 0.2) but still don't
  meaningfully constrain their shape parameters (huge or one-sided/unconstrained
  errors) — unchanged conclusion from the first round; this epoch's 109 counts
  just isn't enough to discriminate models regardless of which of these 4 is
  tried.
- **31211+31996 (best-constrained epoch): bremss_powerlaw ties bremss-alone as the
  best fit** (W-stat 34.4 vs 34.4/34.35 — statistically indistinguishable), despite
  bremss_powerlaw having 2 more free parameters (dof 28 vs 30). That the extra
  power-law component buys essentially **zero** improvement in fit quality here is
  itself informative: it means this epoch's residual structure (the ~2.9/6.7 keV
  bumps flagged in the 2026-09-02 advisor guidance below) looks like **discrete
  emission lines**, not a smooth added nonthermal continuum — consistent with
  going the Gaussian-line route (next section) rather than the two-continuum
  route for improving this epoch's fit. bremss_powerlaw's best-fit values here
  (kT=8.12 keV, Γ=1.28) sit between the bremss-alone (kT=8.61 keV) and
  powerlaw-alone (Γ=1.59) single-model results, as expected for a blend, but with
  large, strongly correlated uncertainties on both (kT +3.6/−3.6, Γ unconstrained
  on the upper side) — the two components are not well separated by the data,
  another sign this model isn't earning its extra complexity for this epoch.
- No formal model-comparison statistic (AIC/BIC/F-test) has been computed for any
  of these comparisons — all of the above is goodness-of-fit (W-stat/rstat)
  reasoning, not a formal statistical preference test.

**Figures:** all 3 first-round notebooks (`xray_spectra_{apec,powerlaw,bremss}_fit.ipynb`)
regenerated with the updated (free-N_H-for-20306) fits — each panel's annotation
now shows the fitted N_H value for the 20306 panel only (the other two panels don't
have N_H as a free parameter, so no N_H line is shown there). A new 4th notebook,
`xray_spectra_bremss_powerlaw_fit.ipynb`, was added for the combined model, same
1×3 chronological-panel + residuals-row layout and viridis per-epoch coloring as
the other three. All 4 re-executed via `jupyter nbconvert --execute --inplace`
under `18ivc_clean`. Outputs:
`figures/xray_spectra_{apec,powerlaw,bremss,bremss_powerlaw}_fit.png/.pdf`.

### Advisor guidance 2026-09-02 — improving the final-epoch (31211+31996) fit

The final epoch (31211+31996, ~2550 days post-explosion, best-constrained at 506
counts) shows residuals in the pure power-law fit consistent with thermal emission
lines — a pure synchrotron power law alone doesn't fully capture the spectral shape
there. Advisor-directed procedure to improve wstat:

1. Identify the line(s) by energy using the X-ray data booklet's emission-line energy
   chart (compare the residual line energy against known transition energies).
2. Add a Gaussian component (`xsgaussian`/`zgauss`-equivalent) at each identified
   line's energy, on top of the existing continuum model.
3. Fix the Gaussian's energy centroid at the identified line energy, then vary its
   other free parameters (width, normalization) to find the combination that gives
   the greatest improvement to wstat.

Not yet started — documenting the plan per advisor meeting 2026-09-02.

**Next step, scoped 2026-09-03 (not yet started):** add a Gaussian at the ~2.9 keV
residual feature (visible in the power-law fit's residual panel, per the 2026-09-02
visual confirmation above) to **3 of the 4 models — powerlaw, bremss, and
bremss_powerlaw — for the 31211+31996 epoch only**. (apec is excluded: adding a
line on top of an already-multi-line thermal plasma model is a different exercise
than adding one to the continuum-only models, and wasn't asked for.) Only this one
epoch gets the treatment — 20306 and 29071+29072 aren't included. Follow the
3-step procedure above (identify energy via the data booklet, add
`xsgaussian`/`zgauss` with centroid fixed at that energy, vary width/norm) for each
of the 3 models separately, and compare the wstat improvement each gets from adding
the line. Given the bremss_powerlaw finding just above (that model already ties
bremss-alone with *no* wstat benefit from its extra nonthermal component at this
epoch), this is a natural next test: if a single Gaussian line closes most of the
gap for all 3 continuum choices, that's further evidence the residual structure is
line emission rather than a missing continuum component, regardless of which
continuum is chosen underneath it.

## Step 5 — Flux / luminosity conversion (not started)

Not yet done. Depends on Step 4 fit results plus the adopted distance/redshift to the
host (NGC 1068, z = 0.003793 — see [[project_sn2018ivc]] memory).

### Methodology (advisor guidance 2026-09-02)

- **Per-epoch model choice:** derive the luminosity at each epoch using whichever
  spectral model best fits *that* epoch (per the Step 4 model comparison), not one
  model forced uniformly across all epochs.
- **Tooling:** either XSPEC's `lum` command (specify the energy range and supply the
  redshift, z = 0.003793) or PIMMS on HEASARC are acceptable for the flux→luminosity
  conversion.
- **Expected model-independence:** the derived luminosity should not change much
  regardless of which model is used, since it's essentially the area under the
  spectral curve (integrated flux) rather than something sensitive to the detailed
  shape. Before committing to final per-epoch numbers, run a sanity-check test on a
  single epoch — compute luminosity from more than one of the fitted models for that
  epoch and confirm they agree — to demonstrate this model-independence.
- **Final numbers:** once the sanity check is done, use the best-fitting model per
  epoch (not an arbitrary/uniform choice) for the adopted luminosity values.

Not yet started — documenting the plan per advisor meeting 2026-09-02.

## Step 6 — Light curve modeling (`redback-csm`), not started

Once the X-ray light curve is constructed (fluxes/luminosities per epoch from Step
5), light curve modeling will be done with
[`redback-csm`](https://github.com/nikhil-sarin/redback_csm) (Sarin & Hirai 2026,
arXiv:2605.19571) — Fortran-based CSM-interaction models plugged into the
[`redback`](https://github.com/nikhil-sarin/redback) transient-modeling/Bayesian
inference package (Sarin et al. 2024, arXiv:2308.12806). Both packages are cloned
locally at `redback/` and `redback-csm/` (untracked in this repo as of 2026-09-03 —
not yet added under version control or set up in a conda env). `redback-csm` model
names follow `{outer_CSM}_{inner_ejecta}` (outer = older progenitor-laid-down CSM
density profile, inner = the more recent transient ejecta profile); once installed,
its models register into redback's model library and are used for inference the same
way as redback's built-in models.

Not yet started — env setup, model selection, and fitting all still to do. Depends on
Step 5 (flux/luminosity conversion) being done first to have a light curve to fit.

## Open items / TODO

- First round of fitting is done: 3 models (apec/powerlaw/bremss) × 3 epochs, wstat,
  `group_counts(15)`, N_H fixed at Galactic — see Step 4 "Fit results" for the full
  table and per-epoch interpretation.
- Follow-ups flagged by the first round, not yet done: (1) 20306's thermal-model fits
  peg at their hard kT boundary and fit poorly — worth trying free intrinsic N_H (or
  a 2nd absorber) for that epoch specifically; (2) no formal model-comparison
  statistic (AIC/BIC or similar) computed yet, so "which model is best" per epoch is
  only informal so far.
- Flux/luminosity conversion (Step 5) — explicitly deferred by the user; do not
  start without being asked. Methodology now documented (advisor guidance
  2026-09-02): best-fitting model per epoch, via XSPEC `lum` or PIMMS, with a
  single-epoch cross-model sanity check first to confirm luminosity is
  model-independent.
- Final-epoch (31211+31996) fit improvement — not yet started: identify thermal
  line residuals via the X-ray data booklet's emission-line chart, add a Gaussian
  per line with fixed centroid, vary width/norm to maximize wstat improvement (see
  Step 4 "Advisor guidance 2026-09-02"). **Scoped 2026-09-03, not yet started:** add
  a Gaussian at the ~2.9 keV feature to powerlaw, bremss, and bremss_powerlaw (not
  apec) for the 31211+31996 epoch only, and compare the wstat improvement each
  model gets — see Step 4 "Advisor guidance 2026-09-02 — improving the final-epoch
  (31211+31996) fit" for the full scoping.
- Residual panels for the fit-comparison figures — done 2026-09-02 (see Step 4
  "Advisor guidance 2026-09-02 — add residual panels: done").
- Collaborator guidance 2026-09-03 — done: (1) N_H freed for the 20306 epoch only
  (frozen elsewhere), across all 4 models; (2) added a combined
  `tbabs*(bremss+powerlaw)` 4th model. Found and fixed a local-minimum degeneracy
  in the combined model (multi-start search now used for it). Key result: for
  31211+31996 the combined model ties bremss-alone, meaning the extra nonthermal
  component buys no fit improvement there — supports the residuals being discrete
  emission lines (see the Gaussian-line plan above) rather than a missing
  continuum component. All 4 fit-comparison figures regenerated. See Step 4
  "Collaborator guidance 2026-09-03" for full results.
- Light curve modeling with `redback-csm` (Step 6) — planned for once the X-ray
  light curve exists (after Step 5); not started, packages cloned locally but not
  yet set up.
