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

**Done 2026-09-03 — Gaussian component added, with scope changed at the user's
request from the plan above:** no X-ray data booklet species identification, and
the ~6.7 keV bump was dropped entirely (not visible in the residuals at
`group_counts(15)`, so presumably not significant). Instead of fixing the
centroid at a booklet-identified energy, the Gaussian's energy, width (`Sigma`),
and norm were all left free (seeded near 2.9 keV), so the fit finds its own
best-fit line energy — done as a separate variant alongside a fixed-at-2.9-keV
comparison run, both for the same 3 models (powerlaw, bremss, bremss_powerlaw) on
the 31211+31996 epoch only. Implementation:
`data/Chandra/spectral_fitting/fit_models_gauss.py` (run under `ciao-4.17`),
built on a small refactor of `fit_models.py` (its per-model-per-epoch fitting body
extracted into a shared `fit_one()` helper, and `BREMSS_POWERLAW_STARTS` changed
from `(kT0, gamma0)` tuples to generic kwargs dicts) so the new script can reuse
the exact same builders, epoch paths, and multi-start machinery rather than
duplicating them — verified bit-identical to the pre-refactor output on the
original 12 fits before building on top of it.

**Two robustness problems found and fixed while doing this fit (both by empirical
multi-start scans, not assumed):**

1. **Local minima in (LineE, Sigma) jointly, not just LineE.** An initial
   multi-start that varied only the Gaussian's seed energy (Sigma always seeded at
   0.1 keV) missed genuinely deeper minima only reachable from specific
   (LineE0, Sigma0) *combinations* — e.g. bremss_powerlaw_gauss's true best
   free-centroid minimum (wstat=25.19) was invisible to a LineE-only scan and only
   turned up once Sigma0 was also varied. Fixed by multi-starting over the full
   (LineE0 × Sigma0) grid — `LINE_E_STARTS = [1.0, 1.9, 2.9, 4.5, 6.0, 6.7]` ×
   `SIGMA_STARTS` — for the free-centroid case, and over `SIGMA_STARTS` alone
   (LineE frozen at 2.9) for the fixed case. This is the same class of problem as
   the bremss_powerlaw local-minimum issue documented above, just in a different
   parameter pair.
2. **Unresolvably narrow "lines" fitting single-bin noise.** Without a width
   floor, the deepest minima for bremss_gauss (fixed-centroid) and
   bremss_powerlaw_gauss (free-centroid) converged to Sigma ~ 0.001–0.02 keV — far
   narrower than Chandra ACIS's actual energy resolution (FWHM ≈ 130 eV at these
   energies → Sigma ≈ 0.055 keV via FWHM/2.355) — i.e. the fit was using an
   essentially infinitely-narrow spike to absorb one noisy bin's excess counts,
   not modeling a real, instrument-resolvable line. Fixed per user direction:
   `gau1.Sigma.min = 0.055` (keV) imposed for every fit, so the optimizer
   physically cannot converge to a sub-resolution width — any wstat improvement
   that survives this floor reflects a real feature, not overfitting.

**Full results, both variants, 31211+31996 only** (`fits/fit_summary_gauss.csv`
free-centroid, `fits/fit_summary_gauss_fixed2p9.csv` fixed-at-2.9-keV):

| Model | Centroid | W-stat/dof | ΔW-stat vs. no-gauss | LineE | Sigma | Verdict |
|---|---|---|---|---|---|---|
| powerlaw | free | 27.80/27 | 10.40 | 1.73 keV | 0.86 keV (broad) | Not a line — broad, off-target, more like continuum curvature |
| powerlaw | fixed @2.9 | 38.19/28 | 0.01 | 2.9 (frozen) | 12.99 keV (pegged near max) | No improvement at all — Gaussian spreads out to contribute nothing |
| bremss | free | 24.85/27 | 9.50 | 2.957±0.032 keV | 0.055 keV (resolution-limited) | **Significant, narrow line right at the expected energy** |
| bremss | fixed @2.9 | 27.95/28 | 6.40 | 2.9 (frozen) | 0.055 keV (resolution-limited) | Confirms the free-centroid result |
| bremss_powerlaw | free | 25.03/25 | 9.41 | 2.958±0.034 keV | 0.055 keV (resolution-limited) | Same narrow line, consistent across continuum choice |
| bremss_powerlaw | fixed @2.9 | 28.15/26 | 6.29 | 2.9 (frozen) | 0.055 keV (resolution-limited) | Confirms the free-centroid result |

**Interpretation:** `powerlaw` gains nothing physically meaningful from a Gaussian
at or near 2.9 keV — its best free-centroid minimum is broad and drifts to
~1.7 keV (continuum-shape compensation, not a line), and forcing the centroid to
2.9 keV finds essentially zero improvement. `bremss` and `bremss_powerlaw` both
show a **significant, narrow, resolution-limited line at ~2.96 keV** (free
centroid) that's fully consistent with the fixed-at-2.9-keV result and with each
other — the fitted Sigma lands exactly at the 0.055 keV instrumental floor in
every one of these 4 fits, meaning the true line width is at or below what
Chandra can resolve, not that the fit is cheating with an artificially narrow
spike (that possibility was specifically what the resolution floor was added to
rule out). Line normalization is nonzero at ~3σ in the free-centroid fits
(`gau1.norm` vs. its `conf()` lower bound). ΔW-stat of ~9-10 for 3 extra free
parameters (free centroid) is suggestive but not highly significant on its own
(roughly 2-2.5σ via Wilks' theorem) — worth keeping in mind before over-claiming a
firm detection, though the cross-model and free-vs-fixed consistency is
reassuring.

Figure: `figure_notebooks/xray_spectra_31211_31996_gauss_fit.ipynb` →
`figures/xray_spectra_31211_31996_gauss_fit.png/.pdf` — 3 panels (powerlaw,
bremss, bremss_powerlaw), single epoch, data + no-gauss (dashed) + free-centroid
gauss (solid) curves, residuals row below using the free-centroid fit's residuals.
Colors: single epoch color (viridis x=0.100, same as the other 31211+31996 panels
project-wide) for data, two `darken_color` shades to distinguish the no-gauss vs.
with-gauss curves (matching `CLAUDE_plotting.md`'s no-arbitrary-color rule) rather
than linestyle-only or an unrelated color. Visually confirms the fit table: a
clear narrow bump at ~2.9-3 keV in the solid curve for the bremss/bremss_powerlaw
panels, and near-total overlap of the two curves in the powerlaw panel.

**Not done:** no formal significance test beyond the ΔW-stat/Wilks'-theorem
estimate above (e.g. a proper simulation-based null distribution for the line
norm). A Protassov et al. (2002)-style Monte Carlo significance test (parametric
bootstrap: simulate from the best-fit bremss-only model, refit null vs.
bremss+gauss on each fake spectrum, compare the real ΔW-stat=9.50 against that
null distribution — the theoretically correct approach here, since the line
energy is an unidentified-under-the-null nuisance parameter that breaks the
Wilks'-theorem chi2 assumption) was started 2026-09-04
(`data/Chandra/spectral_fitting/gauss_significance_test.py`) but **deprioritized
by the user before completion** — not resumed. If revisited, the script and
method are ready to rerun.

**Model choice for this epoch, decided 2026-09-04: bremss+gauss (free centroid)**
adopted going forward for 31211+31996, without waiting on the formal significance
test above — kT=8.61 keV continuum, line at ~2.96 keV, W/dof=24.85/27. Chosen over
bremss_powerlaw_gauss (statistically near-identical, W/dof=25.03/25) since the
extra power-law component isn't earning its keep here (see the 2026-09-03
collaborator-guidance finding above that bremss_powerlaw ties bremss-alone at this
epoch), and over plain bremss/powerlaw since the line addition gives a real
W-stat improvement. This is the model to carry forward into Step 5
(flux/luminosity) for this epoch once that step is started.

## Step 5 — Flux / luminosity conversion (done 2026-09-04)

Depends on Step 4 fit results plus the adopted distance/redshift to the host
(NGC 1068, z = 0.003793 — see [[project_sn2018ivc]] memory).

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

### Done 2026-09-04 — implementation and results

**Superseded 2026-09-07/08 — see the update further below.** Tool changed from
Sherpa's `sample_flux` to PyXspec's `lum`/`flux`, distance changed from the
Cepheid 10.72 Mpc to Maeda et al. 2023b's 10 (+1.8/-1.5) Mpc, and band changed
from 0.3-8 keV to 0.3-10 keV. The numbers in this subsection are kept for
history but are no longer the adopted ones — use the "Redone 2026-09-07/08"
subsection's table instead.

Per-epoch adopted models come from the user's `xray_epoch_spec_models.xlsx`
(2026-09-04) — see the "Best-fit model per epoch" figure below and the model-choice
summary earlier in this Step: **20306** → powerlaw (N_H frozen at Galactic);
**29071+29072** → bremss_powerlaw; **31211+31996** → bremss+free-centroid gauss.

**Tooling:** Sherpa has no direct `lum`-command equivalent, so used
`sherpa.astro.ui.sample_flux` instead — the modern, more rigorous analog (same
underlying physics: integrate flux over a band, convert via distance), with
parameter-covariance Monte Carlo sampling (`num=2000`) for 1-sigma bounds rather
than `lum`'s simpler error propagation. `modelcomponent` passed to `sample_flux`
excludes the `tbabs` absorber (just the continuum(+line) piece), giving the
**unabsorbed** (intrinsic) flux directly. Band: **0.3–8 keV observer-frame**,
matching the `notice()` range used throughout this project's X-ray fitting (not
`redback-csm`'s 0.3–10 keV default — that model's band edges are adjustable at
Step 6 fit time, so this choice isn't locked in for later).

**Distance: 10.72 Mpc** (direct Cepheid P-L distance to NGC 1068, 38 candidates,
arXiv:2602.22407, D=10.72±0.52 Mpc; cross-validated by an independent TRGB
distance of 11.14±0.54 Mpc) — **not** the naive Hubble-flow redshift distance
(~16.2 Mpc at z=0.003793 with H0=70), which overstates the true distance because
NGC 1068 has a significant peculiar velocity. This is the same distance already
adopted for the radio luminosity comparison (`data/sn_reference_crosswalk.xlsx`,
"Figure 8 References" sheet, decision recorded 2026-08-27) — using it here keeps
X-ray and radio luminosities on a consistent footing. L = 4πD_L²F; the ±4.9%
distance uncertainty is a **global multiplicative systematic (±9.7% in L)**,
reported separately, not folded into the per-epoch statistical errors below.

**Model-independence sanity check (31211+31996, the best-constrained epoch):**
computed luminosity from 3 models — the adopted bremss_gauss, plain bremss, and
bremss_powerlaw:

| Model | Luminosity (erg/s) |
|---|---|
| bremss_gauss (adopted) | 3.838×10³⁹ |
| bremss | 3.886×10³⁹ |
| bremss_powerlaw | 6.747×10³⁹ |

bremss_gauss and bremss agree to **~1.2%** — exactly the model-independence the
advisor guidance expects, and it directly validates the adopted number (adding a
narrow line barely changes the broadband integrated flux, as expected).
**bremss_powerlaw diverges by ~76%**, but this is not a failure of the
model-independence expectation in general — it's a symptom of the parameter
degeneracy in that model already flagged in the 2026-09-03 collaborator-guidance
section above (the thermal and nonthermal components trade off against each other
with large, correlated, partly-unconstrained uncertainties at this epoch, so a
combined-fit component that "ties" bremss-alone in fit quality can still carry a
very different, poorly-constrained integrated flux). This is itself a useful
confirmation that bremss_powerlaw would have been a poor choice for flux
extraction at this epoch even though it wasn't the epoch's adopted model.
Full comparison: `fits/xray_luminosity_sanity_check_31211_31996.csv`.

**Final adopted numbers** (`data/Chandra/spectral_fitting/xray_flux_luminosity.py`,
run under `ciao-4.17`; output `fits/xray_flux_luminosity.csv`):

| Epoch | Phase (days) | Model | Unabsorbed flux (0.3–8 keV, erg/s/cm²) | Luminosity (erg/s) |
|---|---|---|---|---|
| 20306 | ~13 | powerlaw (N_H frozen) | 5.64×10⁻¹³ $^{+1.63\times10^{-13}}_{-1.29\times10^{-13}}$ | 7.75×10³⁹ $^{+2.25\times10^{39}}_{-1.78\times10^{39}}$ |
| 29071+29072 | ~1869 | bremss_powerlaw | 4.02×10⁻¹³ $^{+8.08\times10^{-12}}_{-2.69\times10^{-13}}$ | 5.53×10³⁹ $^{+1.11\times10^{41}}_{-3.70\times10^{39}}$ |
| 31211+31996 | ~2550 | bremss+gauss | 2.79×10⁻¹³ $^{+1.81\times10^{-14}}_{-2.10\times10^{-14}}$ | 3.84×10³⁹ $^{+2.49\times10^{38}}_{-2.89\times10^{38}}$ |

(All errors above are flux-statistical only — add the ±9.7% distance systematic
separately when quoting a final number.) **29071+29072's luminosity error is huge
and one-sided** (+1.11×10⁴¹, i.e. ~20× the median) — this is the same
"bremss_powerlaw doesn't meaningfully constrain its shape parameters at this
epoch" finding from Step 4 showing up again here, not a new problem; the median
is still a reasonable point estimate but the interval should not be over-read at
this epoch. 20306 and 31211+31996 have much better-behaved, roughly symmetric
errors.

**Implementation notes:**
- `ciao-4.17` has no `astropy` (unlike `18ivc_clean`), so `DATE-OBS` → phase-days
  used a small hand-rolled MJD conversion (`datetime` stdlib) reading the header
  directly off the already-loaded Sherpa PHA object (`get_data(1).header`) instead
  of `astropy.time.Time`.
- Refit each epoch's adopted model fresh (with the same multi-start machinery as
  the production fits — `BREMSS_POWERLAW_STARTS` for 29071+29072,
  `LINE_E_STARTS × SIGMA_STARTS` for 31211+31996) rather than trying to restore
  fitted state from CSV, since `sample_flux` needs a live Sherpa fit/covariance
  context, not just best-fit values.
- **Bug found and fixed while writing this script:** `fit_models_gauss.py`'s
  fitting calls were not guarded by `if __name__ == "__main__":`, so importing it
  just to reuse its `LINE_E_STARTS`/`SIGMA_STARTS`/`SIGMA_FLOOR_KEV` constants
  silently re-ran its entire 6-fit batch as an import side effect. Fixed by
  wrapping those calls the same way `fit_models.py` already was (see Step 4's
  "Done 2026-09-03 — Gaussian component added" for that earlier fix).

Not done: no attempt yet to fold the ±9.7% distance systematic into a single
combined error bar — kept separate per the table note above. Revisit if Step 6
light-curve fitting needs one combined number per epoch.

### Redone 2026-09-07/08 — PyXspec `lum`, Maeda et al. 2023b distance, 0.3-10 keV band

Motivated by two things: (1) an explicit request to use XSPEC's own `lum` command
(via the PyXspec build done this session — see the PyXspec build notes at the top
of this file's session history / the environment setup below) instead of Sherpa's
`sample_flux`; (2) adopting Maeda et al. 2023b's distance, D = 10 (+1.8/-1.5) Mpc,
in place of the Cepheid 10.72 Mpc used in the 2026-09-04 round.

**Tooling:** each epoch's adopted model refit directly in PyXspec (not just
Sherpa's best-fit values carried over frozen), seeded from the Sherpa best-fit
values, so XSPEC's own covariance matrix exists for `lum`/`flux`'s Monte-Carlo
error propagation. Fit statistic: `cstat` — this HEASOFT build has no separate
`wstat` name; `cstat` auto-applies the W-statistic whenever a background file is
loaded (same underlying calculation Sherpa calls `wstat`). Background loaded, not
subtracted, matching the Step 4 convention. Grouping: 20306 reuses its existing
`NUM_CTS`=15 grouped file; the two merged spectra (29071+29072, 31211+31996) were
regrouped via the `grppha` FTOOL's `GROUP MIN 15` (not bit-identical to Sherpa's
`group_counts(15)` — grppha groups the full channel range before any energy
notice, Sherpa notices then groups — but refit statistics land close to the
original Sherpa fits: 29071+29072 0.80/2 vs. 0.85/4, 31211+31996 24.03/25 vs.
24.85/27, both good matches; 20306 20.43/12 vs. 31.8/14, a bigger absolute shift
but the same qualitative story, powerlaw still the least-bad fit there).
Unabsorbed flux/lum: `TBabs.nH` temporarily zeroed (then restored), same
"exclude the absorber" definition Sherpa's `sample_flux(modelcomponent=...)` used.

**Distance — "effective H0" trick:** `lum`'s luminosity distance comes from
redshift + cosmology (H0, q0, Lambda0 via `Xset.cosmo`), not a directly specified
Mpc value. NGC 1068's actual z=0.003793 implies a naive Hubble-flow distance of
~16.2-16.9 Mpc (H0=70-67.4), but NGC 1068 has a large peculiar velocity, so the
true distance is much closer (Maeda et al. 2023b: 10 Mpc). Solved for an adjusted
H0 (H0_eff = c·z/D, exact to <0.5% at this tiny z, verified numerically against
XSPEC's own cosmological D_L before use) that makes `lum`'s internal D_L equal the
target distance: H0_eff=113.71 (central, D=10 Mpc), 96.37 (D=11.8 Mpc, +1.8
bound), 133.78 (D=8.5 Mpc, −1.5 bound). Flux-statistical error (MC at the central
H0) and the distance-systematic bracket (point estimate at the H0 bounds) computed
separately, then combined in quadrature per side when plotting (see below).

**Discrepancy check against an independent analysis (2026-09-07/08):** the
resulting luminosities were noticeably (~3-4x) lower than an independent X-ray
light curve the user's advisor (Poonam Chandra) made
(`figures/Poonam_xray_LC.png`, labeled 0.3-10 keV). Diagnosed via two isolated
tests (holding everything else fixed, evaluating at the same naive D~16.24 Mpc,
H0=70, to separate the two effects cleanly):

1. **Distance:** rescaling the (already-computed, band-independent) 0.3-8 keV
   fluxes to the naive D~16.24 Mpc closed most of the gap — residual ratio to
   Poonam's plot values dropped from ~3-4x to 1.16-1.52x (largest residual for
   20306, the hardest/flattest-spectrum epoch).
2. **Band:** refitting at 0.3-10 keV (vs. this project's earlier 0.3-8 keV) and
   evaluating at the same naive distance closed the remainder — residual ratio
   0.93-1.08x, i.e. full agreement with the advisor's numbers to within
   plot-read-off precision.

Together, these confirm the two analyses are consistent once distance and
bandpass are matched — the advisor's numbers were not independently wrong, this
project was just using a smaller, physically-motivated distance and a narrower
band than she did.

**Decision: adopt 0.3-10 keV going forward** (matches Chandra ACIS's full nominal
calibrated bandpass, `redback-csm`'s default, and reconciles with the advisor's
analysis), superseding the 0.3-8 keV band used earlier in this Step. **This does
not require re-running Step 4's spectral fits**: none of the 3 epochs have any
source counts above ~6.4-7.2 keV once grouped at 15 counts/bin (`QUALITY==0`
channels stop there), so widening the notice band to 10 keV adds zero new
constraining data — confirmed empirically (refit statistic/dof at 0.3-10 keV is
numerically identical to the 0.3-8 keV refit, epoch by epoch). **Caveat:** the
8-10 keV contribution to flux/luminosity is therefore pure model extrapolation of
the best-fit continuum past the last channel any data actually constrains, not a
directly measured excess — legitimate standard use of `flux`/`lumin` (they
integrate the best-fit model over whatever band is requested, independent of the
noticed data range), but worth remembering before treating these numbers as
purely data-driven, especially for 20306 (hardest spectrum, so the largest
extrapolated fraction of its total band flux). Response coverage checked and
confirmed adequate (RMF calibrated to 11 keV in all 3 epochs) before trusting the
wider band.

Scripts: `data/Chandra/spectral_fitting/xray_flux_luminosity_pyxspec.py` (0.3-8
keV version) → superseded by
`xray_flux_luminosity_pyxspec_0p3_10kev.py` (adopted, 0.3-10 keV). Output:
`fits/xray_flux_luminosity_pyxspec_0p3_10kev.csv`.

**Adopted numbers (0.3-10 keV, D=10 Mpc central, Maeda et al. 2023b):**

| Epoch | Phase (days) | Model | Unabsorbed flux (0.3-10 keV, erg/s/cm²) | L central (erg/s) | L stat range | L distance-systematic range |
|---|---|---|---|---|---|---|
| 20306 | ~12.7 | powerlaw (N_H frozen) | 8.988×10⁻¹³ | 1.072×10⁴⁰ | +1.16×10³⁹/−1.40×10³⁹ | 7.74×10³⁹–1.49×10⁴⁰ |
| 29071+29072 | ~1868.7 | bremss_powerlaw | 1.539×10⁻¹³ | 1.841×10³⁹ | +1.20×10⁴²/−8.19×10³⁸ (degenerate) | 1.33×10³⁹–2.56×10³⁹ |
| 31211+31996 | ~2550.3 | bremss+gauss | 3.068×10⁻¹³ | 3.681×10³⁹ | +1.59×10³⁸/−3.14×10³⁸ | 2.66×10³⁹–5.13×10³⁹ |

**29071+29072's statistical error remains degenerate/one-sided** (huge upper
bound, ~800x the median) — the same known `bremss_powerlaw` parameter degeneracy
flagged earlier in this Step (thermal/nonthermal components trade off against
each other at only 109 counts), unaffected by the tooling, distance, or band
change. Not investigated further, per the user's explicit request (2026-09-07)
to defer that.

**Plotted error bars combine the two error sources in quadrature, per side** —
previously (2026-09-04 round) the distance systematic was reported separately
and not plotted at all; now: `lum_err_total = sqrt(lum_err_stat² +
lum_err_dist²)`, independently for the +/- sides. For 29071+29072 specifically,
quadrature-combining also cleanly absorbs the sign oddity in its raw stat errlo
(a negative value, since its degenerate MC's "low" bound landed above the central
value) without needing a special case.

**Figure:** `figure_notebooks/xray_luminosity_light_curve.ipynb` →
`figures/xray_luminosity_light_curve.png/.pdf`, updated 2026-09-08 for the new
tool/distance/band/combined-error convention.

### Redone 2026-09-08 — 29071+29072 switched from bremss+powerlaw to powerlaw alone

Motivated by that epoch's persistently degenerate, huge one-sided luminosity
error (see immediately above): `bremss_powerlaw`'s 2 extra free parameters
(relative to a single continuum) trade off almost freely against each other at
only 109 counts, so while the fit itself is fine (W-stat/dof ≈ 0.8/4), the
flux/luminosity built from it inherits that degeneracy. All 4 models explored
at this epoch in Step 4's first round fit statistically indistinguishably well
(rstat ≲ 0.2 for all of apec/powerlaw/bremss/bremss_powerlaw) — so switching to
`powerlaw` alone isn't a worse fit, just a simpler, already well-constrained one
(Γ = 1.33 ± 0.19, ~20% norm error in the original Step 4 fit) that trades away
the (unconstrained) nonthermal component for a usable error bar. Implemented by
changing `setup_29071_29072()` in
`data/Chandra/spectral_fitting/xray_flux_luminosity_pyxspec_0p3_10kev.py` from
`TBabs*(bremss+powerlaw)` to `TBabs*powerlaw`, seeded from the original Step 4
powerlaw best-fit values (Γ=1.335, norm=1.404×10⁻⁵), N_H still frozen at
Galactic. Refit: W-stat/dof = 0.85/4 (vs. bremss_powerlaw's 0.80/2 — comparable,
2 more dof since powerlaw has 2 fewer free parameters).

**Effect on the error bar — as expected, large:**

| | bremss_powerlaw (previous) | powerlaw alone (adopted 2026-09-08) |
|---|---|---|
| L central (erg/s) | 1.841×10³⁹ | 1.676×10³⁹ |
| L stat error | +1.20×10⁴²/−8.19×10³⁸ (degenerate) | +2.08×10³⁸/−2.14×10³⁸ (~13%) |
| L distance-systematic | 1.33×10³⁹–2.56×10³⁹ | 1.21×10³⁹–2.33×10³⁹ |

The central value barely moved (~9%, within the models' mutual consistency),
but the statistical error collapsed from ~800x the median (effectively
uninformative) to a normal, well-behaved ~13% — confirming the error was a
`bremss_powerlaw`-specific parameter-degeneracy artifact, not a property of the
data itself.

**Adopted models as of 2026-09-08:** 20306 → powerlaw (N_H frozen); 29071+29072
→ **powerlaw (N_H frozen)** (was bremss_powerlaw); 31211+31996 → bremss+gauss
(unchanged). `xray_epoch_spec_models.xlsx` updated to match.

**Updated adopted numbers (0.3-10 keV, D=10 Mpc central, Maeda et al. 2023b):**

| Epoch | Phase (days) | Model | Unabsorbed flux (0.3-10 keV, erg/s/cm²) | L central (erg/s) | L stat range | L distance-systematic range |
|---|---|---|---|---|---|---|
| 20306 | ~12.7 | powerlaw (N_H frozen) | 8.988×10⁻¹³ | 1.072×10⁴⁰ | +1.10×10³⁹/−1.31×10³⁹ | 7.74×10³⁹–1.49×10⁴⁰ |
| 29071+29072 | ~1868.7 | powerlaw (N_H frozen) | 1.399×10⁻¹³ | 1.676×10³⁹ | +2.08×10³⁸/−2.14×10³⁸ | 1.21×10³⁹–2.33×10³⁹ |
| 31211+31996 | ~2550.3 | bremss+gauss | 3.068×10⁻¹³ | 3.681×10³⁹ | +1.43×10³⁸/−3.30×10³⁸ | 2.66×10³⁹–5.13×10³⁹ |

(20306/31211+31996 fluxes/errors shifted at the ~1-5% level from the previous
table purely from Monte-Carlo sampling noise on rerun — their models/setup were
untouched.)

**Figure updated again:** `figures/xray_luminosity_light_curve.png/.pdf` —
power-law-index annotations between epochs also updated (t⁻⁰·³⁷ then t²·⁵³,
vs. t⁻⁰·³⁵/t²·²³ before, since the middle point's central value shifted
slightly).

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
- Flux/luminosity conversion (Step 5) — **done 2026-09-04, superseded
  2026-09-07/08**: per-epoch adopted models from the user's
  `xray_epoch_spec_models.xlsx`. Original round used Sherpa's `sample_flux`
  (0.3-8 keV) and D=10.72 Mpc (Cepheid); superseded by a redo using PyXspec's
  `lum`/`flux` commands, D=10 (+1.8/-1.5) Mpc (Maeda et al. 2023b, via an
  "effective H0" trick), and a 0.3-10 keV band — the tool/distance/band change
  was prompted by, and resolved, a ~3-4x discrepancy against an independent
  X-ray light curve the user's advisor made (see "Redone 2026-09-07/08" in
  Step 5). Original sanity check (model-independence at 31211+31996, bremss_gauss
  vs. bremss-alone agreeing to ~1.2%) still stands as a validation of the
  continuum+line model choice, just not the current adopted flux/lum numbers.
  Current adopted table: `fits/xray_flux_luminosity_pyxspec_0p3_10kev.csv`. See
  Step 5 "Redone 2026-09-07/08" for full results and caveats (29071+29072's
  error bar is still huge and one-sided, reflecting that epoch's poor shape
  constraints — unaffected by the tool/distance/band change; the 8-10 keV
  contribution to flux/luminosity is model extrapolation, not measured, since no
  epoch has source counts above ~6.4-7.2 keV).
- Final-epoch (31211+31996) fit improvement — **done 2026-09-03**: added a
  Gaussian (free-centroid, and a fixed-at-2.9-keV comparison) to powerlaw, bremss,
  and bremss_powerlaw (not apec) for the 31211+31996 epoch only. Key result:
  bremss and bremss_powerlaw both show a significant, narrow, resolution-limited
  line at ~2.96 keV (ΔW-stat ~9-10 for the free-centroid fit); powerlaw shows no
  real improvement. Two robustness fixes were needed along the way — a 2D
  (LineE × Sigma) multi-start to avoid local minima, and an ACIS-resolution floor
  on Sigma (0.055 keV) to rule out the fit exploiting single-bin noise with an
  unresolvably narrow spike. See Step 4 "Done 2026-09-03 — Gaussian component
  added" for the full table, caveats, and figure reference. No formal significance
  test beyond ΔW-stat/Wilks'-theorem has been done yet.
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
