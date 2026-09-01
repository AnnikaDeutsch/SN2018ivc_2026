# X-ray (Chandra) Analysis SOP — SN 2018ivc

Standard operating procedure for the Chandra/X-ray analysis of SN 2018ivc. This is a
living document, built up step by step as the analysis proceeds — each stage gets
documented here once it's actually been done, not in advance. Read this before
picking the X-ray analysis back up in a new session.

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
| 29071 | NGC 1068 | 2024-01-04 17:29–21:14 | 10.3 ks | Andrea Marinucci / 705139 | **Post-explosion** (SN went off Nov 2018), ~5 yr after explosion. Serendipitous — from an AGN-monitoring program targeting the host galaxy, not a SN 2018ivc-targeted program. SN position confirmed on active chip (chip_id 7 / ACIS-S3, off-axis 0.33′, well clear of edges/gaps). Reprocessed + extracted 2026-09-01 — 64 src / 4 bkg counts, faint but clearly detected (SN faded substantially by this epoch). |
| 29072 | NGC 1068 | 2024-01-28 17:36–21:01 | 9.3 ks | Andrea Marinucci / 705140 | Same program as 29071, ~3.5 weeks later. SN position confirmed on active chip (chip_id 7, off-axis 0.25′). Reprocessed + extracted 2026-09-01 — 45 src / 8 bkg counts, faint but detected. |

So: **4 genuinely independent epochs** so far (20306; the merged 31211+31996 visit;
29071; 29072), spanning ~1 month to ~5 years post-explosion, not 5.

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

## Step 4 — Spectral fitting (not started)

Not yet done for any epoch. Will use Sherpa (within `ciao-4.17`) or XSPEC on the
grouped/background-subtracted spectra from Step 3 (or the combined spectrum from
Step 3.5, once that exists).

## Step 5 — Flux / luminosity conversion (not started)

Not yet done. Depends on Step 4 fit results plus the adopted distance/redshift to the
host (NGC 1068, z = 0.003793 — see [[project_sn2018ivc]] memory).

## Open items / TODO

- All 4 independent epochs now have extracted spectra (20306; merged 31211+31996;
  29071; 29072). Grouping has not been applied/reapplied to the merged 31211+31996
  spectrum (see Step 3.5) — decide at fitting time.
- Spectral fitting (Step 4) and flux/luminosity conversion (Step 5) — explicitly
  deferred by the user as of 2026-09-01; do not start without being asked.
