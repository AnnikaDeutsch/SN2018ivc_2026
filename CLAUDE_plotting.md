# Plotting Style Guide — SN 2018ivc Paper Figures

Conventions for producing format-consistent plots for the 18ivc paper. Read this
before writing or editing any plotting code in this repo.

# Project Context & Guidelines

## Communication
- Be direct, concise, and technical. Skip preambles, pleasantries, and unnecessary explanations.
- Do not use em dashes; use colons or short, separate sentences.
- Lead with code or direct solutions, then follow with high-level conceptual summaries.

## File Saving & Modifications
- Always verify target file paths before writing or modifying code.
- Write complete, production-ready code blocks; never use placeholders like `// TODO: implement rest`.
- Run local formatters or linters immediately after updating any file.

## Debugging Workflow
- Reproduce or isolate errors with targeted logs or test commands before changing implementation logic.
- Analyze stack traces from the source upward; do not guess fixes.
- If a test fails twice, stop and output a diagnostic summary instead of blindly iterating.

## Safety & Hard Constraints
- **CRITICAL:** Never expose, hardcode, or echo secrets, API keys, or connection strings into code or logs.
- Never run destructive shell commands (`rm -rf`, database drops) without explicit user confirmation.
- Restrict file operations strictly to the workspace directory.

## Git Conventions
- Run relevant unit tests or typechecks before staging changes.
- Use Conventional Commits format: `type(scope): concise description` (e.g., `fix(auth): handle expired token redirect`).


## General formatting reference

`figures/19yvr_figformat_example.png` is the reference for general figure formatting
(font, text size, axes style, tick style, legend style). Before building a new
figure, look at this image directly rather than relying solely on the summary below
— the summary may drift out of date.

Observed conventions from the reference figure:
- **Font: Times New Roman** (serif), used throughout — labels, tick labels, legend,
  in-panel text. Confirmed by direct glyph comparison against candidate fonts (the
  "J" descender curl and stroke contrast rule out DejaVu Serif/STIX).
- Large, clearly legible text — axis labels and tick labels are comparable in size to
  body text in a two-column journal figure, not small/cramped.
- Log–log axes for SED/light-curve-style panels.
- Panels share a single outer y-axis label when tiled side by side, rather than
  repeating the label on every panel.
- In-panel annotations (e.g. frequency labels) placed in a corner as plain text
  rather than in the legend.
- Legend (when present) uses plain lines with no marker glyphs, minimal frame.
- Tick marks point inward and are mirrored on the top/right axes.
- Small vertical dotted tick marks along the bottom of a panel are used to flag
  specific epochs of interest, independent of the data points themselves.

**No gridlines, on any axis, including log-scale plots.** Do not call
`ax.grid(...)` with `True` (decision made 2026-08-27, after finding log-scale
gridlines visually cluttered the light-curve/SED figures). If a plotting
function needs a gridline call at all, use `ax.grid(False)` — matplotlib
defaults to no grid, so in practice this means simply not calling `ax.grid()`.

**Colors always come from the viridis/turbo schemes below — do not fall back to an
arbitrary palette (e.g. red data points with black/blue model curves) even though
that appears in the reference image.** The reference image is for typography and
axes formatting only, not for color choices.

### Shared style helper

Use `functions.set_plot_style()` instead of redefining `rcParams` inline in each
notebook:

```python
import functions
functions.set_plot_style()
```

This applies the shared baseline (defined in `functions.py`):

```python
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})
```

If a figure needs a one-off tweak, call `set_plot_style()` first and then override
just the specific `rcParams` key(s) locally — don't duplicate the whole block.

## File organization: one notebook per figure

**Every new figure gets its own notebook**, saved in a dedicated subdirectory:

```
figure_notebooks/<figure_name>.ipynb
```

- `<figure_name>` should be a short, descriptive, snake_case name for what the
  figure shows (e.g. `figure_notebooks/late_time_sed.ipynb`,
  `figure_notebooks/vla_lightcurve_by_band.ipynb`) — not a generic name like
  `plot.ipynb` or `figure1.ipynb`, since names should stay meaningful as the paper's
  figure numbering shifts.
- Do not add new figures as additional cells in an existing catch-all notebook
  (e.g. `all_sed_plot.ipynb`, `vlba_proposal_figures.ipynb`) going forward — those
  predate this convention. Start a fresh notebook under `figure_notebooks/` instead.
- Each figure notebook should be self-contained: load data, call
  `functions.set_plot_style()`, build the figure, and save it out (following the
  existing pattern of saving both `.png` and `.pdf` into `figures/`).
- If a figure is revised, edit its existing notebook in place rather than creating a
  second notebook for the same figure.

## Color scheme rules

- **SEDs showing time evolution (epochs):** use the **viridis** colormap. Color
  encodes epoch (phase post-explosion).
- **Light curves showing multiple frequencies/bands on the same plot:** use the
  **turbo** colormap. Color encodes frequency.
- **Once a color is established for a given epoch or frequency/band, it must stay
  that color in every subsequent plot.** Never re-derive a color mapping locally
  within a single figure's code in a way that could shift colors as epochs/bands are
  added or removed elsewhere. Use the canonical mappings below (extend, don't
  reorder or renumber, when adding new epochs/bands).

### Epoch → viridis color (SEDs)

Colors are assigned from `viridis` via `cmap(x)` for `x = linspace(1, 0, n)` over the
**full canonical epoch list for the whole project**, sorted chronologically —
the youngest epoch gets `x=1` (yellow end), the oldest gets `x=0` (dark purple end):

```python
cmap = mpl.colormaps["viridis"]
colors = [cmap(x) for x in np.linspace(1, 0, len(epoch_info))]
```

**The full range spans as early as ~4 days post-explosion (the earliest data in
`data/radio_18ivc_data.csv`) through ~2650+ days — not just the late-time epochs.**
The youngest epoch in the full list is what gets `x=1.00` (yellow), regardless of
which epochs happen to be included in any one figure. **As of 2026-09-01 this list
is shared across wavelengths, not just radio** — the 3 X-ray epochs (see
`CLAUDE_xray_SOP.md`) are interleaved into the same chronological list on the same
color axis, not a separate mapping.

Canonical epoch list, `x = linspace(1, 0, 11)` (radio epochs originally tracked in
`figure_notebooks/sed_epoch_grid.ipynb`; X-ray epochs added 2026-09-01):

| epoch (days) | phase range | x (viridis position) | source |
|---|---|---|---|
| 4.0 | 0–10 | 1.000 | raw data (no model fit yet) |
| 12.7 | — (single X-ray spectrum, ObsID 20306) | 0.900 | `data/Chandra/20306/repro/` |
| 20.0 | 10–30 | 0.800 | raw data (no model fit yet) |
| 200.0 | 150–280 | 0.700 | raw data (no model fit yet) |
| 1300.0 | 1200–1400 | 0.600 | `model_params/best_fit_params_1300.0_days.csv` |
| 1700.0 | 1600–1800 | 0.500 | `model_params/best_fit_params_1700.0_days.csv` |
| 1868.7 | — (single X-ray spectrum, merged ObsIDs 29071+29072) | 0.400 | `data/Chandra/29071_29072_merged/` |
| 2100.0 | 2000–2200 | 0.300 | `model_params/best_fit_params_2100.0_days.csv` |
| 2500.0 | 2350–2550 | 0.200 | `model_params/best_fit_params_2500.0_days.csv` |
| 2550.3 | — (single X-ray spectrum, merged ObsIDs 31211+31996) | 0.100 | `data/Chandra/31211_31996_merged/` |
| 2650.0 | 2550–2750 | 0.000 | `model_params/best_fit_params_2650.0_days.csv` |

X-ray epochs have no "phase range" (each is one specific merged/unmerged spectrum,
not a window that other survey data points get binned into the way VLA/ALMA points
are for the radio epochs) — their day value is the exact phase computed from the
spectrum's `DATE-OBS` header vs. explosion epoch MJD 58445.0 (Maeda et al. 2023a),
not a rounded bin center. Note 2550.3 (X-ray) sits between the existing 2500.0 and
2650.0 radio bins but is kept as a separate list entry, not merged into either —
it's a different instrument/observation, not the same epoch.

**No early-epoch (~4–1000 day) model fits exist yet in `model_params/`** — the three
early radio epochs above are data-points-only (see `sed_epoch_grid.ipynb`).

**Known inconsistency #1 (intentional, by user decision on 2026-08-19):** adding the
three early radio epochs shifted the `x` position of every late-time epoch versus the
old 5-epoch-only mapping (`x = linspace(1, 0, 5)` → 1300.0/1700.0/2100.0/2500.0/2650.0
at 1.00/0.75/0.50/0.25/0.00). The user chose *not* to regenerate the older figures
that still use that 5-epoch mapping — `figures/vlba_proposal_sed.png/pdf`,
`figures/sed_all_epochs_one_plot.png`, and `figures/sed_subplots.png` (built in
`vlba_proposal_figures.ipynb` and `all_sed_plot.ipynb`) — so those figures'
late-epoch colors do NOT match the same epochs in `sed_epoch_grid.ipynb` or any
newer figure. Do not silently "fix" this by regenerating those older notebooks; ask
first, since one of them feeds a proposal document.

**Resolved 2026-09-01 — `sed_epoch_grid.ipynb` regenerated for the 11-epoch mapping.**
Adding the 3 X-ray epochs shifted the `x` position of every radio epoch versus the
old 8-epoch-only mapping (e.g. 1300.0 moves from `x=0.571` to `x=0.600`).
`sed_epoch_grid.ipynb` now hardcodes each radio epoch's `x` from the current
11-epoch canonical table above (`CANONICAL_X` dict in the notebook) instead of
computing `linspace(1, 0, len(epoch_info))` locally over just its own 8 epochs —
that local-linspace approach is what silently produced the old, now-wrong mapping,
so don't revert to it. `figures/sed_epoch_grid.png/.pdf` reflect the update.

`vlba_proposal_sed.png/pdf`, `sed_all_epochs_one_plot.png`, and `sed_subplots.png`
(the legacy 5-epoch-mapping figures from inconsistency #1) were **not** touched by
this update either — they were already inconsistent with `sed_epoch_grid.ipynb`
before this change and remain so; this update didn't add a new inconsistency for
them, just carried the existing one forward.

If new epochs are introduced again in the future:
1. Insert them in chronological order at the front of the list (they're younger).
2. Recompute `linspace(1, 0, n)` over the *complete* updated list.
3. Ask the user before regenerating other existing SED figures that share this
   mapping — don't assume regeneration is wanted, since some of those figures may
   already be in an external-facing document.

Do not manually override an individual epoch's color to something off-map (e.g. a
custom hex) unless the figure is deliberately highlighting that epoch —
`all_sed_plot.ipynb` does this once (`colors[0] = "#D4710A"`); treat that as an
intentional one-off, not the default pattern.

**Model/fit-curve color, when a figure overlays a best-fit curve on epoch-colored
data points** (e.g. the X-ray `xray_spectra_*_fit.ipynb` notebooks): don't introduce
an unrelated fixed color (black, blue, etc.) for the curve — that's the "arbitrary
palette" the reference-figure caveat above warns against. Instead derive the curve
color from the same epoch's viridis color via a `darken_color` helper (already used
this way in `sed_epoch_grid.ipynb`):
```python
def darken_color(color, factor=0.7):
    rgb = mcolors.to_rgb(color)
    return tuple(factor * c for c in rgb)
```
so the data points and their model curve are visibly linked (same hue family) while
still distinguishable from each other.

### Frequency/band → turbo color (light curves)

Color is a continuous function of frequency (log-scaled) via `LogNorm`, not a fixed
per-band swatch:

```python
norm = LogNorm(vmin=1, vmax=250)
cmap = mpl.colormaps["turbo"]
color = cmap(norm(frequency))
```

**Canonical norm: `LogNorm(vmin=1, vmax=250)`**, covering the full frequency range
of the dataset — from ~1 GHz (L band) through ALMA mm-wave coverage up to 250 GHz
(`data/radio_18ivc_data.csv` spans 1.52–250 GHz). Use these bounds for every
light-curve figure so a given frequency always maps to the same color, regardless of
which subset of bands appears in that particular figure.

Representative frequencies (GHz) for common bands, for labeling/reference:

| band | representative freq (GHz) |
|---|---|
| L | 1.5 |
| S | 3 |
| C | 6 |
| X | 10 |
| Ku | 15 |
| K | 22 |
| Ka | 33 |
| Q | 44 |
| ALMA (mm) | 90–250 |

If a future observation falls outside 1–250 GHz, widen the norm bounds and note the
change here — then regenerate other light-curve figures that share this mapping,
since widening the norm shifts every color slightly. Prefer widening the norm over
clamping an out-of-range point to an edge color.

## Marker shape by data source (SED figures)

Introduced in `sed_epoch_grid.ipynb`. When a figure plots literature data alongside
new data, marker shape encodes provenance (independent of the color, which still
encodes epoch or frequency per the rules above):

| source | marker |
|---|---|
| ALMA (Maeda+2023a,b) | circle (`o`) |
| VLA (Bill Cotton) | square (`s`) |
| This work (the 5 late canonical epochs: 1300/1700/2100/2500/2650 days) | star (`*`) |

The "this work" epochs get a star regardless of which telescope/`telescope` column
value the row has — the attribution is by epoch (the late-time monitoring campaign),
not by facility. For the three early epochs (4/20/200 days), marker follows the row's
`telescope` column (`ALMA`→circle, `VLA`/`VLASS`→square) since those are literature
points from more than one source. Stars need a larger `ms` than circles/squares
(`*` renders visually smaller at the same `ms`) — `sed_epoch_grid.ipynb` uses 15 vs. 9.

## Open items / TODO

- No early-epoch (~4–1000 day) SED *model fits* exist yet in `model_params/` (raw
  data points for 4/20/200 days are plotted in `sed_epoch_grid.ipynb`). Once fits
  exist, update the epoch table above with their source files.
- `figures/vlba_proposal_sed.png/pdf`, `sed_all_epochs_one_plot.png`, and
  `sed_subplots.png` still use the old 5-epoch viridis mapping and were
  deliberately left unregenerated when the epoch list was extended to 8 (see the
  "Known inconsistency" note above) — their late-epoch colors won't match newer
  figures until/unless the user asks for them to be regenerated.
