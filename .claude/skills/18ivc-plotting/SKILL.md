---
name: 18ivc-plotting
description: Use whenever creating, editing, or discussing a figure for the SN 2018ivc paper: SEDs, radio light curves, VLA/VLBA/ALMA frequency-coverage plots, or any matplotlib plot in this repo. Triggers on requests like "make a plot", "new figure", "plot the SED", "plot the light curve", "add a panel", "update the figure".
---

# 18ivc Plotting Skill

Source of truth: `CLAUDE_plotting.md` at the repo root. Read it in full before writing
any plotting code, every time this skill runs. Do not rely on a cached summary of it:
the epoch list, frequency norm bounds, and style rules change as the project
progresses, and this skill file is not the place those updates live.

## Workflow for a new figure

1. Read `CLAUDE_plotting.md` in full.
2. Pick a short, descriptive, snake_case figure name.
3. Create `figure_notebooks/<figure_name>.ipynb` (create the `figure_notebooks/`
   directory if it does not exist yet). Do not add the figure as a new cell in an
   existing catch-all notebook (`all_sed_plot.ipynb`, `vlba_proposal_figures.ipynb`,
   etc.); those predate this convention.
4. In the notebook: load data, call `functions.set_plot_style()`, build the figure,
   save both `.png` and `.pdf` to `figures/`.
5. Colors: viridis for epoch/time-evolution SEDs, turbo for frequency light curves,
   using the canonical mappings in `CLAUDE_plotting.md` (do not invent a new mapping
   or fall back to an arbitrary palette).
6. If new epochs or frequencies are introduced that fall outside the current
   canonical ranges, follow the extension procedure in `CLAUDE_plotting.md`
   (recompute the full mapping, regenerate affected figures) and update that file's
   tables accordingly.

## Workflow for revising an existing figure

1. Read `CLAUDE_plotting.md` in full.
2. Edit the figure's existing notebook under `figure_notebooks/` in place. Do not
   create a second notebook for the same figure.
3. Re-verify the color mapping and style helper call are still correct after the
   edit.

## Also apply

`CLAUDE_plotting.md` also carries this project's general working conventions
(communication style, file-saving discipline, debugging workflow, safety
constraints, git conventions). Those apply to this work too, not just the plotting
rules.
