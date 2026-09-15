"""
Step 1 sanity checks (CLAUDE_modelling_SOP.md) for SN 2018ivc `redback-csm` modeling.

Run with: conda activate 18ivc_csm && python csm_modeling/step1_sanity_checks.py

1. Confirm the 18ivc_csm env resolves and wind_bpl_{radio,xray} are registered.
2. Build the radio (flux_density) and X-ray (luminosity) redback transient objects.
3. Quick non-Bayesian look: wind_bpl_radio/wind_bpl_xray at the fixed
   mexp/eexp/vwind (Maeda et al. 2023b) and a few representative mdot guesses,
   plotted against the real data, before spending compute on nested sampling.

The X-ray Lum50_err used here (quadrature sum of the statistical error and the
distance-systematic bracket) is PROVISIONAL, for this plot only — the actual
error handling for the Step 3 fit is still an open decision (see SOP Step 1/3).

Radio panel colors follow CLAUDE_plotting.md's "Frequency/band -> turbo color
(light curves)" rule: turbo colormap, LogNorm(vmin=1, vmax=250) over frequency.
The X-ray panel isn't a multi-frequency light curve (single 0.3-10 keV band,
comparing mdot guesses) so that rule doesn't apply there; it uses a fixed
purple, deliberately off the turbo scale so it doesn't read as part of the
radio frequency axis.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import redback
import redback.model_library as ml
from redback_csm.models import wind_bpl_radio, wind_bpl_xray

REDSHIFT = 0.003793  # NGC 1068

# Fixed ejecta parameters (Maeda et al. 2023b; user decision 2026-09-14, see SOP)
MEXP = 3.0    # M_sun
EEXP = 1.2    # foe
VWIND = 20.0  # km/s

# --- 1. Env / model-registration check -------------------------------------
assert 'wind_bpl_radio' in ml.all_models_dict, "wind_bpl_radio not registered"
assert 'wind_bpl_xray' in ml.all_models_dict, "wind_bpl_xray not registered"
print("[1] wind_bpl_radio / wind_bpl_xray both present in redback.model_library.all_models_dict")

# --- 2. Build transient objects ---------------------------------------------
radio = pd.read_csv('data/radio_18ivc_data.csv')
radio_transient = redback.transient.Transient(
    name='SN2018ivc',
    data_mode='flux_density',
    time=radio['phase'].values,             # days
    flux_density=radio['flux'].values,       # mJy
    flux_density_err=radio['flux_err'].values,
    frequency=radio['freq'].values * 1e9,    # GHz -> Hz
)
print(f"[2] Built radio Transient: {len(radio)} points, "
      f"{radio['phase'].min():.1f}-{radio['phase'].max():.1f} d, "
      f"{radio['freq'].min():.2f}-{radio['freq'].max():.2f} GHz")

xray = pd.read_csv('data/Chandra/spectral_fitting/fits/xray_flux_luminosity_pyxspec_0p3_10kev.csv')
lum50_central = xray['luminosity_central'].values / 1e50
stat_err = xray[['lum_errlo', 'lum_errhi']].mean(axis=1).values / 1e50
dist_err = ((xray['luminosity_dist_upper'] - xray['luminosity_central']).abs()
            + (xray['luminosity_central'] - xray['luminosity_dist_lower']).abs()).values / 2 / 1e50
lum50_err_provisional = np.sqrt(stat_err**2 + dist_err**2)  # PROVISIONAL — see Step 3

xray_transient = redback.transient.Supernova(
    name='SN2018ivc',
    data_mode='luminosity',
    time_rest_frame=xray['phase_days'].values,
    Lum50=lum50_central,
    Lum50_err=lum50_err_provisional,
    redshift=REDSHIFT,
)
print(f"[2] Built X-ray Supernova: {len(xray)} points, "
      f"{xray['phase_days'].min():.1f}-{xray['phase_days'].max():.1f} d "
      "(Lum50_err = provisional quadrature sum of stat + distance-systematic, not final)")

# --- 3. Quick non-Bayesian look ---------------------------------------------
ejecta_kw = dict(mexp=MEXP, eexp=EEXP, delta=1.0, nn=10.0, eff=0.5)
sync_kw = dict(logepsb=-2.0, logepse=-1.0, p=3.0)
xray_kw = dict(logepsx=-1.0, e_min_kev=0.3, e_max_kev=10.0, mode="simple")

mdot_guesses = [1e-5, 1e-4, 1e-3]  # M_sun/yr, representative guesses
time_grid = np.geomspace(1, 3000, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Radio panel: representative frequencies spanning the data.
# Color follows CLAUDE_plotting.md: turbo colormap, LogNorm(vmin=1, vmax=250) on frequency.
freqs_ghz = [6.0, 15.0, 33.0, 100.0]
freq_norm = LogNorm(vmin=1, vmax=250)
freq_cmap = mpl.colormaps["turbo"]
colors = [freq_cmap(freq_norm(f)) for f in freqs_ghz]
linestyles = ['--', '-', ':']
ax = axes[0]
for freq_ghz, color in zip(freqs_ghz, colors):
    mask = np.isclose(radio['freq'].values, freq_ghz, atol=1.5)
    if mask.any():
        ax.errorbar(radio['phase'].values[mask], radio['flux'].values[mask],
                     yerr=radio['flux_err'].values[mask], fmt='o', color=color,
                     label=f'{freq_ghz:.0f} GHz data', alpha=0.8, ms=5)
    for ls, mdot in zip(linestyles, mdot_guesses):
        flux = wind_bpl_radio(time=time_grid, redshift=REDSHIFT, mdot=mdot, vwind=VWIND,
                               frequency=freq_ghz * 1e9, **ejecta_kw, **sync_kw)
        ax.plot(time_grid, flux, ls=ls, color=color, lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Phase (days)'); ax.set_ylabel('Flux density (mJy)')
ax.set_title('Radio: wind_bpl_radio vs data (fixed mexp/eexp/vwind)')

# Legend: frequency (color, from the data) and mdot (linestyle, gray proxy handles
# so the linestyle key isn't tied to any one frequency's color).
freq_handles, freq_labels = ax.get_legend_handles_labels()
mdot_handles = [Line2D([], [], color='gray', ls=ls, lw=1.5) for ls in linestyles]
mdot_labels = [f'mdot={mdot:.0e}' for mdot in mdot_guesses]
ax.legend(freq_handles + mdot_handles, freq_labels + mdot_labels, fontsize=7, ncol=2)

# X-ray panel
ax = axes[1]
ax.errorbar(xray['phase_days'].values, lum50_central * 1e50,
             yerr=lum50_err_provisional * 1e50, fmt='o', color='k', label='data (provisional err)')
for ls, mdot in zip(['--', '-', ':'], mdot_guesses):
    lx = wind_bpl_xray(time=time_grid, redshift=REDSHIFT, mdot=mdot, vwind=VWIND,
                        output_format="luminosity", **ejecta_kw, **xray_kw)
    # purple: deliberately off the radio panel's turbo (blue-cyan-green-yellow-orange-red)
    # scale, so the X-ray curves don't read as if they belong on the frequency color axis.
    ax.plot(time_grid, lx, ls=ls, color='darkviolet', lw=1.5, label=f'mdot={mdot:.0e}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Phase (days)'); ax.set_ylabel('0.3-10 keV luminosity (erg/s)')
ax.set_title('X-ray: wind_bpl_xray vs data (fixed mexp/eexp/vwind)')
ax.legend(fontsize=8)

fig.tight_layout()
out = 'csm_modeling/diagnostics/step1_sanity_check.png'
fig.savefig(out, dpi=150)
print(f"[3] Saved sanity-check plot: {out}")
