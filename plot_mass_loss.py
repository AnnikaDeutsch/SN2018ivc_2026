import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/Users/adeutsch/SN2018ivc_2026/mass_loss_rates.csv')

# ── Physical conversion ────────────────────────────────────────────────────────
# v_ejecta = 10,000 km/s,  v_wind = 20 km/s  =>  ratio = 500
v_ejecta      = 10000.0
v_wind        = 20.0
v_ratio       = v_ejecta / v_wind
days_per_year = 365.25

def days_to_years(days):
    return np.asarray(days) * v_ratio / days_per_year

def years_to_days(years):
    return np.asarray(years) * days_per_year / v_ratio

df['time_to_SN_yr'] = days_to_years(df['phase (days)'])

# ── Maeda+ 2023b sits at the same location as Component 1 @ phase=1300 days ──
maeda_row  = df[(df['component'] == 'Component 1') & (df['phase (days)'] == 1300.0)].iloc[0]
maeda_yr   = maeda_row['time_to_SN_yr']   # ~1780 yr
maeda_mdot = maeda_row['Mdot (Msun/yr)']  # ~6e-4
# Keep the point in df — it stays as a normal Component 1 scatter point too

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         13,
    'axes.linewidth':    1.8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.major.size':  7,
    'ytick.major.size':  7,
    'xtick.minor.size':  4,
    'ytick.minor.size':  4,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
})

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

styles = {
    'Component 1': dict(color='#E45756', marker='D', ms=9,  lw=2.2, label='Component 1'),
    'Component 2': dict(color='#2A9D8F', marker='o', ms=9,  lw=2.2, label='Component 2'),
    'Component 3': dict(color='#7B2CBF', marker='s', ms=9,  lw=2.2, label='Component 3'),
}

def smooth_log_line(x, y, color, lw, n=300):
    """Smooth line in log-log space using pchip interpolation."""
    lx = np.log10(x)
    ly = np.log10(y)
    lx_fine = np.linspace(lx.min(), lx.max(), n)
    ly_fine = pchip_interpolate(lx, ly, lx_fine)
    ax.plot(10**lx_fine, 10**ly_fine, color=color, lw=lw, ls='-', zorder=4, alpha=0.85)

# ── Plot each component ───────────────────────────────────────────────────────
for comp, grp in df.groupby('component'):
    grp = grp.sort_values('time_to_SN_yr')
    s   = styles[comp]

    # All components: smooth through all points (Maeda is already in Component 1 data)
    if False:
        pass
    else:
        if len(grp) >= 2:
            x_s = grp["time_to_SN_yr"].values
            y_s = grp["Mdot (Msun/yr)"].values
            idx_s = np.argsort(x_s)
            smooth_log_line(x_s[idx_s], y_s[idx_s],
                            s['color'], s['lw'])

    # Scatter points (normal, non-Maeda)
    ax.scatter(grp['time_to_SN_yr'], grp['Mdot (Msun/yr)'],
               color=s['color'], marker=s['marker'], s=s['ms']**2,
               edgecolors='black', linewidths=0.8, label=s['label'], zorder=5)

# ── Maeda+ 2023b point ────────────────────────────────────────────────────────
ax.errorbar(maeda_yr, maeda_mdot,
            xerr=[[maeda_yr*0.15], [maeda_yr*0.20]],
            yerr=[[maeda_mdot*0.5], [maeda_mdot*1.0]],
            fmt='D', color='black', ms=10,
            ecolor='gray', elinewidth=1.5, capsize=0,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Maeda+ 2023b (Comp. 1)', zorder=6)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(5000, 1000)
ax.set_ylim(1e-4, 2e-2)

# Bottom x: scientific notation (10^N)
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=20))
ax.xaxis.set_minor_formatter(NullFormatter())

# Y-axis
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(which='both', right=True)

ax.set_xlabel('Time to SN [yr]', fontsize=14)
ax.set_ylabel(r'Mass-loss rate [$M_\odot$ yr$^{-1}$]', fontsize=14)

# ── Top axis: Phase [days] ────────────────────────────────────────────────────
ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xlim(*[years_to_days(x) for x in ax.get_xlim()])

phase_day_ticks = np.array([3000, 2500, 2000, 1500])
ax2.set_xticks(phase_day_ticks)
ax2.set_xticklabels([str(int(t)) for t in phase_day_ticks], fontsize=11)
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=20))
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xlabel('Phase [days]', fontsize=14, labelpad=8)
ax2.tick_params(which='both', direction='in')

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(loc='upper left', fontsize=10, frameon=True,
                   edgecolor='black', fancybox=False,
                   handlelength=2.0, handletextpad=0.6)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig('./figures/mass_loss_rate_plot.pdf', bbox_inches='tight', dpi=200)
#plt.savefig('/mnt/user-data/outputs/mass_loss_rate_plot.png', bbox_inches='tight', dpi=200)
print("Saved!")
