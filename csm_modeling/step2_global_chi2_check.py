"""
Diagnostic (not a Step in the SOP itself): global chi2 minimization for wind_bpl_radio
against the full 63-point radio dataset, run 2026-09-15 after two nlive=50 pilot fits
both pinned multiple free parameters (p, nn, then mdot/p/nn again after widening) at
their prior walls with no sign of settling. Question: is there ANY point in a wide,
physically generous parameter volume where wind_bpl_radio achieves a statistically
reasonable fit (chi2/dof ~ a few), or is the wall-chasing a symptom of the model
family genuinely being unable to fit this multi-frequency light curve shape?

Uses scipy.optimize.differential_evolution (global, bounded) rather than nested
sampling -- much cheaper per outcome for a yes/no "can this model fit this data at
all" question. mexp/eexp/vwind/redshift held fixed per the SOP's Step 2 decision.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from redback_csm.models import wind_bpl_radio

REDSHIFT = 0.003793
MEXP = 3.0
EEXP = 1.2
VWIND = 20.0

radio = pd.read_csv('../data/radio_18ivc_data.csv')
t = radio['phase'].values
freq = radio['freq'].values * 1e9
y = radio['flux'].values
yerr = radio['flux_err'].values


def model_multifreq(time, frequency, **kwargs):
    time = np.asarray(time, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    out = np.empty_like(time)
    for f in np.unique(frequency):
        mask = frequency == f
        out[mask] = wind_bpl_radio(time=time[mask], frequency=f, **kwargs)
    return out


# Parameter vector: [log10(mdot), delta, nn, eff, logepsb, logepse, p]
# Wide, physically generous bounds -- wider than either pilot's priors on every axis.
BOUNDS = [
    (-8.0, 1.0),    # log10(mdot), Msun/yr
    (0.0, 3.0),     # delta
    (6.0, 300.0),   # nn
    (0.001, 1.0),   # eff
    (-6.0, 0.0),    # logepsb
    (-6.0, 0.0),    # logepse
    (1.1, 5.0),     # p
]


def chi2(x):
    log_mdot, delta, nn, eff, logepsb, logepse, p = x
    kw = dict(redshift=REDSHIFT, mdot=10 ** log_mdot, vwind=VWIND, delta=delta, nn=nn,
              mexp=MEXP, eexp=EEXP, eff=eff, logepsb=logepsb, logepse=logepse, p=p)
    try:
        pred = model_multifreq(t, frequency=freq, **kw)
    except Exception:
        return 1e12
    if not np.all(np.isfinite(pred)):
        return 1e12
    return np.sum(((y - pred) / yerr) ** 2)


if __name__ == '__main__':
    result = differential_evolution(
        chi2, BOUNDS, seed=42, maxiter=150, popsize=20, tol=1e-8,
        mutation=(0.5, 1.5), recombination=0.7, polish=True, workers=-1, disp=True,
    )
    dof = len(y) - 7
    print()
    print(f"Best chi2 = {result.fun:.1f}, dof = {dof}, chi2/dof = {result.fun / dof:.2f}")
    names = ['log10(mdot)', 'delta', 'nn', 'eff', 'logepsb', 'logepse', 'p']
    for name, val in zip(names, result.x):
        print(f"  {name:12s} = {val:.4f}")
    print(f"  mdot = {10**result.x[0]:.3e} Msun/yr")
