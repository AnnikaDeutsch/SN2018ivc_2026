"""
Step 2 (CLAUDE_modelling_SOP.md): baseline radio-only wind_bpl_radio fit for SN 2018ivc.

Run with: conda activate 18ivc_csm && python csm_modeling/step2_radio_only_fit.py [--nlive N] [--label LABEL]
(run from the repo root -- paths below are relative to it)

Fixed: mexp=3, eexp=1.2, vwind=20 (Maeda et al. 2023b), redshift=0.003793 (NGC 1068).
Free: mdot, delta, nn, eff, logepsb, logepse, p.
Data: data/radio_18ivc_data.csv, all 63 points, multi-frequency.

IMPORTANT implementation note (found 2026-09-15, not anticipated in the SOP as
originally written): wind_bpl_radio's `frequency` kwarg must broadcast against its
own internal Fortran time grid (length ~459), not the caller's `time` array -- it
cannot take a per-point (time_i, frequency_i) array the way redback's built-in
multiband/SED models can (confirmed: passing the full 63-point frequency array
errors with a shape-broadcast mismatch, (63,) vs (459,)). The redback_fit_example.py
pattern the SOP cited ("redback fits the frequency-dependence directly via the
frequency array passed in model_kwargs") does NOT work for this model -- that script
was never actually run (see its own docstring), and Step 1's sanity check didn't
catch this because it looped over one scalar frequency at a time, never passing a
mixed array. Fixed here with a thin wrapper, `wind_bpl_radio_multifreq`, that groups
data by unique frequency and calls wind_bpl_radio once per group (scalar frequency),
reassembling results in the original order.

Second implementation subtlety: the wrapper's signature must be `(time, **kwargs)`
with `frequency` popped out of **kwargs inside the function body, NOT a named
parameter (`def f(time, frequency, **kwargs)`). redback's GaussianLikelihood calls
`self.function(self.x, **self.parameters, **self.kwargs)`, and bilby infers which
parameter names go in `self.parameters` from the function's *signature*
(`infer_parameters_from_function`). If `frequency` were a named parameter, it would
end up in both `self.parameters` (as None, from the signature-inference step) and
`self.kwargs` (the actual per-point array, via model_kwargs) -- a duplicate-keyword
TypeError when the two dicts get unpacked together. Keeping the signature to just
`(time, **kwargs)` avoids this (confirmed via
`bilby.core.utils.introspection.infer_parameters_from_function`).
"""
import argparse
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd
import bilby
import redback

from redback_csm.models import wind_bpl_radio

REPO_ROOT = Path(__file__).resolve().parent.parent
REDSHIFT = 0.003793  # NGC 1068
MEXP = 3.0    # M_sun, Maeda et al. 2023b
EEXP = 1.2    # foe, Maeda et al. 2023b
VWIND = 20.0  # km/s, Maeda et al. 2023b


def wind_bpl_radio_multifreq(time, **kwargs):
    """Per-point (time_i, frequency_i) wrapper around wind_bpl_radio -- see module
    docstring for why this is needed and why `frequency` is popped from **kwargs
    rather than declared as a named parameter."""
    kwargs = dict(kwargs)
    frequency = np.asarray(kwargs.pop('frequency'), dtype=float)
    time = np.asarray(time, dtype=float)
    out = np.empty_like(time)
    for f in np.unique(frequency):
        mask = frequency == f
        out[mask] = wind_bpl_radio(time=time[mask], frequency=f, **kwargs)
    return out


def build_transient(radio):
    return redback.transient.Transient(
        name='SN2018ivc',
        data_mode='flux_density',
        time=radio['phase'].values,
        flux_density=radio['flux'].values,
        flux_density_err=radio['flux_err'].values,
        frequency=radio['freq'].values * 1e9,
        # Required for fit_model: Transient.active_bands defaults to None (not
        # 'all'), and get_filtered_data() -> filtered_indices does
        # `b in self.active_bands for b in self.bands`, which TypeErrors on None.
        # Step 1's sanity check never hit this since it never called fit_model.
        active_bands='all',
    )


def build_priors():
    priors = redback.priors.get_priors(model='wind_bpl_radio')
    # Fixed per Maeda et al. 2023b / NGC 1068 (user decision 2026-09-14, see SOP).
    priors['redshift'] = REDSHIFT
    priors['mexp'] = MEXP
    priors['eexp'] = EEXP
    priors['vwind'] = VWIND
    # vej_max_ratio isn't an actual wind_bpl_radio parameter -- it's absorbed into
    # **kwargs and never used (confirmed via inspect.signature: the real signature
    # is time, redshift, mdot, vwind, delta, nn, mexp, eexp, eff, logepsb, logepse,
    # p, **kwargs). The auto-prior generator adds it for any BPL-ejecta model
    # regardless of whether the specific wrapper uses it; sampling it would waste a
    # dimension on a completely flat direction.
    del priors['vej_max_ratio']
    # mdot: tightened from the auto default (LogUniform 1e-5 - 10 Msun/yr) based on
    # the Step 1 sanity check, which found mdot ~ 1e-4 - 1e-3 Msun/yr in the right
    # ballpark against both radio and X-ray -- 2 dex of margin on each side of that.
    priors['mdot'] = bilby.core.prior.LogUniform(
        1e-6, 1e-1, name='mdot', latex_label=r'$\dot{M}~(M_\odot/\mathrm{yr})$')
    # p and nn: widened from the auto defaults (p: Uniform(2,4), nn: Uniform(6,14))
    # after the nlive=50 pilot pinned BOTH at their prior walls (p -> 2.0002,
    # nn -> 13.99 across the entire posterior). A profile-likelihood scan (holding
    # the other 5 free params at the pilot's posterior median) found this was a
    # real interior optimum being clipped, not a runaway: chi2 has a genuine
    # minimum at p ~ 1.95 (chi2 rises steeply on both sides), and at nn ~ 40-50
    # before asymptoting flat by nn ~ 100+ (not diverging). Widened with margin
    # around both findings; see CLAUDE_modelling_SOP.md Step 2 for the full
    # investigation and the p<2 physical caveat (marginal violation of the
    # p>2-for-finite-energy condition, not alarming this close to 2).
    priors['p'] = bilby.core.prior.Uniform(
        1.5, 4.0, name='p', latex_label=r'$p$')
    priors['nn'] = bilby.core.prior.Uniform(
        6.0, 100.0, name='nn', latex_label=r'$n$')
    return priors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--label', type=str, default='step2_wind_bpl_radio')
    parser.add_argument('--outdir', type=str, default=str(REPO_ROOT / 'csm_modeling' / 'outdir'))
    parser.add_argument('--clean', action='store_true', default=True)
    args = parser.parse_args()

    radio = pd.read_csv(REPO_ROOT / 'data' / 'radio_18ivc_data.csv')
    transient = build_transient(radio)
    priors = build_priors()
    free_params = [k for k, v in priors.items() if hasattr(v, 'sample')]
    print(f"Free parameters ({len(free_params)}): {free_params}")

    model_kwargs = dict(frequency=radio['freq'].values * 1e9, output_format='flux_density')

    t0 = timemod.time()
    result = redback.fit_model(
        transient=transient,
        model=wind_bpl_radio_multifreq,
        model_kwargs=model_kwargs,
        prior=priors,
        sampler='dynesty',
        nlive=args.nlive,
        outdir=args.outdir,
        label=args.label,
        plot=False,
        clean=args.clean,
    )
    dt = timemod.time() - t0
    print(f"Done in {dt/60:.1f} min ({dt:.0f} s). "
          f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")

    corner_path = f"{args.outdir}/{args.label}_corner.png"
    lc_path = f"{args.outdir}/{args.label}_lightcurve.png"
    result.plot_corner(filename=corner_path, show=False)
    result.plot_lightcurve(model=wind_bpl_radio_multifreq, filename=lc_path, show=False)
    print(f"Saved: {corner_path}")
    print(f"Saved: {lc_path}")

    mdot_post = result.posterior['mdot']
    print(f"mdot posterior: median={mdot_post.median():.3e}, "
          f"16-84%=[{mdot_post.quantile(0.16):.3e}, {mdot_post.quantile(0.84):.3e}] Msun/yr")


if __name__ == '__main__':
    main()
