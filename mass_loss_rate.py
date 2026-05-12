#!/usr/bin/env python3

'''
Calculate the progenitor mass loss rate from each SSA component at each epoch
using Weiler et al. (1986) equation 16.

The optical depth at 5 GHz is derived from the SSA peak frequency of each
component assuming free-free absorption turns the source on when tau_ff = 1
at the peak frequency:
    tau_5GHz = (nu_p / 5 GHz)^2.1

Equation 16 (Weiler+1986), simplified for cosmic abundances (Z=1, singly ionized):
    Mdot = 3.02e-6 * tau_5^0.5 * (w/10 km/s) * (v_e/1e4 km/s)^1.5
           * (t/45 d)^1.5 * (T/1e4 K)^0.675   [Msun/yr]

@author: Annika Deutsch
'''

import os
import glob
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl

sns.set_theme(style="white", context="paper")
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.bottom": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
})

blue = "#4C78A8"
pink = "#F5859E"


def tau_at_5GHz(nu_p, p=None, method="ff"):
    """
    Optical depth at 5 GHz derived from the SSA peak frequency.

    Two methods are supported:

    'ff'  (free-free): assumes tau_ff(nu_p) = 1 and tau_ff propto nu^{-2.1},
          so tau_ff(5 GHz) = (nu_p / 5)^2.1.

    'ssa' (synchrotron self-absorption): assumes tau_SSA(nu_p) = 1 and
          tau_SSA propto nu^{-(p+4)/2}, so tau_SSA(5 GHz) = (nu_p / 5)^((p+4)/2).

    Parameters
    ----------
    nu_p : float
        SSA peak frequency in GHz.
    p : float, optional
        Electron energy index. Required when method='ssa'.
    method : str
        'ff' for free-free (default) or 'ssa' for synchrotron self-absorption.

    Returns
    -------
    tau : float
        Optical depth at 5 GHz.
    """
    if method == "ff":
        return (nu_p / 5.0) ** 2.1
    elif method == "ssa":
        if p is None:
            raise ValueError("p (electron energy index) must be provided when method='ssa'")
        return (nu_p / 5.0) ** ((p + 4) / 2)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'ff' or 'ssa'.")


def weiler1986_mdot(tau_5, w, v_e, t, T=1e4):
    """
    Progenitor mass loss rate from Weiler et al. (1986) equation 16,
    simplified for cosmic abundances.

    Parameters
    ----------
    tau_5 : float
        Free-free optical depth at 5 GHz.
    w : float
        Stellar wind velocity in km/s.
    v_e : float
        Supernova ejecta velocity in km/s.
    t : float
        Age of the supernova in days.
    T : float
        Electron temperature in the wind in K (default 1e4 K).

    Returns
    -------
    Mdot : float
        Mass loss rate in Msun/yr.
    """
    return (3.02e-6
            * tau_5**0.5
            * (w / 10.0)
            * (v_e / 1e4)**1.5
            * (t / 45.0)**1.5
            * (T / 1e4)**0.675)


def main():
    parser = argparse.ArgumentParser(
        description="Compute progenitor mass loss rate from SSA component fits "
                    "using Weiler et al. (1986) equation 16.")

    parser.add_argument("--model_params_dir", type=str, default="model_params",
                        help="Directory containing best_fit_params_*_days.csv files "
                             "(default: model_params/).")
    parser.add_argument("--w", type=float, default=10.0,
                        help="Progenitor wind velocity in km/s (default: 10.0).")
    parser.add_argument("--v_e", type=float, default=1e4,
                        help="Ejecta velocity in km/s (default: 1e4).")
    parser.add_argument("--T", type=float, default=1e4,
                        help="Electron temperature in the wind in K (default: 1e4).")
    parser.add_argument("--t_offset", type=float, default=0.0,
                        help="Offset to add to phase to convert to age from explosion "
                             "in days (default: 0.0, i.e. phase = age).")
    parser.add_argument("--plot", action="store_true",
                        help="Plot mass loss rate vs epoch for each component.")
    parser.add_argument("--outfile", type=str, default="mass_loss_rates.csv",
                        help="Output CSV filename (default: mass_loss_rates.csv).")

    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.model_params_dir, "best_fit_params_*_days.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No best_fit_params_*_days.csv files found in {args.model_params_dir}")

    rows = []
    for f in csv_files:
        basename = os.path.basename(f)
        phase = float(basename.replace("best_fit_params_", "").replace("_days.csv", ""))
        age = phase + args.t_offset

        df = pd.read_csv(f, index_col=0)

        for comp in df.index:
            nu_p = df.loc[comp, "nu_p (GHz)"]
            p = df.loc[comp, "p"]
            tau_5 = tau_at_5GHz(nu_p)
            mdot = weiler1986_mdot(tau_5, w=args.w, v_e=args.v_e, t=age, T=args.T)
            rows.append({
                "phase (days)": phase,
                "age (days)": age,
                "component": comp,
                "nu_p (GHz)": nu_p,
                "p": p,
                "tau_5GHz": tau_5,
                "Mdot (Msun/yr)": mdot,
            })

        print(f"Phase {phase:.1f} d (age {age:.1f} d):")
        for comp in df.index:
            r = [r for r in rows if r["phase (days)"] == phase and r["component"] == comp][0]
            print(f"  {comp}: nu_p={r['nu_p (GHz)']:.2f} GHz, "
                  f"tau_5={r['tau_5GHz']:.3f}, "
                  f"Mdot={r['Mdot (Msun/yr)']:.2e} Msun/yr")

    results = pd.DataFrame(rows)
    outpath = os.path.join(os.getcwd(), args.outfile)
    results.to_csv(outpath, index=False)
    print(f"\nResults saved to {outpath}")

    if args.plot:
        components = results["component"].unique()
        colors = [blue, pink, "green", "purple", "orange"]
        fig, ax = plt.subplots(dpi=300, figsize=(8, 5))

        for i, comp in enumerate(components):
            sub = results[results["component"] == comp].sort_values("age (days)")
            ax.plot(sub["age (days)"], sub["Mdot (Msun/yr)"],
                    marker="o", ms=6, lw=1.5,
                    color=colors[i % len(colors)], label=comp)

        ax.set_xlabel("Age (days)")
        ax.set_ylabel(r"$\dot{M}$ ($M_\odot$ yr$^{-1}$)")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(
            rf"Mass loss rate ($w={args.w}$ km/s, $v_e={args.v_e:.0f}$ km/s, "
            rf"$T={args.T:.0e}$ K)")
        figpath = os.path.join(os.getcwd(), "figures", "mass_loss_rates.png")
        os.makedirs(os.path.dirname(figpath), exist_ok=True)
        fig.savefig(figpath, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {figpath}")


if __name__ == "__main__":
    main()
