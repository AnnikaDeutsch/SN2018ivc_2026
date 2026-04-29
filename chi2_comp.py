#!/usr/bin/env python3

'''
Script to take some radio SED data, fit 1, 2, and 3 SSA components to it,
compare the chi2 values of the fits, and determine which model is best supported by the data.

@author: Annika Deutsch
@date: 04/20/2026
@title: chi2_comp.py
@CPU: Apple M3
@Operating System: macOS 15.7.1 24G231
@Interpreter and version no.: Python 3.10.20
'''

from os import system, chdir, makedirs, getcwd
from sys import stdout
from glob import glob
import time
from datetime import datetime
import sys
import os
import subprocess
from pathlib import Path
import argparse
import yaml
import functions
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import pandas as pd

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
# colors
blue = "#4C78A8"
pink = "#F5859E"

start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Each time this script is run, create a new directory and store all outputs in it
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_dir = os.getcwd() # the location of this script

def main():
    parser = argparse.ArgumentParser(description="Take some radio SED data, fit 1, 2, and 3 SSA components to it," "\
    compare the chi2 values of the fits, and determine which model is best supported by the data.")
    parser.add_argument("datafile", help="Path to data file (csv with columns: phase, freq, flux, flux_err).")

    # --- fitting range ---
    parser.add_argument("--phase_lower", type=float, required=True)
    parser.add_argument("--phase_upper", type=float, required=True)
    parser.add_argument("--xmin", type=float, default=1.21)
    parser.add_argument("--xmax", type=float, default=40.0)
    parser.add_argument("--ymin", type=float, default=2.0)
    parser.add_argument("--ymax", type=float, default=22.0)

    # --- component guesses (3 values each) ---
    parser.add_argument("--comp1_guess", type=float, nargs=3, required=True,
                        metavar=("F_p1", "nu_p1", "alpha1"))
    parser.add_argument("--comp2_guess", type=float, nargs=3, required=True,
                        metavar=("F_p2", "nu_p2", "alpha2"))
    parser.add_argument("--comp3_guess", type=float, nargs=3, required=True,
                        metavar=("F_p3", "nu_p3", "alpha3"))

    # --- previous peak frequencies ---
    parser.add_argument("--nu1last", type=float, required=True)
    parser.add_argument("--nu2last", type=float, required=True)
    parser.add_argument("--nu3last", type=float, required=True)

    # --- optional parameters ---
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--amp_lower", type=float, default=3.0)

    # --- plotting options ---
    parser.add_argument("--plot", action="store_true", help="Plot the data with the best fit model.")

    # --- labels ---
    parser.add_argument("--label1", type=str, default="Component 1")
    parser.add_argument("--label2", type=str, default="Component 2")
    parser.add_argument("--label3", type=str, default="Component 3")

    # --- colors ---
    parser.add_argument("--color1", type=str, default="orange")
    parser.add_argument("--color2", type=str, default="green")
    parser.add_argument("--color3", type=str, default="purple")

    args = parser.parse_args()

    data = Table.read(args.datafile, format="csv")
    comp1_guess = args.comp1_guess
    comp2_guess = args.comp2_guess
    comp3_guess = args.comp3_guess

    # perform linear least squares fit for each model
    phase_lower = args.phase_lower
    phase_upper = args.phase_upper
    avg_phase = (phase_lower + phase_upper) / 2

    nup1last = args.nu1last
    nup2last = args.nu2last
    nup3last = args.nu3last

    fit_1comp, chi2_1comp = functions.one_comp_ls_fit(data, phase_lower, phase_upper, args.xmin, args.xmax,
                                                        comp1_guess, nup1last, offset=args.offset, amp_lower=args.amp_lower,
                                                        plot=False, label1=args.label1, color1=args.color1, chi2=True)
    
    fit_2comp, chi2_2comp = functions.two_comp_ls_fit(data, phase_lower, phase_upper, args.xmin, args.xmax,
                                                        comp1_guess, comp2_guess, nup1last, nup2last, offset=args.offset, 
                                                        amp_lower=args.amp_lower, plot=False, label1=args.label1, 
                                                        label2=args.label2, color1=args.color1, color2=args.color2, chi2=True)

    fit_3comp, chi2_3comp = functions.three_comp_ls_fit(data, phase_lower, phase_upper, args.xmin, args.xmax,
                                                        comp1_guess, comp2_guess, comp3_guess, nup1last, nup2last,
                                                        nup3last, offset=args.offset, amp_lower=args.amp_lower, plot=False,
                                                        label1=args.label1, label2=args.label2, label3=args.label3, 
                                                        color1=args.color1, color2=args.color2, color3=args.color3, chi2=True)

    # compare chi2 values of each fit, keep the one with the chi2 closest to 1
    fits = [fit_1comp, fit_2comp, fit_3comp]
    labels = [args.label1, args.label2, args.label3]
    chi2s = [chi2_1comp, chi2_2comp, chi2_3comp]
    closest_idx = min(range(len(chi2s)), key=lambda i: abs(chi2s[i] - 1.0))
    best_fit_params = fits[closest_idx]
    # turn best_fit_params into a dictionary by component label
    param_dict = {
        label: best_fit_params[i*3:(i+1)*3] for i, label in enumerate(labels[:closest_idx + 1])
    }

    # plot the data with that fit, print to terminal, and save fit params in a csv file
    print(f"Best fit model: {closest_idx + 1} component(s) with chi2 = {chi2s[closest_idx]:.2f}")
    print("-------------------------------------------------------------")
    for i in range(closest_idx + 1):
        print(f"Best-fit parameters for {labels[i]}:")
        print(f"Peak flux density: {param_dict[labels[i]][0]:.2f} mJy")
        print(f"Peak frequency: {param_dict[labels[i]][1]:.2f} GHz")
        print(f"Electron energy index: {param_dict[labels[i]][2]:.2f}")
        print(f"Spectral index (alpha): {(param_dict[labels[i]][2] - 1)/2:.2f}")
        print("------------------")
    print("-------------------------------------------------------------")

    df = pd.DataFrame.from_dict(param_dict, orient='index', columns=['F_p (mJy)', 'nu_p (GHz)', 'p'])
    df.to_csv(os.path.join(base_dir, "model_params", f"best_fit_params_{avg_phase}_days.csv"))

    if args.plot:
        data = data[((data['phase']>=phase_lower) & (data['phase']<=phase_upper))]
        # plot the data and the fit
        fig, ax = plt.subplots(dpi=300, figsize=(8,6))
        # plot data with error bars
        freq = data['freq']
        flux = data['flux']
        ax.errorbar(freq, flux, yerr=data['flux_err'], fmt='o', ms=7, capsize=2, elinewidth=1, color=blue, 
                    mec='black', mew=0.5,label='Data')
        # plot best fit curve
        freq_range = np.logspace(np.log10(min(freq)*0.5), np.log10(max(freq)*1.5), 200)
        # plot depending on how many components should be plotted
        if closest_idx == 0:
            best_fit = functions.F_SSA_Nayana(freq_range, *best_fit_params)
            ax.plot(freq_range, best_fit, color=pink, lw=2.5, label='Best Fit')
            comp1_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[0], best_fit_params[1], best_fit_params[2])
            ax.plot(freq_range, comp1_curve, label=args.label1, color=args.color1, linestyle='--', alpha=0.5, lw=1.8)
        elif closest_idx == 1:
            best_fit = functions.F_SSA_Nayana_2comp(freq_range, *best_fit_params)
            ax.plot(freq_range, best_fit, color=pink, lw=2.5, label='Best Fit')
            # plot component curves
            comp1_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[0], best_fit_params[1], best_fit_params[2])
            comp2_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[3], best_fit_params[4], best_fit_params[5])
            ax.plot(freq_range, comp1_curve, label=args.label1, color=args.color1, linestyle='--', alpha=0.5, lw=1.8)
            ax.plot(freq_range, comp2_curve, label=args.label2, color=args.color2, linestyle='--', alpha=0.5, lw=1.8)
        else:
            best_fit = functions.F_SSA_Nayana_3comp(freq_range, *best_fit_params)
            ax.plot(freq_range, best_fit, color=pink, lw=2.5, label='Best Fit')
            # plot component curves
            comp1_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[0], best_fit_params[1], best_fit_params[2])
            comp2_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[3], best_fit_params[4], best_fit_params[5])
            comp3_curve = functions.F_SSA_Nayana(freq_range, best_fit_params[6], best_fit_params[7], best_fit_params[8])
            ax.plot(freq_range, comp1_curve, label=args.label1, color=args.color1, linestyle='--', alpha=0.5, lw=1.8)
            ax.plot(freq_range, comp2_curve, label=args.label2, color=args.color2, linestyle='--', alpha=0.5, lw=1.8)
            ax.plot(freq_range, comp3_curve, label=args.label3, color=args.color3, linestyle='--', alpha=0.5, lw=1.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Flux Density (mJy)')
        ax.set_title(f"SED at {avg_phase:.1f} days")
        ax.legend() 
        fig.savefig(os.path.join(base_dir, "figures", f"best_fit_model_{avg_phase}_days.png"), bbox_inches='tight', dpi=300) 

if __name__ == "__main__":
    main()
