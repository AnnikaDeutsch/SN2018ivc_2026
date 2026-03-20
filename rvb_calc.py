#!/usr/bin/env python3
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


start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Each time this script is run, create a new directory and store all outputs in it
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_dir = os.getcwd() # the location of this script

def main():
    parser = argparse.ArgumentParser(description="Calculate the radius, velocity, and B field of the emitting region given an SSA curve, either " \
    "reading in from a file or taking in the parameters directly from the command line.")
    parser.add_argument("--nu_peak", "-n", type=float, help="Peak frequency of the SSA curve in GHz.")
    parser.add_argument("--flux_peak", "-f", type=float, help="Peak flux density of the SSA curve in mJy.")
    parser.add_argument("--electron_index", "-p", required=True, type=float, help="Electron energy distribution index (p).")
    parser.add_argument("--K1", "-k", type=float, default=1.0, help="K1 constant for the SSA model.")
    parser.add_argument("--K2", "-l", type=float, default=1.0, help="K2 constant for the SSA model.")
    parser.add_argument("--time_obs", "-t", required=True, type=float, help="Time of observation in days post explosion.")
    parser.add_argument("--fmin", "-m", type=float, default=2.0, help="Minimum frequency for the SSA curve in GHz.")
    parser.add_argument("--fmax", "-M", type=float, default=275.0, help="Maximum frequency for the SSA curve in GHz.")
    args = parser.parse_args()

    p = args.electron_index
    t = args.time_obs

    # read in params from model independent config file
    config_path = os.path.join(base_dir, "model_indep_params.yml")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        D = config["physical"]["D"]["value"]
        D_scale = config["scales"]["D_scale"]["value"]
        nu_p_scale = config["scales"]["nu_p_scale"]["value"]
        F_p_scale = config["scales"]["F_p_scale"]["value"]
        vel_conv = config["conversion"]["cmday_to_kmsec"]["value"]

    # check if nu_peak and flux_peak need to be calculated. If so, calculate
    if args.nu_peak is None or args.flux_peak is None:
        print("Calculating nu_peak and flux_peak from SSA curve...")
        nu_peak, flux_peak = functions.find_peak_SSA(args.K1, args.K2, args.electron_index, args.fmin, args.fmax)
    else:
        nu_peak = args.nu_peak
        flux_peak = args.flux_peak
    
    # calculate R, v, and B
    R = functions.R_peak_SSA(p, flux_peak, D, nu_peak, F_p_scale, D_scale, nu_p_scale)
    v = (R/t) * vel_conv
    B = functions.B_peak_SSA(p, flux_peak, D, nu_peak, F_p_scale, D_scale, nu_p_scale)

    # nicely print results to command line
    print(f"\nAn SSA dominated synchrotron emission component with\n:")
    print(f"Electron index (p): {p}, peak frequency (nu_peak): {nu_peak:.2f} GHz, and peak flux density (flux_peak): {flux_peak:.2f} mJy\n")
    print(f"at time of observation (t): {t:.2f} days post explosion and a distance {D:.2f} Mpc, corresponds to the following physical parameters:")
    print("-------------------------------------------------------------")
    print(f"Radius (R): {R:.2e} cm")
    print(f"Velocity (v): {v:.2f} km/s")
    print(f"Magnetic Field (B): {B:.2f} G")

if __name__ == "__main__":
    main()
