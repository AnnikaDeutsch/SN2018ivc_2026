#!/usr/bin/env python3

'''
Script to take in a temperature, ejecta mass, velocity, and frequency,
and return the estimated time for the ejecta to become optically thin to FFA

@author: Annika Deutsch
@date: 04/27/2026
@title: tffa_calc.py
@CPU: Apple M3
@Operating System: macOS 15.7.1 24G231
@Interpreter and version no.: Python 3.10.20
'''

import time
from datetime import datetime
import os
import argparse
import functions



start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Each time this script is run, create a new directory and store all outputs in it
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_dir = os.getcwd() # the location of this script

def main():
    parser = argparse.ArgumentParser(description="Take T, M, v, and nu, and return time to become optically thin to FFA.")
    parser.add_argument("-t", "--temp", type=float, required=True, help="Temperature of the ejecta in K.")
    parser.add_argument("-m", "--mass", type=float, required=True, help="Ejecta mass in solar masses.")
    parser.add_argument("-v", "--velocity", type=float, required=True, help="Ejecta velocity in km/s.")
    parser.add_argument("-n", "--frequency", type=float, required=True, help="Frequency in GHz.")
    args = parser.parse_args()

    T = args.temp
    M = args.mass
    v = args.velocity
    nu = args.frequency

    tffa = functions.t_ffa_optically_thin(T, M, v, nu)
    print(f"----Time to become optically thin to FFA: {tffa:.2f} years----")
    print(f"Temperature: {T:.2e} K")
    print(f"Ejecta mass: {M:.2f} solar masses")
    print(f"Ejecta velocity: {v:.2f} km/s")
    print(f"Peak Frequency: {nu:.2f} GHz")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
