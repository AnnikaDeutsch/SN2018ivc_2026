'''
Module to hold all general use plotting, modelling, and physical parameter calculation functions.
Created September 2023

@author: Annika Deutsch
@date: 09/2023
@title: functions.py
@CPU: Apple M3
@Operating System: macOS 15.7.1 24G231
@Interpreter and version no.: Python 3.12.2
'''

import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.io import ascii
import corner
import os
from timeit import default_timer as timer
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.lines as mlines
from scipy.optimize import least_squares, curve_fit
from scipy.stats import f
from astropy.table import Table, vstack
import pandas as pd
import lmfit
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import yaml

# define colors used in dark mode
lime = '#e4ffba'
lemon = '#fef6c7'
bora = '#b186ce'
sky = '#b2eaf0'
strawberry = '#ff9cff'
coral = '#feb5a4'


# Set Matplotlib style parameters for dark background
rcParams['figure.facecolor'] = 'white'  # Dark background color
rcParams['axes.facecolor'] = 'white'
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['text.color'] = 'black'
rcParams['axes.titlecolor'] = 'black'

# Other settings to set
#cmap = LinearSegmentedColormap.from_list('custom_cmap', [coral, bora], N=256)
cmap = "viridis"


#---------------Plotting Functions---------------#
def plot_single_epoch(phase_lower, phase_upper, data, xlim, ylim,
                      comp1=None, comp2=None, comp3=None,
                      comp1_type='SSA', comp2_type='FFA', comp3_type='SSA',
                      color=None, ax=None, freq_scale=[10,10,10]):
    """Plot SED for a specific epoch, with up to 3 components fit by eye"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    else:
        fig = None

    if color == None:
        mask = ((data['phase'] > phase_lower) & (data['phase'] < phase_upper))
        data_epoch = data[mask]

        avg_epoch = np.mean(data_epoch['phase'])

        # color code by telescope
        ALMA_mask = (data_epoch['telescope'] == 'ALMA')
        VLA_mask = (data_epoch['telescope'] == 'VLA')

        x_ALMA = data_epoch['freq'][ALMA_mask]
        y_ALMA = data_epoch['flux'][ALMA_mask]
        yerr_ALMA = data_epoch['flux_err'][ALMA_mask]

        x_VLA = data_epoch['freq'][VLA_mask]
        y_VLA = data_epoch['flux'][VLA_mask]
        yerr_VLA = data_epoch['flux_err'][VLA_mask]

        if (comp1 != None) or (comp2 != None) or (comp3 != None):
            if len(x_ALMA) > 0:
                ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^', color='r', label='ALMA data')
            ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s', color='b', label='VLA data')
        else:
            if len(x_ALMA) > 0:
                ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^-', color='r', label='ALMA data')
            ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s-', color='b', label='VLA data')

        freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
        total_model = None

        if comp1 != None:
            if comp1_type == 'SSA':
                K11, K21, alpha1 = comp1
                F_by_eye_1 = F_SSA(freq_range, K11, K21, alpha1, freq_scale=freq_scale[0])
            elif comp1_type == 'FFA':
                K11, K21, alpha1 = comp1
                F_by_eye_1 = F_FFA(freq_range, K11, K21, alpha1, freq_scale=freq_scale[0])
            ax.plot(freq_range, F_by_eye_1, label=f'{comp1_type} Comp 1', color='orange', linestyle='--')
            total_model = F_by_eye_1 if total_model is None else total_model + F_by_eye_1

        if comp2 != None:
            if comp2_type == 'SSA':
                K12, K22, p2 = comp2
                F_by_eye_2 = F_SSA(freq_range, K12, K22, p2, freq_scale=freq_scale[1])
            elif comp2_type == 'FFA':
                K12, K22, p2 = comp2
                F_by_eye_2 = F_FFA(freq_range, K12, K22, p2, freq_scale=freq_scale[1])
            ax.plot(freq_range, F_by_eye_2, label=f'{comp2_type} Comp 2', color='magenta', linestyle='--')
            total_model = F_by_eye_2 if total_model is None else total_model + F_by_eye_2

        if comp3 != None:
            if comp3_type == 'SSA':
                K13, K23, p3 = comp3
                F_by_eye_3 = F_SSA(freq_range, K13, K23, p3, freq_scale=freq_scale[2])
            elif comp3_type == 'FFA':
                K13, K23, p3 = comp3
                F_by_eye_3 = F_FFA(freq_range, K13, K23, p3, freq_scale=freq_scale[2])
            ax.plot(freq_range, F_by_eye_3, label=f'{comp3_type} Comp 3', color='green', linestyle='--')
            total_model = F_by_eye_3 if total_model is None else total_model + F_by_eye_3

        if total_model is not None and ((comp1 != None) + (comp2 != None) + (comp3 != None) > 1):
            ax.plot(freq_range, total_model, label='Combined', color='cyan', linestyle='-')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Flux Density [mJy]')
        ax.set_title(f'SN 2018ivc SED at {avg_epoch:.2f} days')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Components", title_fontsize='large')
        if fig is not None:
            fig.tight_layout()
    else:
        mask = ((data['phase'] > phase_lower) & (data['phase'] < phase_upper))
        data_epoch = data[mask]

        avg_epoch = np.mean(data_epoch['phase'])

        # color code by telescope
        ALMA_mask = (data_epoch['telescope'] == 'ALMA')
        VLA_mask = (data_epoch['telescope'] == 'VLA')

        x_ALMA = data_epoch['freq'][ALMA_mask]
        y_ALMA = data_epoch['flux'][ALMA_mask]
        yerr_ALMA = data_epoch['flux_err'][ALMA_mask]

        x_VLA = data_epoch['freq'][VLA_mask]
        y_VLA = data_epoch['flux'][VLA_mask]
        yerr_VLA = data_epoch['flux_err'][VLA_mask]
    
        if (comp1 != None) or (comp2 != None) or (comp3 != None):
            freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            total_model = None

            if comp1 != None:
                if comp1_type == 'SSA':
                    K11, K21, alpha1 = comp1
                    F_by_eye_1 = F_SSA(freq_range, K11, K21, alpha1, freq_scale=freq_scale[0])
                elif comp1_type == 'FFA':
                    K11, K21, alpha1 = comp1
                    F_by_eye_1 = F_FFA(freq_range, K11, K21, alpha1, freq_scale=freq_scale[0])
                total_model = F_by_eye_1 if total_model is None else total_model + F_by_eye_1

            if comp2 != None:
                if comp2_type == 'SSA':
                    K12, K22, p2 = comp2
                    F_by_eye_2 = F_SSA(freq_range, K12, K22, p2, freq_scale=freq_scale[1])
                elif comp2_type == 'FFA':
                    K12, K22, p2 = comp2
                    F_by_eye_2 = F_FFA(freq_range, K12, K22, p2, freq_scale=freq_scale[1])
                total_model = F_by_eye_2 if total_model is None else total_model + F_by_eye_2

            if comp3 != None:
                if comp3_type == 'SSA':
                    K13, K23, p3 = comp3
                    F_by_eye_3 = F_SSA(freq_range, K13, K23, p3, freq_scale=freq_scale[2])
                elif comp3_type == 'FFA':
                    K13, K23, p3 = comp3
                    F_by_eye_3 = F_FFA(freq_range, K13, K23, p3, freq_scale=freq_scale[2])
                total_model = F_by_eye_3 if total_model is None else total_model + F_by_eye_3

            ax.plot(freq_range, total_model, color=color, linestyle='-')
            ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^', color=color, markersize=10)
            ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s', color=color, label=f'{avg_epoch:.2f}', markersize=10)
        else:
            ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^-', color=color, markersize=10)
            ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s-', color=color, label=f'{avg_epoch:.2f}', markersize=10)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Frequency [GHz]', fontsize=14)
        ax.set_ylabel('Flux Density [mJy]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        if fig is not None:
            fig.tight_layout()


def plot_data(ax, sm, data, mode, scaled=False, **kwargs):
    telescope_marker_dict = {'VLA':('s', lime), 'ALMA':('o', bora), 'e-MERLIN':('d', coral)}

    for row in data:
        if mode == 'lc':
            x = row['phase']
            # set marker color based on frequency
            freq = row['freq']
            colorval = sm.to_rgba(freq)
        if mode == 'sed':
            x = row['freq']
            # set marker color based on time
            time = row['phase']
            colorval = sm.to_rgba(time)

        telescope = row['telescope']
        marker = telescope_marker_dict[telescope][0]
        
        if scaled:
            flux = row['scaled_flux']
            err = row['scaled_flux_err']
        else:
            flux = row['flux']
            err = row['flux_err']

        if colorval is None:
            colorval = telescope_marker_dict[telescope][1]

        ax.errorbar(x, flux, yerr=err, marker=marker, c=colorval)
    return


def make_plot(data, mode, title='', xlabel='', ylabel='', freq_vals=np.linspace(0, 300, 300), cbar=True, scaled=False, models=None,
            params=None, model_names=None, plot_models=False, modelcolors=None, vline=None, vlinecolors=None, ylim=None):
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = fig.add_subplot(111)
    ax.grid(False)

    if cbar:
        min_freq = np.min(freq_vals)
        max_freq = np.max(freq_vals)

        min_time = np.min(data['phase'])
        max_time = np.max(data['phase'])
        # get the scalar map, plot the data using the plot_data function
        sm = cmap_setup(mode, cmap=cmap, min_freq=min_freq, 
                        max_freq=max_freq, min_time=min_time, max_time=max_time)
        plot_data(ax, sm, data, mode, scaled=scaled)

        # set up colorbar
        if mode == 'lc':
            fig = ax.get_figure()
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, label=r'$\nu$ [GHz]')
        elif mode == 'sed':
            fig.colorbar(sm, fraction=0.046, label='time [Days]')
    else:
        sm = None
        plot_data(ax, sm, data, mode, scaled=scaled)

    # set axis scales to log
    ax.set_yscale('log')
    ax.set_xscale('log')

    #Label axes, set axis limits etc.
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if scaled:
        ax.set_ylabel('Scaled Flux Density (mJy)')
        ax.set_title('Scaled to 3 GHz')
    else:
        ax.set_ylabel(ylabel)

    if mode == 'lc':
        x = data['phase']
    elif mode == 'sed':
        x = data['freq']

    if models!=None:
        for i, (model,param,model_name,color) in enumerate(zip(models,params,model_names,modelcolors)):
            plot_model(model, param, x, ax, model_name, color, freq_vals=freq_vals, ylim=ylim)

    if vline != None:
        for i, (line, linecolor) in enumerate(zip(vline, vlinecolors)):
            ax.axvline(x=line, color=linecolor, linestyle='--')


def cmap_setup(mode, cmap=cmap, min_freq=0, max_freq=400, min_time=1360, max_time=1370):
    '''
    color markers by frequency/time
    '''
    if mode == 'lc':
        cNorm  = colors.Normalize(vmin=min_freq, vmax=max_freq)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        sm = scalarMap
        sm._A = []
    elif mode == 'sed':
        cNorm  = colors.Normalize(vmin=min_time, vmax=max_time)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        sm = scalarMap
        sm._A = []
    
    return sm   


def plot_model(model, params, x, ax, label, modelcolor, freq_vals=np.linspace(0, 300, 300), ylim=None):
    '''
    define model plotting function to be incorporated into makeplot()
    '''

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]

    fit = model(freq_vals, *params)
    ax.plot(freq_vals, fit, label=label, color=modelcolor)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.legend()
    return


def make_chain_plot(chain, chain_cut):
    niters = chain.shape[1]
    ndim = chain.shape[2]

    fig, axes = plt.subplots(ndim, 1, sharex=True, dpi=200)
    fig.set_size_inches(7, 12)

    param_names = ['$K_1$', '$K_2$', r'$p$', r'$a$', r'$b$']

    for i, (ax, pname) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, :, i].T, linestyle='-', color='steelblue', alpha=0.3)
        ax.set_ylabel(pname)
        ax.axvline(chain_cut, color='coral', linestyle='--')
        ax.set_xlim(0, niters)

    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()


def make_corner_plot(good_chain, savefile='corner.png'):
    ndim = good_chain.shape[2]
    param_names = ['$K_1$', '$K_2$', r'$p$', r'$a$', r'$b$']

    fig = corner.corner(
        good_chain.reshape((-1, ndim)),
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True
    )

    plt.savefig(savefile, dpi=300, bbox_inches='tight')

#---------------Modelling Functions---------------#
def find_peak_SSA(K1,K2,p,freqmin,freqmax,freq_scale=10):
    '''find the peak frequecny and flux of a single component SSA curve'''
    freq_range = np.logspace(np.log10(freqmin), np.log10(freqmax), 1000)
    flux = F_SSA(freq_range, K1, K2, p, freq_scale=freq_scale)
    peak_idx = np.argmax(flux)
    return freq_range[peak_idx], flux[peak_idx]


def find_peak_FFA(K1,K2,alpha,freqmin,freqmax,freq_scale=10):
    '''find the peak frequecny and flux of a single component FFA curve'''
    freq_range = np.logspace(np.log10(freqmin), np.log10(freqmax), 1000)
    flux = F_FFA(freq_range, K1, K2, alpha, freq_scale=freq_scale)
    peak_idx = np.argmax(flux)
    return freq_range[peak_idx], flux[peak_idx]


def time_slope_for_freq(data, nu_target, tol=1e-3):
    """
    Fit log F vs log t at approximately fixed frequency nu_target.
    Returns the slope d(log F) / d(log t).
    """
    mask = (np.abs(data['freq'] - nu_target) < tol) & (data['flux'] > 0)
    t = data['phase'][mask]
    F = data['flux'][mask]

    if len(t) < 2:
        raise ValueError(f"Not enough epochs at {nu_target} GHz to fit a time slope.")

    logt = np.log10(t)
    logF = np.log10(F)
    slope, intercept = np.polyfit(logt, logF, 1)
    return slope


def F_SSA(freq, K1, K2, p, freq_scale=10):
    """
    Calculate the flux density using Synchotron Self Absorption (SSA) model. (based on Chandra 2018)

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    p (float): Power-law index.
    freq_scale (float): Frequency scale for normalization (default is 10 GHz).

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    tau = K2 * (freq/freq_scale)**(-(p + 4) / 2)
    F = K1 * (freq/freq_scale)**(5/2) * (1 - np.exp(-tau))
    return F


def F_SSA_time(freq, time, K1, K2, p, a, b, freq_scale=10):
    """
    Calculate the flux density using Synchotron Self Absorption (SSA) model, time dependent. (based on Chandra 2018)

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    time (float or numpy.ndarray): Time values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    p (float): Power-law index.
    a (float): Time evolution index for the flux density.
    b (float): Time evolution index for the optical depth.
    freq_scale (float): Frequency scale for normalization (default is 10 GHz).

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    tau = K2 * (freq/freq_scale)**(-(p + 4) / 2) * time**(-(a+b))
    F = K1 * (freq/freq_scale)**(5/2) * (time**a) * (1 - np.exp(-tau))
    return F    


def F_FFA(freq, K1, K2, alpha, freq_scale=10):
    """
    Calculate the flux density using Free Free Absorption (FFA) model. Based on Chandra 2018

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    alpha (float): Spectral index for the flux density.
    freq_scale (float): Frequency scale for normalization (default is 10 GHz).

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    tau = K2 * (freq/freq_scale)**(-2.1)
    F = K1 * (freq/freq_scale)**(-alpha) * (np.exp(-tau))
    return F


def F_FFA_time(freq, time, K1, K2, alpha, beta, delta, freq_scale=10):
    """
    Calculate the flux density using Free Free Absorption (FFA) model. Based on Chandra 2018

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    time (float or numpy.ndarray): Time values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    alpha (float): Spectral index for the flux density.
    beta (float): Time evolution index for the flux density.
    delta (float): Time evolution index for the optical depth.
    freq_scale (float): Frequency scale for normalization (default is 10 GHz).

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    tau = K2 * (freq/freq_scale)**(-2.1) * time**(-delta)
    F = K1 * (freq/freq_scale)**(-alpha) * (time**(-beta)) * (np.exp(-tau))
    return F


def F_SSA_Nayana(nu, F_p, nu_p, alpha):
    """Calculate the flux density using the SSA, single epoch model from Nayana et al. 2022

    Parameters: 
    nu (float or numpy.ndarray): Frequency values.
    F_p (float): Peak flux density.
    nu_p (float): Peak frequency.
    alpha (float): Spectral index for the flux density.
    
    """
    exp = (1 - np.exp(-(nu/nu_p)**(-(5-2*alpha)/2)))
    F = 1.582 * F_p * (nu/nu_p)**(5/2) * exp
    return F


def calc_params(data, model, initial_guess, bounds):
    """
    Find the best fit parameters for either SSA or FFA (optionally time dependent) using curve_fit.
    These will be supplied as the inital guess for the MCMC fitting.
    Parameters:
      data (astropy table or pandas dataframe): The data to fit, must contain columns 'freq', 'phase', 'flux', and 'flux_err'.
      model (function): The model function to fit, either F_SSA, F_SSA_time, F_FFA, or F_FFA_time.
      initial_guess (tuple): Initial guess for the parameters to be fitted, in the order (K1, K2, p/alpha, a/beta, b/delta).
      bounds (tuple): Bounds for the parameters to be fitted, in the order ((K1_min, K2_min, p/alpha_min, a/beta_min, b/delta_min), (K1_max, K2_max, p/alpha_max, a/beta_max, b/delta_max)).

    Returns:
      results (dict): A dictionary containing the best fit parameters and their uncertainties, in the format {'K1': (K1_fit, K1_err), 'K2': (K2_fit, K2_err), 'p/alpha': (p_fit, p_err), 'a/beta': (a_fit, a_err), 'b/delta': (b_fit, b_err)}.
    """
    # define variables from data
    freq = data['freq'].astype(float)
    time = data['phase'].astype(float)
    flux = data['flux'].astype(float)
    flux_err = data['flux_err'].astype(float)

    # Wrap the model so that curve_fit sees only (freq, params...)
    def wrapped_model(freq, K1, K2, p, a, b):
        return model(freq, time, K1, K2, p, a, b)

    params, covariance = curve_fit(
        wrapped_model,
        freq,
        flux,
        p0=initial_guess,
        bounds=bounds,
        sigma=flux_err,
        absolute_sigma=True
    )

    K1_fit, K2_fit, p_fit, a_fit, b_fit = params
    K1_err, K2_err, p_err, a_err, b_err = np.sqrt(np.diag(covariance))

    results = {
        'K1': (K1_fit, K1_err),
        'K2': (K2_fit, K2_err),
        'p':  (p_fit,  p_err),
        'a':  (a_fit,  a_err),
        'b':  (b_fit,  b_err),
    }

    return results


def lnprior(theta):
    # uniform prior
    K1, K2, p, a, b = theta

    if not (0 < K1 < 1e5):
        return -np.inf
    if not (0 < K2 < 1e5):
        return -np.inf
    if not (1.0 < p < 4.0):
        return -np.inf
    if not (-10 < a < 10):
        return -np.inf
    if not (-10 < b < 10):
        return -np.inf

    return 0.0


def lnlike(theta, nu, time, F, F_err, model_type=F_SSA_time):
    K1, K2, p, a, b = theta

    model = model_type(nu, time, K1, K2, p, a, b)
    inv_sigma2 = 1.0/F_err**2

    return -0.5*(np.sum((F-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, nu, time, F, F_err, model_type=F_SSA_time):
    K1, K2, p, a, b = theta

    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, nu, time, F, F_err, model_type)


def get_starting_pos(params, nwalkers, ndim=5):
    pos = [np.asarray(params) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    return pos


def run_mcmc(params, data, niters=1000, nthreads=1, nwalkers=200, ndim=5):
    nu = data['freq']
    F = data['flux']
    F_err = data['flux_err']
    time = data['phase']
    
    pos = get_starting_pos(params, nwalkers, ndim=ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(nu, time, F, F_err),threads=nthreads)
    
    start = timer()
    sampler.run_mcmc(pos, niters)
    end = timer()
    
    print("Computation time: %f s"%(end-start))
    
    return sampler


def get_best_params(chain):
    ndim = chain.shape[2] # 3 in this case
    chain = chain.reshape((-1, ndim))
    # median, +1σ, –1σ
    percentiles = np.percentile(chain, [16, 50, 84], axis=0)
    vals = [(p50, p84 - p50, p50 - p16)
            for p16, p50, p84 in zip(*percentiles)]
    param_names = ['$K_1$', '$K_2$','$p$', '$a$', '$b$']
    param_dict = dict(zip(param_names,vals))
    return param_dict


def load_model_indep_params(config_path):
    """
    Load in model independent parameters from the config file in the directory
    Parameters:
      config_path (str): Path to the config file, should be in the same directory as the data file and the notebook.
    Returns:
      D (float): Distance to the supernova in cm.
      D_scale (float): Distance scale for normalization, in cm.
      nu_p_scale (float): Peak frequency scale for normalization, in GHz.
      F_p_scale (float): Peak flux density scale for normalization, in mJy.
      vel_conv (float): Velocity conversion factor from cm/day to km/s.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        D = config["physical"]["D"]["value"]
        D_scale = config["scales"]["D_scale"]["value"]
        nu_p_scale = config["scales"]["nu_p_scale"]["value"]
        F_p_scale = config["scales"]["F_p_scale"]["value"]
        vel_conv = config["conversion"]["cmday_to_kmsec"]["value"]
    return D, D_scale, nu_p_scale, F_p_scale, vel_conv

#---------------Physical Parameter Calculations---------------#
def B_peak_SSA(p, F_p, D, nu_p, F_p_scale, D_scale, nu_p_scale):
    '''B field calculation for SSA dominated absorption, as from eqn 11 from Chandra 2018'''
    power = (-4 / ((2 * p) + 13))
    B = 0.58 * ((p - 2)**power) * ((F_p/F_p_scale)**(power/2)) * ((D/D_scale)**power) * (nu_p/nu_p_scale)
    return B

def R_peak_SSA(p, F_p, D, nu_p, F_p_scale, D_scale, nu_p_scale):
    '''shock radius for SSA dominated absorption'''
    R = 8.8*10**15 * ((p - 2)**(-1 / ((2 * p) + 13))) * ((F_p/F_p_scale)**((p + 6)/((2 * p) + 13))) * ((D/D_scale)**(((2 * p) + 12)/((2 * p) + 13))) * ((nu_p/nu_p_scale)**(-1))
    return R

def Mdot_peak_SSA(m_H, B, t, v_wind, B_scale, t_scale, v_wind_scale):
    '''mass loss rate at a given epoch for SSA dominated absorption'''
    Mdot = ((6*10**(-7))/(m_H**2)) * ((B/B_scale)**2) * ((t/t_scale)**2) * ((v_wind/v_wind_scale))
    return Mdot

def Mdot_peak_FFA(tau, v_ej, t, T_e, v_wind, v_ej_scale, t_scale, T_e_scale, v_wind_scale):
    '''mass loss rate at a given epoch for FFA dominated absorption, assuming fully, singly ionized wind'''
    Mdot = (4.76 * 10**(-5)) * (tau**(0.5)) * ((v_ej/v_ej_scale)**1.5) * ((t/t_scale)**1.5) * ((T_e/T_e_scale)**0.675) * (v_wind/v_wind_scale)
    return Mdot

def Mdot_peak_mult(nu_p, T_e, v_ej, t):
    '''Mass loss rate considering multiple absorption components from Chandra et al. 2020'''
    vw1 = 2
    Mdot_scale = 10**-3
    Mdot = Mdot_scale * vw1 * (7.5*10**-2) * ((nu_p)**1.06) * ((T_e/(10**4))**0.67) * (v_ej/(10**4)) * ((t/1000)**1.5)
    return Mdot 

def nu_syn(epsilon_B, Mdot, v_wind, t, epsilon_B_scale, Mdot_scale, v_wind_scale):
    '''frequency if synchrotron cooling is dominant at a given epoch'''
    nu_syn = 240.0 * ((epsilon_B/epsilon_B_scale)**(-3.0/2.0)) * ((Mdot/Mdot_scale)**(-3.0/2.0)) * ((v_wind/v_wind_scale)**(3.0/2.0)) * (t/60.0)
    return nu_syn

def t_syn_ratio(epsilon_B, Mdot, v_wind, nu, t, epsilon_B_scale, Mdot_scale, v_wind_scale, nu_scale):
    '''cooling timescale if synchrotron cooling dominant'''
    t_syn_ratio = 2.0 * ((epsilon_B/epsilon_B_scale)**(-3.0/4.0)) * ((Mdot/Mdot_scale)**(-3.0/4.0)) * ((v_wind/v_wind_scale)**(3.0/4.0)) * ((nu/nu_scale)**(-0.5)) * ((t/10.0)**(0.5))
    return t_syn_ratio

def nu_IC(epsilon_B, Mdot, v_wind, v_ej, L_bol, t, epsilon_B_scale, Mdot_scale, v_wind_scale, v_ej_scale, L_bol_scale):
    '''frequency if inverse Compton cooling is dominant at a given epoch'''
    nu_IC = 8.0 * ((epsilon_B/epsilon_B_scale)**(0.5)) * ((Mdot/Mdot_scale)**(0.5)) * ((v_wind/v_wind_scale)**(-0.5)) * ((v_ej/v_ej_scale)**(4.0)) * (L_bol/L_bol_scale) * (t/60.0)
    return nu_IC

def t_IC_ratio(L_bol, epsilon_B, Mdot, v_wind, v_ej, nu, t, L_bol_scale, epsilon_B_scale, Mdot_scale, v_wind_scale, v_ej_scale, nu_scale):
    '''cooling timescale if inverse Compton cooling dominant'''
    t_IC_ratio = 0.18 * ((L_bol/L_bol_scale)**(-1.0)) * ((epsilon_B/epsilon_B_scale)**(1.0/4.0)) * ((Mdot/Mdot_scale)**(1.0/4.0)) * ((v_wind/v_wind_scale)**(-1.0/4.0)) * ((v_ej/v_ej_scale)**(2.0)) * ((nu/nu_scale)**(-0.5)) * ((t/10.0)**(0.5))
    return t_IC_ratio