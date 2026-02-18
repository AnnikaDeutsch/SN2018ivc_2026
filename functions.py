'''
.py file to hold all general use plotting and modelling functions 
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
cmap = LinearSegmentedColormap.from_list('custom_cmap', [coral, bora], N=256)


#---------------Plotting Functions---------------#
def plot_single_epoch(phase_lower, phase_upper, data, xlim, ylim, comp1=None, comp2=None, color=None):
    """Plot SED for a specific epoch, with a by-eye 2 component fit"""
    if color == None:
        fig, ax = plt.subplots(figsize=(8,6), dpi=300)
        mask = ((data['phase'] > phase_lower) & (data['phase'] < phase_upper))
        data_epoch = data[mask]

        # color code by telescope
        ALMA_mask = (data_epoch['telescope'] == 'ALMA')
        VLA_mask = (data_epoch['telescope'] == 'VLA')

        x_ALMA = data_epoch['freq'][ALMA_mask]
        y_ALMA = data_epoch['flux'][ALMA_mask]
        yerr_ALMA = data_epoch['flux_err'][ALMA_mask]

        x_VLA = data_epoch['freq'][VLA_mask]
        y_VLA = data_epoch['flux'][VLA_mask]
        yerr_VLA = data_epoch['flux_err'][VLA_mask]

        ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^', color='r', label='ALMA data')
        ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s', color='b', label='VLA data')

        if comp1 != None:
            # fit the F_SSA model by eye with 2 components
            freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            K11, K21, alpha1 = comp1
            F_by_eye_low = F_FFA(freq_range, K11, K21, alpha1)
            ax.plot(freq_range, F_by_eye_low, label='Component 1', color='orange', linestyle='--')


        if comp2 != None:
            # fit the F_SSA model by eye with 2 components
            freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            K12, K22, p = comp2
            F_by_eye_high = F_SSA(freq_range, K12, K22, p)
            ax.plot(freq_range, F_by_eye_high, label='Component 2', color='magenta', linestyle='--') 

        if (comp1 != None) and (comp2 != None):
            ax.plot(freq_range, F_by_eye_low + F_by_eye_high, label='Combined', color='cyan', linestyle='-')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Flux Density [mJy]')
        avg_epoch = (phase_lower + phase_upper) / 2
        ax.set_title(f'SN 2018ivc SED at {avg_epoch} days')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8,6), dpi=300)
        mask = ((data['phase'] > phase_lower) & (data['phase'] < phase_upper))
        data_epoch = data[mask]

        # color code by telescope
        ALMA_mask = (data_epoch['telescope'] == 'ALMA')
        VLA_mask = (data_epoch['telescope'] == 'VLA')

        x_ALMA = data_epoch['freq'][ALMA_mask]
        y_ALMA = data_epoch['flux'][ALMA_mask]
        yerr_ALMA = data_epoch['flux_err'][ALMA_mask]

        x_VLA = data_epoch['freq'][VLA_mask]
        y_VLA = data_epoch['flux'][VLA_mask]
        yerr_VLA = data_epoch['flux_err'][VLA_mask]

        ax.errorbar(x_ALMA, y_ALMA, yerr=yerr_ALMA, fmt='^', color=color, label='ALMA data')
        ax.errorbar(x_VLA, y_VLA, yerr=yerr_VLA, fmt='s', color=color, label='VLA data')

        if comp1 != None:
            # fit the F_SSA model by eye with 2 components
            freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            K11, K21, alpha1 = comp1
            F_by_eye_low = F_FFA(freq_range, K11, K21, alpha1)
            ax.plot(freq_range, F_by_eye_low, label='Component 1', color=color, linestyle='--')


        if comp2 != None:
            # fit the F_SSA model by eye with 2 components
            freq_range = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            K12, K22, p = comp2
            F_by_eye_high = F_SSA(freq_range, K12, K22, p)
            ax.plot(freq_range, F_by_eye_high, label='Component 2', color=color, linestyle='--') 

        if (comp1 != None) and (comp2 != None):
            ax.plot(freq_range, F_by_eye_low + F_by_eye_high, label='Combined', color=color, linestyle='-')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Flux Density [mJy]')
        avg_epoch = (phase_lower + phase_upper) / 2
        ax.set_title(f'SN 2018ivc SED at {avg_epoch} days')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        fig.tight_layout()


def plot_data(ax, sm, data, mode, scaled=False, **kwargs):
    telescope_marker_dict = {'VLA':('s', lime), 'ALMA':('o', bora), 'e-MERLIN':('d', coral)}

    for row in data:
        if mode == 'lc':
            x = row['time']
            # set marker color based on frequency
            freq = row['freq']
            colorval = sm.to_rgba(freq)
        if mode == 'sed':
            x = row['freq']
            # set marker color based on time
            time = row['time']
            # colorval = sm.to_rgba(time)

        telescope = row['telescope']
        marker = telescope_marker_dict[telescope][0]
        color = telescope_marker_dict[telescope][1]
        
        if scaled:
            flux = row['scaled_flux']
            err = row['scaled_flux_err']
        else:
            flux = row['flux']
            err = row['flux_err']

        ax.errorbar(x, flux, yerr=err, marker=marker, c=sky)
    return


def make_plot(data, mode, title='', xlabel='', ylabel='', freq_vals=np.linspace(0, 300, 300), cbar=True, scaled=False, models=None, params=None, model_names=None, plot_models=False, modelcolors=None, vline=None, vlinecolors=None, ylim=None):
    fig = plt.figure(figsize=(8,6),dpi=200)
    ax = fig.add_subplot(111)
    ax.grid(False)

    if cbar:
        # get the scalar map, plot the data using the plot_data function
        sm = cmap_setup(mode)
        plot_data(ax, sm, data, mode, scaled=scaled)

        # set up colorbar
        if mode == 'lc':
            fig.colorbar(sm, fraction=0.046, label=r'$\nu$ [GHz]')
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
        x = data['time']
    elif mode == 'sed':
        x = data['freq']

    if models!=None:
        for i, (model,param,model_name,color) in enumerate(zip(models,params,model_names,modelcolors)):
            plot_model(model, param, x, ax, model_name, color, freq_vals=freq_vals, ylim=ylim)

    if vline != None:
        for i, (line, linecolor) in enumerate(zip(vline, vlinecolors)):
            ax.axvline(x=line, color=linecolor, linestyle='--')


def cmap_setup(mode, cmap=cmap, min_freq=0, max_freq=300, min_time=1360, max_time=1370):
    '''
    color markers by frequency/time
    '''
    if mode == 'lc':
        freq_cmap = plt.cm.get_cmap(cmap)
        
        cNorm  = colors.Normalize(vmin=min_freq, vmax=max_freq)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        sm = scalarMap
        sm._A = []
    elif mode == 'sed':
        time_cmap = plt.cm.get_cmap(cmap)
        
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


def F_SSA(freq, K1, K2, p):
    """
    Calculate the flux density using Synchotron Self Absorption (SSA) model.

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    p (float): Power-law index.

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    #scale the freqs by 10
    tau = K2 * (freq/10)**(-(p + 4) / 2)
    F = K1 * (freq/10)**(5/2) * (1 - np.exp(-tau))
    return F


def F_SSA_time(freq, time, K1, K2, p, a, b):
    # scale the freqs by 10
    tau = K2 * (freq/10)**(-(p + 4) / 2) * time**(-(a+b))
    F = K1 * (freq/10)**(5/2) * (time**a) * (1 - np.exp(-tau))
    return F    


def F_FFA(freq, K1, K2, alpha):
    """
    Calculate the flux density using Synchotron Self Absorption (FFA) model.

    Parameters:
    freq (float or numpy.ndarray): Frequency values.
    K1 (float): Scaling factor.
    K2 (float): Scaling factor for the optical depth.
    p (float): Power-law index.

    Returns:
    F (numpy.ndarray): Flux density values.
    """
    #scale the freqs by 10
    tau = K2 * (freq/10)**(-2.1)
    F = K1 * (freq/10)**(-alpha) * (np.exp(-tau))
    return F


def F_FFA_time(freq, time, K1, K2, alpha, beta, delta):
    #scale the freqs by 10
    tau = K2 * (freq/10)**(-2.1) * time**(-delta)
    F = K1 * (freq/10)**(-alpha) * (time**(-beta)) * (np.exp(-tau))
    return F


def calc_params(data, model, initial_guess, bounds):
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