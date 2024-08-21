from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
import os

# Define the path to paths.json
paths_file = os.path.join(os.path.dirname(__file__), 'MagE_paths.json')

# Load the paths.json file
with open(paths_file, 'r') as f:
    paths = json.load(f)

# # Get the path to bad pixel file
# flag = os.path.join(os.path.dirname(__file__), paths['flag'])
# flag = np.loadtxt(flag)

def get_spec1d_fnames(data_dir, target='ATLASJ1138'):

    fnames = os.listdir(data_dir)
    fnames = [f for f in fnames if 'spec1d' in f and \
              target in f and '.fits' in f]
    fnames = sorted(fnames)

    return fnames


def load_spectrum(filename, flag=None):
    """
    Load reduced flux-calibrated spectrum from a FITS file.
    """
    
    with fits.open(filename) as hdul:
        header = hdul[0].header
        wave_arr, flux_arr, ferr_arr, order_arr = [], [], [], []
        for i, hdu in enumerate(hdul[1:-1]):
                        
            data = hdu.data
            wave = data['OPT_WAVE'] 
            flux = data['OPT_FLAM']
            ferr = data['OPT_FLAM_SIG']
            order = hdu.header['ECH_ORDER']

            if flag is not None: # Remove funky pixels
                data = np.loadtxt(flag, delimiter=',', skiprows=1)
                inds = np.nonzero( data[data[:,0] == order][0][1:] )
                wave, flux, ferr = wave[inds], flux[inds], ferr[inds]
                
            wave_arr.extend(wave)
            flux_arr.extend(flux)
            ferr_arr.extend(ferr)
            order_arr.extend(np.ones(len(wave))*order)

    wave_arr = np.array(wave_arr)
    flux_arr = np.array(flux_arr)
    ferr_arr = np.array(ferr_arr)
    order_arr = np.array(order_arr)

    inds = np.argsort(wave_arr)
    wave_arr, flux_arr = wave_arr[inds], flux_arr[inds]
    ferr_arr, order_arr = ferr_arr[inds], order_arr[inds]
            
    return wave_arr, flux_arr, ferr_arr, order_arr, header

def normalize_spectrum(modwave, modflux, obswave, obsflux, obsferr):
    """
    Normalize a model spectrum by matching the median and standard deviation of
    the data.
    """

    # Interpolate both the data and the model spectra to the same evenly sampled
    # wavelength grid
    intwave = np.linspace(np.min(obswave), np.max(obswave), 1000)
    intobsflux = np.interp(intwave, obswave, obsflux)
    intmodflux = np.interp(intwave, modwave, modflux)
    
    # Standardize model to have standard deviation of 1 and median of 0
    modflux = (modflux - np.median(intmodflux)) / np.std(intmodflux)
    intmodflux = (intmodflux - np.median(intmodflux)) / np.std(intmodflux)
    
    # Match standard deviation to data
    modflux = modflux * np.std(intobsflux)
    intmodflux = intmodflux * np.std(intobsflux)

    # Match median to data
    modflux = (modflux - np.median(intmodflux) ) + np.median(intobsflux)
    
    return modflux

def sigma_clipping(fname):
    """
    Edges of orders suffer from lower SNR. 
    """

    wave, flux, ferr, order_arr, header = load_spectrum(fname)
    order_names = np.unique(order_arr)
    
    flag = []

    for i, order in enumerate(order_names):

        # Grab data points in order
        inds_ord = np.array(order_arr == order)
        wave_ord = wave[inds_ord]
        flux_ord = flux[inds_ord]
        ferr_ord = ferr[inds_ord]        

        # Compute quartiles
        q3_ferr, q1_ferr = np.percentile(ferr_ord, [75, 25])
        q3_flux, q1_flux = np.percentile(flux_ord, [75, 25])
        iqr_ferr = (q3_ferr - q1_ferr)/2        
        iqr_flux = (q3_flux - q1_flux)/2

        # Compute upper and lower bounds on FLAM and FLAM_SIG
        upbnd_ferr = q3_ferr
        med_flux = np.median(flux_ord)
        upbnd_flux = med_flux + 7*iqr_flux
        lobnd_flux = med_flux - 12*iqr_flux
        good_inds = np.nonzero( (ferr_ord < upbnd_ferr) *\
                                (flux_ord < upbnd_flux) *\
                                (flux_ord > lobnd_flux))[0]        
        bad_inds = np.setdiff1d(np.arange(1024), good_inds)
        flag_ord = np.zeros(1024)
        flag_ord[bad_inds] = 1.
        flag.append(flag_ord)

        # Visualize
        fig = plt.figure(figsize=(10,10))
        plt.suptitle('ECH_ORDER = '+str(int(order)))        
        gs = fig.add_gridspec(3, 2, height_ratios=(1,1,2))
        ax = fig.add_subplot(gs[2,:])
        ax_1dflux = fig.add_subplot(gs[0,0])
        ax_1dferr = fig.add_subplot(gs[1,0])
        ax_2dhist = fig.add_subplot(gs[0:-1,1])
        ax.errorbar(wave_ord, flux_ord, ferr_ord, ls='', marker='.', c='k',
                    ms=0.5)        
        ax.errorbar(wave_ord[bad_inds], flux_ord[bad_inds], ferr_ord[bad_inds],
                    ls='', marker='x', c='r', ms=0.5)
        ax.axhline(med_flux, ls='-', c='b', lw=1, alpha=0.8)
        ax.axhline(upbnd_flux, ls='--', c='r', lw=1, alpha=0.8)
        ax.axhline(lobnd_flux, ls='--', c='r', lw=1, alpha=0.8)        
        ax.set_ylabel('OPT_FLAM')
        ax.set_xlabel('OPT_WAVE')        
        ax_1dferr.hist(ferr_ord, bins=100)
        ax_1dferr.set_xlabel('OPT_FLAM_SIG')        
        ax_1dferr.axvline(q1_ferr, ls='--', c='k')
        ax_1dferr.axvline(q3_ferr, ls='--', c='k')
        ax_1dferr.axvline(upbnd_ferr, ls='--', c='r')
        ax_1dferr.text(q1_ferr, ax_1dferr.get_ylim()[1], ' 25th percentile ', ha='right', va='top')
        ax_1dferr.text(q3_ferr, ax_1dferr.get_ylim()[1], ' 75th percentile ', ha='left', va='top')
        ax_1dflux.hist(flux_ord, bins=100)
        ax_1dflux.axvline(q1_flux, ls='--', c='k')
        ax_1dflux.axvline(q3_flux, ls='--', c='k')
        ax_1dflux.axvline(np.median(upbnd_flux), ls='--', c='r')
        ax_1dflux.axvline(np.median(lobnd_flux), ls='--', c='r')
        ax_1dflux.set_xlabel('OPT_FLAM')
        ax_1dflux.text(q1_flux, ax_1dflux.get_ylim()[1], ' 25th percentile ', ha='right', va='top')
        ax_1dflux.text(q3_flux, ax_1dflux.get_ylim()[1], ' 75th percentile ', ha='left', va='top')        
        ax_2dhist.plot(ferr_ord, flux_ord, '.', alpha=0.75, ms=2)
        ax_2dhist.plot(ferr_ord[bad_inds], flux_ord[bad_inds], 'xr', alpha=0.75, ms=2)     
        ax_2dhist.set_xlabel('OPT_FLAM_SIG')
        ax_2dhist.set_ylabel('OPT_FLAM')
        ax_2dhist.set_ylim([np.min(flux_ord), np.max(flux_ord)])
        plt.tight_layout()
        outf = 'Plots/'+fname.split('_')[-3]+'_ECH_ORDER_'+str(int(order))+'.png'
        plt.savefig(outf)
        print('Saved '+outf)
        plt.close()
        
    return np.array(flag), order_names

def outlier_pixels(standard_MagE, standard_true):
    """
    Identify pixels of MagE spectrum with significant deviation from ESO
    standard, due to detector hot or dead pixels, cosmic rays, or other
    instrumental artifacts.
    """

    # import pypeit
    import pandas as pd
    from scipy.interpolate import CubicSpline        

    # Load MagE standard
    wave, flux, ferr, order_arr, header = load_spectrum(standard_MagE)
    target = header['TARGET']
    order_ids = np.unique(order_arr)

    #  # Remove dead pixels
    # inds = np.nonzero( flux )
    # wave, flux, ferr, order_arr = wave[inds], flux[inds], ferr[inds], order_arr[inds]
    
    # esofil_dir = pypeit.__file__[:-11]+"data/standards/esofil/"

    # # esofile_info.columns = ['File', 'Name', 'RA_2000', 'DEC_2000']
    # esofil_info = pd.read_csv(esofil_dir+'esofil_info.txt', comment='#', sep='\s+')
    # standard_file = esofil_dir + \
    #     esofil_info[esofil_info['Name'] == target]['File'].values[0]
    # print(esofil_info[esofil_info['Name'] == target])
    
    wave_true, flux_true, ferr_true = np.loadtxt(standard_true).T # wave flux ferr

    # >> Fit spline to ESO spectrum
    spline = CubicSpline(wave_true, flux_true, extrapolate=False)
    
    flag = []

    for i, order in enumerate(order_ids):

        # Grab data points in order
        inds_ord = np.array(order_arr == order)
        wave_ord = wave[inds_ord]
        flux_ord = flux[inds_ord]
        ferr_ord = ferr[inds_ord]

        # Evaluate spline for current order
        flux_spl = spline(wave_ord)
        
        # >> Normalize true spectrum
        flux_spl = normalize_spectrum(wave_ord, flux_spl, wave_ord, flux_ord, ferr_ord)         

        # Compute quartiles
        q3, q1 = np.percentile(flux_ord, [75, 25])
        iqr = (q3 - q1)/2

        # Computer upper and lower bounds
        upbnd = flux_spl + 7*iqr
        lobnd = flux_spl - 12*iqr
        good_inds = np.nonzero((flux_ord < upbnd)*(flux_ord>lobnd))[0]
        bad_inds = np.setdiff1d(np.arange(1024), good_inds)
        
        flag_ord = np.zeros(1024)
        flag_ord[bad_inds] = 1.
        flag.append(flag_ord)

        # Visualize
        fig, ax = plt.subplots(nrows=2, figsize=(10,10))
        plt.suptitle('ECH_ORDER = '+str(int(order)))
        ax[0].hist(flux_ord, bins=100)
        ax[0].axvline(q1, ls='--', c='k')
        ax[0].axvline(q3, ls='--', c='k')
        ax[0].axvline(np.median(upbnd), ls='--', c='r')
        ax[0].axvline(np.median(lobnd), ls='--', c='r')
        ax[0].set_xlabel('F_lambda')
        ax[0].text(q1, ax[0].get_ylim()[1], ' 25th percentile ', ha='right', va='top')
        ax[0].text(q3, ax[0].get_ylim()[1], ' 75th percentile ', ha='left', va='top')
        ax[1].errorbar(wave_ord, flux_ord, ferr_ord, ls='', marker='.', c='k', ms=0.5)
        ax[1].errorbar(wave_ord[bad_inds], flux_ord[bad_inds], ferr_ord[bad_inds],
                       ls='', marker='.', c='r', ms=0.5)
        ax[1].plot(wave_ord, flux_spl, '-b', lw=1, alpha=0.8)
        ax[1].plot(wave_ord, upbnd, '--r', lw=1, alpha=0.8)
        ax[1].plot(wave_ord, lobnd, '--r', lw=1, alpha=0.8)
        # inds = np.nonzero( (wave_true > np.min(wave_ord)) * (wave_true < np.max(wave_ord)) )
        # ax[1].errorbar(wave_true[inds], flux_true[inds], ferr_true[inds], ls='', c='b', alpha=0.8)
        ax[1].set_ylabel('F_lambda')
        ax[1].set_xlabel('Wavelength (Angstroms)')
        plt.tight_layout()
        outf = 'Plots/STD_'+target+'_ECH_ORDER_'+str(int(order))+'.png'
        plt.savefig(outf)
        print('Saved '+outf)
        plt.close()

    return np.array(flag), order_ids

def join_flag(flag1, flag2):

    flag = np.zeros(flag1.shape)
    flag[np.nonzero(flag1)] = 1.
    flag[np.nonzero(flag2)] = 1.

    return flag
    

def save_flag(outf, flag, order):
    cols = ['ECH_ORD'] + ['PIX_'+str(i) for i in range(1,1025)]
    header = ','.join(cols)
    
    data = np.insert(flag, 0, order, axis=1)
    np.savetxt(outf, data, header=header, delimiter=',')
    print('Saved '+outf)

