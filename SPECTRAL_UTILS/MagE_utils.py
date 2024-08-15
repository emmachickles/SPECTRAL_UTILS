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
        wave_arr, flux_arr, ferr_arr, order_arr = [], [], [], []
        for i, hdu in enumerate(hdul[1:-1]):
                        
            data = hdu.data
            wave = data['OPT_WAVE'] 
            flux = data['OPT_FLAM']
            ferr = data['OPT_FLAM_SIG']
            # chi2 = data['OPT_CHI2']
            order = hdu.header['ECH_ORDER']

            if flag is not None: # Remove funky pixels
                data = np.loadtxt(flag)
                inds = np.nonzero( data[:,2] [data[:,0] == order] )
                wave, flux, ferr = wave[inds], flux[inds], ferr[inds]
                # chi2 = chi2[inds]
                

            # # Clip based on chi2
            # threshold = 5
            # inds = np.nonzero(chi2 < threshold)
            # wave, flux, ferr = wave[inds], flux[inds], ferr[inds]
            
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
            
    return wave_arr, flux_arr, ferr_arr, order_arr

def normalize_spectrum(modwave, modflux, obswave, obsflux):
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

def trim_edges(fname):
    """
    Edges of orders suffer from lower SNR. 
    """

    wave, flux, ferr, order_arr = load_spectrum(fname)

    flag = []
    order_flag = []
    wave_flag = []        

    for i, order in enumerate(np.unique(order_arr)):

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
        
        # x = np.linspace(np.min(ferr_ord), np.max(ferr_ord), 1000)
        # ax_2dhist.plot(x, p[0]*x + p[1], '-k')
        # x = np.linspace(np.min(ferr_ord), np.max(ferr_ord), 1000)
        # ax_2dhist.plot(x, p[0]*x + p[1] - p[0]*0.5*np.std(ferr_ord), '--k')
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

        

def outlier_pixels(standard_MagE, STD_NAME='BPM16274'):
    """
    Identify pixels of MagE spectrum with significant deviation from ESO
    standard, due to detector hot or dead pixels, cosmic rays, or other
    instrumental artifacts.
    """

    import pypeit
    import pandas as pd
    from scipy.interpolate import CubicSpline        
    
    esofil_dir = pypeit.__file__[:-11]+"data/standards/esofil/"

    # esofile_info.columns = ['File', 'Name', 'RA_2000', 'DEC_2000']
    esofil_info = pd.read_csv(esofil_dir+'esofil_info.txt', comment='#', sep='\s+')
    standard_file = esofil_dir + \
        esofil_info[esofil_info['Name'] == STD_NAME]['File'].values[0]
    print(esofil_info[esofil_info['Name'] == STD_NAME])
    
    wave_eso, flux_eso, ferr_eso, _ = np.loadtxt(standard_file).T # wave flux ferr
    
    wave, flux, ferr, order_arr = load_spectrum(standard_MagE)

    # >> Normalize ESO spectrum
    flux_eso = normalize_spectrum(wave_eso, flux_eso, wave, flux)
    
    # >> Fit spline to ESO spectrum
    spline = CubicSpline(wave_eso, flux_eso, extrapolate=False)
    
    flag = []
    order_flag = []
    wave_flag = []    

    for i, order in enumerate(np.unique(order_arr)):

        # Grab data points in order
        inds_ord = np.array(order_arr == order)
        wave_ord = wave[inds_ord]
        flux_ord = flux[inds_ord]
        ferr_ord = ferr[inds_ord]

        # Evaluate spline for current order
        flux_spl = spline(wave_ord)

        # Compute quartiles
        q3_ferr, q1_ferr = np.percentile(ferr_ord, [75, 25])
        q3_flux, q1_flux = np.percentile(flux_ord, [75, 25])
        iqr_ferr = (q3_ferr - q1_ferr)/2        
        iqr_flux = (q3_flux - q1_flux)/2

        # Compute upper and lower bounds on FLAM and FLAM_SIG
        upbnd_ferr = q3_ferr
        upbnd_flux = flux_spl + 7*iqr_flux
        lobnd_flux = flux_spl - 12*iqr_flux
        good_inds = np.nonzero( (ferr_ord < upbnd_ferr) *\
                                (flux_ord < upbnd_flux) *\
                                (flux_ord > lobnd_flux))[0]        
        bad_inds = np.setdiff1d(np.arange(1024), good_inds)                

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
        ax.plot(wave_ord, flux_spl, '-b', lw=1, alpha=0.8)
        ax.plot(wave_ord, upbnd_flux, '--r', lw=1, alpha=0.8)
        ax.plot(wave_ord, lobnd_flux, '--r', lw=1, alpha=0.8)        
        
        inds = np.nonzero( (wave_eso > np.min(wave_ord)) * (wave_eso < np.max(wave_ord)) )
        ax.errorbar(wave_eso[inds], flux_eso[inds], ferr_eso[inds], ls='', c='b', alpha=0.8)        
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
        
        # x = np.linspace(np.min(ferr_ord), np.max(ferr_ord), 1000)
        # ax_2dhist.plot(x, p[0]*x + p[1], '-k')
        # x = np.linspace(np.min(ferr_ord), np.max(ferr_ord), 1000)
        # ax_2dhist.plot(x, p[0]*x + p[1] - p[0]*0.5*np.std(ferr_ord), '--k')
        ax_2dhist.plot(ferr_ord, flux_ord, '.', alpha=0.75, ms=2)
        ax_2dhist.plot(ferr_ord[bad_inds], flux_ord[bad_inds], 'xr', alpha=0.75, ms=2)     
        ax_2dhist.set_xlabel('OPT_FLAM_SIG')
        ax_2dhist.set_ylabel('OPT_FLAM')
        ax_2dhist.set_ylim([np.min(flux_ord), np.max(flux_ord)])
        plt.tight_layout()
        plt.savefig('Plots/outlier_ech_order_'+str(int(order))+'.png')        
        plt.close()        
        
        flag_ord = np.zeros(1024)
        flag_ord[bad_inds] = 1.        
        flag.extend(flag_ord)
        order_flag.extend(np.ones(len(flag_ord))*order)
        wave_flag.extend(wave_ord)


    flag = np.array(flag).astype('int')
    np.savetxt('Analysis/outlier_pix.txt', 
               np.array([ order_flag, wave_flag, flag ]).T, 
               header='ECH_ORDER WAVELENGTH FLAG')
    
    

