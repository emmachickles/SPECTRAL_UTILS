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

def get_fnames(data_dir, target='ATLASJ1138'):

    fnames = os.listdir(data_dir)
    fnames = [f for f in fnames if 'spec1d' in f and \
              target in f and '.fits' in f]
    fnames = sorted(fnames)

    return fnames


def load_reduced_spectrum(filename, fluxcal=True):
    """
    Load reduced flux-calibrated spectrum from a FITS file.
    * output_shape = '2d' for loading the spectra in (# orders, # pixels) shape
    """
    
    with fits.open(filename) as hdul:
        header = hdul[0].header
        wave_arr, flux_arr, ferr_arr, order_arr = [], [], [], []
        for i, hdu in enumerate(hdul[1:-1]):
                        
            data = hdu.data
            wave = data['OPT_WAVE']
            if fluxcal:            
                flux = data['OPT_FLAM']
                ferr = data['OPT_FLAM_SIG']
            else:
                flux = data['OPT_COUNTS']
                ferr = data['OPT_COUNTS_SIG']
            order = hdu.header['ECH_ORDER']
            
            wave_arr.append(wave)
            flux_arr.append(flux)
            ferr_arr.append(ferr)
            order_arr.append(order)

    wave_arr = np.array(wave_arr)
    flux_arr = np.array(flux_arr)
    ferr_arr = np.array(ferr_arr)
    order_arr = np.array(order_arr)
            
    return wave_arr, flux_arr, ferr_arr, order_arr, header

def load_processed_spectra(filename):
    return

def normalize_spectrum(modwave, modflux, obswave, obsflux, obsferr):
    """
    Normalize a model spectrum by matching the median and standard deviation of
    the data.
    """
    
    # Interpolate both the data and the model spectra to the same evenly sampled
    # wavelength grid
    intwave = np.linspace(np.min(obswave), np.max(obswave), int(len(obswave)*0.05))
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

# def normalize_spectrum(target_spectrum, reference_spectrum, flag=False):

#     import pandas as pd

#     # Calculate rolling standard deviation of reference spectrum
#     window = int(len(reference_spectrum)*0.01)    
#     series = pd.Series(reference_spectrum)
#     ref_std = np.nanmedian(series.rolling(window=window).std())

#     # Calculate rolling standard deviation of target spectrum
#     window = int(len(target_spectrum)*0.01)
#     print(window)
#     series = pd.Series(target_spectrum)
#     targ_std = np.nanmedian(series.rolling(window=window).std())

#     # Standardize target spectrum to standard deviation of 1 and median of 0
#     target_spectrum = (target_spectrum - np.median(target_spectrum)) / targ_std

#     # Match target spectrum standard deviation to reference spectrum
#     target_spectrum *= ref_std

#     # Match target spectrum median to reference spectrum
#     target_spectrum += np.median(reference_spectrum)

#     return target_spectrum

# def normalize_spectrum(target_spectrum, reference_spectrum, reference_errors):

#     from scipy.optimize import minimize
    
#     def objective_function(params, target_spectrum, reference_spectrum,
#                            reference_errors):
#         scale, offset = params
#         normalized_spectrum = scale * target_spectrum + offset
#         residuals = (normalized_spectrum - reference_spectrum) / reference_errors
#         return np.sum( residuals**2 )

#     # Initialize guesses for scale and offset
#     initial_params = [1.0, 0.0]

#     # Perform minimization
#     result = minimize(objective_function, initial_params,
#                       args=(target_spectrum, reference_spectrum, reference_errors))

#     # Apply the optimized parameters
#     scale, offset = result.x
#     normalized_spectrum = scale * target_spectrum + offset
    
#     return normalized_spectrum

def sigma_clipping(fname, nstd=5):
    """
    Perform sigma clipping
    """
    
    from scipy.stats import binned_statistic
    from astropy.io import fits
    from astropy.table import Table, Column
    hdul = fits.open(fname)

    for hdu in hdul[1:-1]:

        # Grab data points in order
        table_data = Table(hdu.data)        
        wave = hdu.data['OPT_WAVE'].copy()
        flux = hdu.data['OPT_FLAM'].copy()
        ferr = hdu.data['OPT_FLAM_SIG'].copy()
        
        # Initialize mask column
        mask = np.zeros(1024, dtype=bool)

        # Remove dead pixels
        mask[ np.nonzero(ferr == 0.) ] = True
        
        # Fit continuum
        nbins = np.ceil(0.1*np.count_nonzero(~mask))
        res = binned_statistic(wave[~mask], flux[~mask], bins=nbins)
        binned_flux = res.statistic
        binned_wave = res.bin_edges[:-1] + np.diff(res.bin_edges)/2
        continuum = np.interp(wave[~mask], binned_wave, binned_flux)        
        
        # Subtract continuum before removing outliers
        flux[~mask] = flux[~mask] - continuum
        
        # Compute weighted standard deviation
        weights = 1/ferr[~mask]**2
        avg = np.average(flux[~mask], weights=weights)
        std = np.average( np.abs(flux[~mask] - avg), weights=weights )

        # Compute upper and lower bounds 
        upbnd = avg + nstd*std
        lobnd = avg - nstd*std

        # Identify outlier pixels that should be masked
        bad_inds = np.concatenate([np.nonzero(flux[~mask]<lobnd)[0],
                                   np.nonzero(flux[~mask]>upbnd)[0]])
        mask[np.nonzero(~mask)[0][bad_inds]] = True

        flux = np.zeros(flux.shape)

        # Add extension to FITS file
        mask_col = Column(name='MASK_SIG', data=mask, dtype=bool)
        table_data.add_column(mask_col)
        new_hdu = fits.BinTableHDU(data=table_data, header=hdul[1].header)
        
        # Visualize
        flux = hdu.data['OPT_FLAM']
        fig = plt.figure(figsize=(10,10))
        order = hdu.header['ECH_ORDER']
        plt.suptitle('ECH_ORDER = '+str(order))        
        gs = fig.add_gridspec(2, 2, width_ratios=(3,1))
        ax_hist1 = fig.add_subplot(gs[0,1])
        ax_hist2 = fig.add_subplot(gs[1,1])        
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax1.errorbar(wave, flux, ferr, ls='', c='k')
        ax1.errorbar(wave[mask], flux[mask], ferr[mask], ls='', c='r')
        ax1.plot(binned_wave, binned_flux+upbnd, ls='--', c='purple')
        ax1.plot(binned_wave, binned_flux+lobnd, ls='--', c='purple')  
        ax1.set_ylabel('Flux [erg/s/cm^2/Ang]')
        ax1.set_xlabel('Wavelength [Ang]')
        ax2.errorbar(wave[~mask], flux[~mask], ferr[~mask], ls='', c='k')
        ax2.set_ylabel('Sigma-clipped flux [erg/s/cm^2/Ang]')
        ax2.set_xlabel('Wavelength [Ang]')          
        ax_hist1.hist(flux, bins=100, color='darkgray', log=True)
        ax_hist1.axvline(lobnd, ls='--', c='purple')
        ax_hist1.axvline(upbnd, ls='--', c='purple')
        ax_hist1.set_xlabel('Flux [erg/s/cm^2/Ang]')
        ax_hist1.text(lobnd, ax_hist1.get_ylim()[1], r'${:d}\times\sigma$'.format(nstd),
                     ha='right', va='top')
        ax_hist1.text(upbnd, ax_hist1.get_ylim()[1], r'${:d}\times\sigma$'.format(nstd),
                     ha='left', va='top')
        ax_hist2.hist(flux[~mask], bins=100, color='darkgray', log=True)
        ax_hist2.set_xlabel('Sigma-clipped flux [erg/s/cm^2/Ang]')        
        plt.tight_layout()
        outf = 'Plots/'+fname.split('_')[-3]+'_ECH_ORDER_'+str(int(order))+'.png'
        plt.savefig(outf)
        print('Saved '+outf)
        plt.close()

    # Add a HISTORY comment
    header = hdul[0].header
    header.add_history('Added a new column MASK_SIG')

    # Save the modified FITS file
    hdul.writeto(fname, overwrite=True)
    print('Added column "MASK_SIG" to '+fname)
    hdul.close()

def outlier_pixels(wave, flux, ferr, order, header, standard_true, out_dir,
                   niqr_lo=10, niqr_up=10):
    """
    Identify pixels of MagE spectrum with significant deviation from ESO
    standard, due to detector hot or dead pixels, cosmic rays, or other
    instrumental artifacts.
    """

    # import pypeit
    import pandas as pd
    from scipy.interpolate import CubicSpline        

    # Load MagE standard
    target = header['TARGET']
    
    wave_true, flux_true, ferr_true = np.loadtxt(standard_true).T # wave flux ferr

    # >> Fit spline to ESO spectrum
    spline = CubicSpline(wave_true, flux_true, extrapolate=False)
    
    # for i, ordid in enumerate(order):
    for i, ordid in enumerate([15]):        

        # Grab data points in order
        iord = np.nonzero(order == ordid)[0][0]
        wave_ord = wave[iord]
        flux_ord = flux[iord]
        ferr_ord = ferr[iord]

        # Remove flagged points (NaNs)
        num_inds = np.nonzero(~np.isnan(flux_ord))[0]
        wave_ord = wave_ord[num_inds]
        flux_ord = flux_ord[num_inds]
        ferr_ord = ferr_ord[num_inds]

        # Idenitfy spike outliers
        bad_inds = []
        mask = np.ones(len(wave_ord), dtype=bool)
        thresh_inds = np.empty(1)
        count = 0
        while len(thresh_inds) > 0:
            # Compute differences
            diff = np.diff(flux_ord[mask])

            # Compute threshold
            med = np.median(diff)
            q3, q1 = np.percentile(diff, [75, 25])
            iqr = (q3 - q1)/2
            threshold = 10*iqr

            # Identify outlier point
            thresh_inds = np.nonzero( np.abs(diff) > threshold )[0]
            for i in range(len(thresh_inds)):
                val1 = flux_ord[mask][ thresh_inds[i] ]
                val2 = flux_ord[mask][ thresh_inds[i]+1 ]
                if np.abs( val1-med ) < np.abs( val2-med ):
                    ind = np.arange(len(wave_ord))[mask][thresh_inds[i]+1]
                    bad_inds.append( ind )
                else:
                    ind = np.arange(len(wave_ord))[mask][thresh_inds[i]]
                    bad_inds.append( ind )
            
            mask[bad_inds] = False        
            count += 1

        bad_inds = np.array(bad_inds)
        good_inds = np.setdiff1d( np.arange(len(wave_ord)), bad_inds )
        diff = np.diff(flux_ord)
                    
        # Visualize
        fig = plt.figure(figsize=(10,10))
        plt.suptitle('ECH_ORDER = '+str(ordid))
        gs = fig.add_gridspec(2, 2, width_ratios=(3,1))
        ax_hist1 = fig.add_subplot(gs[0,1])
        ax_hist2 = fig.add_subplot(gs[1,1])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax1.plot(wave_ord[good_inds], diff[good_inds-1], '.k')
        ax1.plot(wave_ord[bad_inds], diff[bad_inds-1], '.r')        
        ax1.set_ylabel('Interval F_lambda [erg/s/cm^2/Ang]')
        ax1.set_xlabel('Wavelength [Ang]')
        ax2.errorbar(wave_ord[good_inds], flux_ord[good_inds], ferr_ord[good_inds],
                     ls='', c='k')
        ax2.errorbar(wave_ord[bad_inds], flux_ord[bad_inds], ferr_ord[bad_inds],
                     ls='', c='r')        
        ax2.set_ylabel('Outlier-clipped F_lambda [erg/s/cm^2/Ang]')
        ax2.set_xlabel('Wavelength [Ang]')
        up, lo = np.percentile(flux_ord, [99, 1])
        ax2.set_ylim([lo, up])
        ax_hist1.hist(diff, bins=100, color='darkgray', log=True)
        ax_hist1.set_xlabel('Interval F_lambda [erg/s/cm^2/Ang]')
        ax_hist2.hist(flux_ord[good_inds], bins=100, color='darkgray', log=True)
        ax_hist2.set_xlabel('Outlier-clipped F_lambda [erg/s/cm^2/Ang]')
        plt.tight_layout()
        outf =  out_dir+target+'_spike_order_'+str(ordid)+'.png'
        plt.savefig(outf)
        print('Saved '+outf)
        plt.close()

        import pdb
        pdb.set_trace()

        # Mask spike outliers
        wave[iord, num_inds[bad_inds]] = np.nan
        flux[iord, num_inds[bad_inds]] = np.nan
        ferr[iord, num_inds[bad_inds]] = np.nan
        wave_ord = np.delete(wave_ord, bad_inds)
        flux_ord = np.delete(flux_ord, bad_inds)
        ferr_ord = np.delete(ferr_ord, bad_inds)
        
        # Evaluate spline for current order
        flux_spl = spline(wave_ord)
        
        # Normalize true spectrum
        flux_spl = normalize_spectrum(wave_ord, flux_spl, wave_ord, flux_ord, ferr_ord)

        # Compute quartiles
        q3, q1 = np.percentile(flux_ord, [75, 25])
        iqr = (q3 - q1)/2
        
        # Computer upper and lower bounds
        upbnd = flux_spl + niqr_up*iqr
        lobnd = flux_spl - niqr_lo*iqr
        good_inds = np.nonzero((flux_ord < upbnd)*(flux_ord>lobnd))[0]
        bad_inds = np.nonzero((flux_ord<lobnd)+(flux_ord>upbnd))[0]
        
        # Visualize
        fig = plt.figure(figsize=(10,10))
        plt.suptitle('ECH_ORDER = '+str(ordid))
        gs = fig.add_gridspec(2, 2, width_ratios=(3,1))
        ax_hist1 = fig.add_subplot(gs[0,1])
        ax_hist2 = fig.add_subplot(gs[1,1])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        q99, q1 = np.percentile(flux_ord, [99, 1])
        up = np.max([q99, np.max(upbnd)])
        lo = np.min([q1, np.min(lobnd)])
        ax1.errorbar(wave_ord, flux_ord, ferr_ord, ls='', c='k',
                       label='MagE Spectrum')
        ax1.errorbar(wave_ord[bad_inds], flux_ord[bad_inds], ferr_ord[bad_inds],
                       ls='', marker='.', c='r', ms=0.5)
        ax1.plot(wave_ord, flux_spl, '-b', lw=1, alpha=0.8, label='HST Spectrum')
        ax1.plot(wave_ord, upbnd, '--r', lw=1, alpha=0.8)
        ax1.plot(wave_ord, lobnd, '--r', lw=1, alpha=0.8)
        ax1.set_ylabel('F_lambda [erg/s/cm^2/Ang]')
        ax1.set_xlabel('Wavelength [Ang]')
        ax1.set_ylim([lo, up])
        ax1.legend()
        ax2.errorbar(wave_ord[good_inds], flux_ord[good_inds], ferr_ord[good_inds],
                     ls='', c='k', label='MagE Spectrum')
        ax2.plot(wave_ord[good_inds], flux_spl[good_inds], '-b', lw=1, alpha=0.8,
                 label='HST Spectrum')
        ax2.set_ylabel('Outlier-clipped F_lambda [erg/s/cm^2/Ang]')
        ax2.set_xlabel('Wavelength [Ang]')
        ax2.legend()
        ax_hist1.hist(flux_ord, bins=100, color='darkgray', log=True)
        ax_hist1.axvline(q1, ls='--', c='k')
        ax_hist1.axvline(q3, ls='--', c='k')
        ax_hist1.axvline(np.median(upbnd), ls='--', c='r')
        ax_hist1.axvline(np.median(lobnd), ls='--', c='r')
        ax_hist1.set_xlabel('F_lambda [erg/s/cm^2/Ang]')
        ax_hist1.text(q1, ax_hist1.get_ylim()[1], ' 25th percentile ',
                      ha='right', va='top')
        ax_hist1.text(q3, ax_hist1.get_ylim()[1], ' 75th percentile ',
                      ha='left', va='top')
        ax_hist1.set_xlim([lo, up])
        ax_hist2.hist(flux_ord[good_inds], bins=100, color='darkgray', log=True)
        ax_hist2.set_xlabel('Outlier-clipped F_lambda [erg/s/cm^2/Ang]')
        plt.tight_layout()
        outf =  out_dir+target+'_outlier_order_'+str(ordid)+'.png'
        plt.savefig(outf)
        print('Saved '+outf)
        plt.close()

        # Mask outliers
        wave[iord, num_inds[bad_inds]] = np.nan
        flux[iord, num_inds[bad_inds]] = np.nan
        ferr[iord, num_inds[bad_inds]] = np.nan
        
    return wave, flux, ferr, order

def trim_order_edges(wave, flux, ferr, order, header, out_dir):
    '''
    Deal with overlapping orders and trim the blue end of the bluest order and
    the red end of the reddest order. (Higher order = bluer)
    '''

    inds = np.argsort(order)
    wave, flux, ferr, order = wave[inds], flux[inds], ferr[inds], order[inds]
    
    for i in range(len(order)-1):

        # Grab data points in first order
        inds1 = np.nonzero(order == order[i])[0][0]
        wave1 = wave[inds1]
        flux1 = flux[inds1]
        ferr1 = ferr[inds1]

        # Grab data points in second order
        inds2 = np.nonzero(order == order[i+1])[0][0]
        wave2 = wave[inds2]
        flux2 = flux[inds2]
        ferr2 = ferr[inds2]

        # Find region of overlap
        inds1 = np.nonzero( wave1 < np.nanmax(wave2) )
        inds2 = np.nonzero( wave2 > np.nanmin(wave1) )

        # Find wavelength to cut off order
        wlin = np.linspace(np.min(wave1[inds1]), np.max(wave2[inds2]), 100)
        f1 = np.interp(wlin, wave1[inds1], ferr1[inds1])
        f2 = np.interp(wlin, wave2[inds2], ferr2[inds2])
        diff = np.abs( f1 - f2 )
        wcut = wlin[np.argmin(diff)]        

        # Visualize
        fig, ax = plt.subplots(figsize=(8,8), nrows=3)
        ax[0].errorbar(wave1, flux1, ferr1, alpha=0.5, c='navy', ls='',
                       label='Order '+str(order[i]))
        ax[0].errorbar(wave2, flux2, ferr2, alpha=0.5, c='purple', ls='',
                       label='Order '+str(order[i+1]))
        ax[0].legend()
        yrng = np.append(flux1, flux2)
        yrng = yrng[~np.isnan(yrng)]
        up, lo = np.percentile(yrng, [99, 1])        
        ax[0].set_ylim([lo, up])
        ax[0].set_xlabel('Wavelength')
        ax[0].set_ylabel('F_lambda')
        ax[1].errorbar(wave1[inds1], flux1[inds1], ferr1[inds1], alpha=0.5, c='navy', ls='',
                       label='Order '+str(order[i]))
        ax[1].errorbar(wave2[inds2], flux2[inds2], ferr2[inds2], alpha=0.5, c='purple', ls='',
                       label='Order '+str(order[i+1]))
        ax[1].legend()
        yrng = np.append(flux1[inds1], flux2[inds2])
        yrng = yrng[~np.isnan(yrng)]
        up, lo = np.percentile(yrng, [99, 1])
        ax[1].set_ylim([lo, up])
        ax[1].set_xlabel('Wavelength')
        ax[1].set_ylabel('F_lambda')
        ax[2].plot(wave1[inds1], ferr1[inds1], '-', c='navy',
                   label='Order '+str(order[i]))
        ax[2].plot(wave2[inds2], ferr2[inds2], '-', c='purple',
                   label='Order '+str(order[i+1]))
        ax[2].axvline(wcut, ls='--', c='k')
        ax[2].set_xlabel('Wavelength')
        ax[2].set_ylabel('dF_lambda')
        ax[2].legend()
        outf = out_dir+'{}_MJD_{}_trim_orders_{}_{}.png'.format(header['TARGET'], header['MJD'],
                                                                order[i], order[i+1])
        plt.tight_layout()
        plt.savefig(outf, dpi=300)
        print('Saved '+outf)        

        # Cut off order at cut off wavelengths
        bad_inds = np.nonzero( wave1 < wcut  )
        wave1[bad_inds], flux1[bad_inds], ferr1[bad_inds] = np.nan, np.nan, np.nan
        bad_inds = np.nonzero( wave2 > wcut  )
        wave2[bad_inds], flux2[bad_inds], ferr2[bad_inds] = np.nan, np.nan, np.nan

    # Trim reddest edge
    ind_r = np.nonzero(order == np.min(order))[0][0]
    wave_r = wave[ind_r]
    flux_r = flux[ind_r]
    ferr_r = ferr[ind_r]
    med_r = np.nanmedian(ferr_r)
    wlin = np.linspace(np.nanmin(wave_r), np.nanmax(wave_r), 100)
    flin = np.interp(wlin, wave_r, ferr_r)
    inds = np.nonzero( wlin > np.nanmedian(wave_r) )
    diff = np.abs( flin[inds] - med_r)    
    wcut_r = wlin[inds][np.argmin(diff)]

    # Trim bluest edge
    ind_b = np.nonzero(order == np.max(order))[0][0]    
    wave_b = wave[ind_b]
    flux_b = flux[ind_b]
    ferr_b = ferr[ind_b]
    med_b = np.nanmedian(ferr_b)
    wlin = np.linspace(np.nanmin(wave_b), np.nanmax(wave_b), 100)
    flin = np.interp(wlin, wave_b, ferr_b)
    diff = np.abs(flin - med_b)
    wcut_b = wlin[np.argmin(diff)]

    # Visualize
    fig, ax = plt.subplots(2, figsize=(8,6))
    ax[0].errorbar(wave_r, flux_r, ferr_r, ls='', c='r')
    ax[0].set_xlabel('Wavelength')
    ax[0].set_ylabel('F_lambda')
    yrng = np.copy(flux_r)
    yrng = yrng[~np.isnan(yrng)]
    up, lo = np.percentile(yrng, [99, 1])
    ax[0].set_ylim([lo, up])
    ax[1].plot(wave_r, ferr_r, '-r')
    ax[1].set_xlabel('Wavelength')
    ax[1].set_ylabel('dF_lambda')
    yrng = np.copy(ferr_r)
    yrng = yrng[~np.isnan(yrng)]
    up, lo = np.percentile(yrng, [99, 1])
    ax[1].set_ylim([lo, up])
    ax[1].axhline(med_r, ls='--', c='k')
    ax[1].axvline(wcut_r, ls='--', c='k')
    plt.tight_layout()
    outf = out_dir+'{}_MJD_{}_trim_order_{}.png'.format(header['TARGET'],
                                                        header['MJD'], np.min(order))
    plt.savefig(outf, dpi=300)
    print('Saved '+outf)

    fig, ax = plt.subplots(2, figsize=(8,6))
    ax[0].errorbar(wave_b, flux_b, ferr_b, ls='', c='b')
    ax[0].set_xlabel('Wavelength')
    ax[0].set_ylabel('F_lambda')
    yrng = np.copy(flux_b)
    yrng = yrng[~np.isnan(yrng)]
    up, lo = np.percentile(yrng, [99, 1])
    ax[0].set_ylim([lo, up])    
    ax[1].plot(wave_b, ferr_b, '-b')
    ax[1].set_xlabel('Wavelength')
    ax[1].set_ylabel('dF_lambda')
    ax[1].axhline(med_b, ls='--', c='k')
    ax[1].axvline(wcut_b, ls='--', c='k')    
    yrng = np.copy(ferr_b)
    yrng = yrng[~np.isnan(yrng)]
    up, lo = np.percentile(yrng, [99, 1])
    ax[1].set_ylim([lo, up])    
    plt.tight_layout()
    outf = out_dir+'{}_MJD_{}_trim_order_{}.png'.format(header['TARGET'],
                                                        header['MJD'], np.max(order))
    plt.savefig(outf, dpi=300)
    print('Saved '+outf)

    # Cut off order at cut off wavelengths
    bad_inds = np.nonzero( wave_r > wcut_r )
    wave_r[bad_inds], flux_r[bad_inds], ferr_r[bad_inds] = np.nan, np.nan, np.nan
    bad_inds = np.nonzero( wave_b < wcut_b )
    wave_b[bad_inds], flux_b[bad_inds], ferr_b[bad_inds] = np.nan, np.nan, np.nan    

    return wave, flux, ferr, order
        
            
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
