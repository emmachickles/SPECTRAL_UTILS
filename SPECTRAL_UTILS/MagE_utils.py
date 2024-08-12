from astropy.io import fits
import numpy as np
import json
import os

# Define the path to paths.json
paths_file = os.path.join(os.path.dirname(__file__), 'paths.json')

# Load the paths.json file
with open(paths_file, 'r') as f:
    paths = json.load(f)

def load_spectrum(filename):
    """
    Load reduced flux-calibrated spectrum from a FITS file. In addition, removes
    pixels with high chi2 (to model fit of trace) and manually specified 'bad'
    pixels.
    """

    # Get the path to bad pixel file
    flag = os.path.join(os.path.dirname(__file__), paths['flag'])
    flag = np.loadtxt(flag)    
    
    with fits.open(filename) as hdul:
        wave_arr, flux_arr, ferr_arr = [], [], []
        for i, hdu in enumerate(hdul[1:-1]):
                        
            data = hdu.data
            wave = data['OPT_WAVE'] 
            flux = data['OPT_FLAM']
            ferr = data['OPT_FLAM_SIG']
            chi2 = data['OPT_CHI2']

            # Remove funky pixels
            ext = i+1
            inds = np.nonzero( flag[:,2] [flag[:,0] == ext] )
            wave, flux, ferr = wave[inds], flux[inds], ferr[inds]
            chi2 = chi2[inds]

            # Clip based on chi2
            threshold = 5
            inds = np.nonzero(chi2 < threshold)
            wave, flux, ferr = wave[inds], flux[inds], ferr[inds]
            
            wave_arr.extend(wave)
            flux_arr.extend(flux)
            ferr_arr.extend(ferr)
            
    return np.array(wave_arr), np.array(flux_arr), np.array(ferr_arr)
