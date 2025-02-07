import numpy as np

def BJDConvert(times, RA, Dec, date_format='mjd', telescope='Las Campanas Observatory'):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, BarycentricTrueEcliptic
    t = Time(times, format=date_format, scale='utc')
    t2 = t.tdb
    c = SkyCoord(RA, Dec, unit='deg')
    d = c.transform_to(BarycentricTrueEcliptic)
    Observatory = EarthLocation.of_site(telescope)
    delta = t2.light_travel_time(c, kind='barycentric', location=Observatory)
    BJD_TDB = t2 + delta
    return BJD_TDB

def shift_wavelength(wave, velocity):
    """
    Shift the wavelength array to rest frame by a given velocity.
    * wave [Angstroms]
    * velocity [km/s]
    """
    c = 299792.458  # Speed of light in km/s
    return wave * (1 + velocity / c)

def clip_outliers(wave, flux, ferr=None):
    """
    Outliers stand out as non-physical, abrupt jumps, in contrast to smoothly
    varying absorption lines.
    """
    import pandas as pd

    window = int(len(flux) * 0.05)
    rolling_std = pd.Series(flux).rolling(window=window).std()
    threshold = 2 * np.nanmedian(rolling_std)
    inds = np.nonzero( rolling_std < threshold )

    # q3, q1 = np.percentile(flux, [75, 25])
    # iqr = q3 - q1
    # med = np.median(flux)
    # inds = np.nonzero( (flux < med+3*iqr) * (flux > med-10*iqr) )

    wave, flux = wave[inds], flux[inds]
    if ferr is None:
        return wave, flux
    else:
        ferr = ferr[inds]
        return wave, flux, ferr
    
def linear_model(wave, slope, const):
    return slope*wave + const

def line_model(wave, x_0, amp_L, fwhm_L, fwhm_G, slope, const):
    from astropy.modeling.models import Voigt1D    
    model = Voigt1D(x_0, amp_L, fwhm_L, fwhm_G)(wave)
    continuum = linear_model(wave, slope, const)
    return model+continuum

def multi_line_model(wave, wave0, rv, slope, const, amp_L, fhwm_L, fwhm_G):
    '''
    Expects wave to be a list, with len(wave) = len(wave0)
    '''
    from astropy.modeling.models import Voigt1D
    import astropy.constants as c
    c = c.c.to('km/s').value
    flux = []
    for i in range(len(wave0)):
        wshift = rv * wave[i] / c
        x_0 = wave0[i] - wshift
        flux.append( line_model(wave[i], x_0, amp_L, fwhm_L, fwhm_G,
                                slope, const ) )
    return np.array(flux)
    
def normalize_continuum(wave, flux, ferr=None, deg=1, pad=20, wave0=None):

    if wave0 is None:
        wave0 = np.median(wave)
    
    # Extract continuum region without absorption line
    inds = np.nonzero( np.abs(wave - wave0) > pad )
    wave_cont, flux_cont = wave[inds], flux[inds]
    if ferr is not None:
        ferr_cont = ferr[inds]
    else:
        ferr_cont = None

    # Fit continuum
    p = np.polyfit(wave_cont, flux_cont, deg)
    
    # Normalize
    continuum = np.poly1d(p)(wave)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(wave, flux, '-k')
    # plt.plot(wave_cont, flux_cont, '-b')
    # plt.plot(wave, continuum, '-r')
    # plt.savefig('/home/echickle/foo.png')
    # import pdb
    # pdb.set_trace()
    
    flux = flux / continuum
        
    if ferr is None:
        return flux
    else:
        ferr = ferr / continuum
        return flux, ferr

def extract_lines(wave, flux, ferr, wave0, pad=100, sigclip=False, norm=False):

    import pandas as pd 
    
    c = 299792 # km/s

    wave_line, flux_line, ferr_line = [], [], [] 
    for i, w0 in enumerate(wave0):

        # Define region around absorption line
        wmin = w0 - pad
        wmax = w0 + pad
        inds = np.nonzero( (wave > wmin) * (wave < wmax) )

        # Extract region
        wave_clip = wave[inds]
        flux_clip = flux[inds]
        ferr_clip = ferr[inds]

        # Remove outliers
        if sigclip:
            wave_clip, flux_clip, ferr_clip = clip_outliers(wave_clip, flux_clip, ferr_clip)

        if norm:
            # Normalize region
            flux_clip, ferr_clip = normalize_continuum(wave_clip, flux_clip, ferr_clip)
            
        # Append to list
        wave_line.append(wave_clip)
        flux_line.append(flux_clip)
        ferr_line.append(ferr_clip)
        
    return wave_line, flux_line, ferr_line

# def binning(wave_new, wave, flux, ferr):
