import os
import pdb
import numpy as np
import pdb
import matplotlib.pyplot as plt

def load_models(data_dir='/data/Models/ELM/', Teff_min=5000):
    '''
    Load Tremblay atmosphere models into a homogenous grid, suitable for
    interpolation.
    
    Returns:
    * modwave : grid of wavelengths (Angstrom)
    * modgrid : array of shape ( len(g_grid), len(Teff_grid), len(modwave) )
                flux density per unit frequency (erg / cm2 Hz s sr)
    * Teff_grid : effective temperatures (K)
    * lgg_grid : log10(surface gravities)
    '''

    fnames = os.listdir(data_dir)
    fnames = sorted(fnames)

    g_grid = np.unique([np.float64(f.split('_')[3]) for f in fnames])    
    Teff_grid = np.unique([np.float64(f.split('_')[2]) for f in fnames])

    Teff_grid = Teff_grid[Teff_grid >= Teff_min]

    # >> Load WD atmosphere models for each logG and Teff
    # >> Will hold len(g_grid) lists, each with len(Teff_grid) arrays of fluxes
    modgrid = [] 
    for gg in g_grid:
        temporaneo=[]
        for tt in Teff_grid:
            tempw,tempf = np.loadtxt(data_dir+'NLTE_H_%.1f_%9.3E_0.txt'%(tt,gg),
                                     unpack=True)
            temporaneo.append(tempf)
        modgrid.append(temporaneo)

    modgrid = np.array(modgrid)
    modwave = tempw

    # >> Remove models with NaNs
    inds = np.nonzero(np.isnan(modgrid))
    modgrid = np.delete(modgrid, np.unique(inds[0]), axis=0)
    g_grid = np.delete(g_grid, np.unique(inds[0]))
    modgrid = np.delete(modgrid, np.unique(inds[1]), axis=1)
    Teff_grid = np.delete(Teff_grid, np.unique(inds[1]))

    # >> Return log of surface gravities
    lgg_grid = np.log10(g_grid)
                                
    return modwave, modgrid, Teff_grid, lgg_grid

def interpolate(wave, modwave, modgrid, Teff_grid, lgg_grid, temp, logg):
    '''Interpolates model atmospheres [modgrid] at given wavelengths [wave]
    * temp : given in K
    '''

    from scipy.ndimage import map_coordinates    
    
    vectemp = 0*wave + temp
    veclogg = 0*wave + logg

    # np.interp takes arguments (x, xp, yp)
    windex = np.interp(wave, modwave, np.arange(len(modwave)))
    tindex = np.interp(np.log10(vectemp), np.log10(Teff_grid),
                       np.arange(len(Teff_grid)))
    gindex = np.interp(np.log10(veclogg), np.log10(lgg_grid),
                       np.arange(len(lgg_grid)))

    flux = map_coordinates(modgrid,np.array([gindex,tindex,windex]))

    return flux

def interpolate_linear(obswave, modwave, modgrid, Teff_grid, lgg_grid, temp, logg):
    '''Interpolate model atmosphere to an evenly spaced wavelength array.'''

    wmin = np.min(obswave) - 10
    wmax = np.max(obswave) + 10
    nbin = np.count_nonzero( (modwave > wmin) * (modwave < wmax) ) 
    linwave = np.linspace(wmin, wmax, 10*nbin)
    modflux = interpolate(linwave, modwave, modgrid, Teff_grid, lgg_grid,
                          temp, logg)

    return linwave, modflux
    
def broadening(obswave, linwave, modwave, modflux, vsini, R=7000, epsilon=0.5):

    from PyAstronomy.pyasl import instrBroadGaussFast, fastRotBroad
    from scipy.ndimage import map_coordinates    
    
    # Apply instrumental broadening
    modflux = instrBroadGaussFast(linwave, modflux, R, edgeHandling='firstlast')
    
    # Apply rotational broadening
    modflux = fastRotBroad(linwave, modflux, epsilon, vsini)

    # Trim edge effects
    inds = np.nonzero( (linwave > np.min(obswave)) * (linwave < np.max(obswave)) )
    linwave = linwave[inds]
    
    # Interpolate model grid to observed wavelength grid
    windex = np.interp(obswave, linwave, np.arange(len(linwave)))
    modflux = map_coordinates(modflux, np.array([windex]))

    return modflux
    
def convert_flux_density(wave, flux):
    '''Convert flux density per unit frequency (F_\nu) to flux density per unit
    wavelength (F_\lambda).
    '''
    import astropy.units as u
    import astropy.constants as c
    
    flux = flux * u.erg / u.cm**2 / u.Hz / u.s / u.sr
    flux = flux * c.c / (wave * u.AA)**2
    flux = flux.to(u.erg / u.cm**2 / u.AA / u.s / u.sr)
    flux = flux.value

    return flux

def convert_to_physical(wave, flux, r, d):

    # model atmospheres in units of 1e-8 ergs / (cm2 Hz s sr)

    r = r * 6.957 * 10**10 # cm
    d = d * 3.086 * 10**18 # cm
    c = 3e10 # cm
    
    # convert B_nu to B_lambda
    flux = flux * c / wave**2 # ergs / (cm3 s sr)

    # assume isotropic emission
    flux = flux * 4 * np.pi * r**2 # ergs / (cm3 s)

    # inverse square law
    flux = flux / d**2
    
    flux = flux * 1e8

    return flux

def plot_line_fits(wave, obsflux, modflux, wave0, out_dir,
                   pad=0.3):

    import matplotlib.pyplot as plt
    import numpy as np
    from SPECTRAL_UTILS.spec_utils import normalize_continuum
    
    plt.figure(figsize=(4,4))
    plt.ylabel('Relative Flux')
    plt.xlabel(r'$\Delta\lambda(Å)$')
    ymin, ymax = 0., 1.02+len(wave)*pad
    plt.yticks(np.arange(ymin, ymax, pad))
    plt.ylim([0., plt.ylim()[1]])

    for i, w0 in enumerate(wave0):
        f = normalize_continuum(wave[i], obsflux[i])
        plt.plot(wave[i]-w0, f+pad*i, '-k', lw=0.5)
        f = normalize_continuum(wave[i], modflux[i])
        plt.plot(wave[i]-w0, f+pad*i, '-r', lw=0.5)        

    # plt.text(0, 0.4, 'Teff = {} K'.format(int(np.round(Teff, -1))), ha='center')
    # plt.text(0, 0.3, 'logg={}'.format(np.round(logg, 2)), ha='center')
    # plt.text(0, 0.2, 'vsini={} km/s'.format(int(np.round(vsini, -1))),
    #          ha='center')    
    plt.tight_layout()
    
    fname = out_dir+'line_fits.png'
    plt.savefig(fname, dpi=300)
    print('Saved '+fname)
        

    
    

    
