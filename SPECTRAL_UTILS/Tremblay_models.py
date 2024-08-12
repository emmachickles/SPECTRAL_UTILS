import os
import pdb
import numpy as np
from scipy.ndimage import map_coordinates


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

    Assumes:
    * temp : given in K
    * r    : given in solar radii
    * d    : given in parsecs'''
    
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

    
    
    

    
