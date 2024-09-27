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

def broadening(Teff, logg, vsini, obswave, modwave, modgrid, Teff_grid, lgg_grid,
               R=7000, epsilon=0.5):

    from PyAstronomy.pyasl import instrBroadGaussFast, fastRotBroad
    from scipy.ndimage import map_coordinates    
    
    # Interpolate model grid to an evenly spaced wavelength array
    wmin = np.min(obswave)*0.95
    wmax = np.max(obswave)*1.05
    nbin = np.count_nonzero( (modwave > wmin) * (modwave < wmax) ) 
    linwave = np.linspace(wmin, wmax, 10*nbin)
    modflux = interpolate(linwave, modwave, modgrid, Teff_grid, lgg_grid,
                          Teff, logg)

    # Apply instrumental broadening
    modflux = instrBroadGaussFast(linwave, modflux, R, edgeHandling='firstlast')
    
    # Apply rotational broadening
    modflux = fastRotBroad(linwave, modflux, epsilon, vsini)

    # Trim edge effects
    wmin = np.min(obswave)
    wmax = np.max(obswave)
    inds = np.nonzero( (linwave > wmin) * (linwave < wmax) )
    linwave, modflux = linwave[inds], modflux[inds]
    
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

def fit_lines(obswave, obsflux, obsferr, modwave, modgrid, Teff_grid, lgg_grid,
              R=7000, epsilon=0.5):
    from scipy.optimize import minimize
    from SPECTRAL_UTILS.spec_utils import normalize_continuum

    def mse(params, obswave, obsflux, obsferr, modwave, modgrid, Teff_grid,
            lgg_grid, R, epsilon):
        
        Teff, logg, vsini = params
        mse = 0
        
        for i in range(len(obsflux)):
            
            # Interpolate model spectrum and apply braodening effects
            modflux = broadening(Teff, logg, vsini, obswave[i], modwave, modgrid,
                                 Teff_grid, lgg_grid, R, epsilon)

            # Normalize
            modflux = normalize_continuum(obswave[i], modflux)

            # Calculated weighted squared error
            # w = 1 / (obsferr[i]**2)
            # mse += np.sum( w * (modflux - obsflux[i])**2 ) / np.sum( w )
            mse += np.sum( (modflux - obsflux[i])**2 )

        return mse
    
    p0 = [9000, 6, 500]
    bounds = [[5000, 50000], [5, 7], [1, 1500]]

    args = (obswave, obsflux, obsferr, modwave, modgrid, Teff_grid, lgg_grid,
            R, epsilon)

    # p0 = [9000, 6, 1]
    # print(p0)
    # print(mse(p0, obswave, obsflux, obsferr, modwave, modgrid, Teff_grid,
    #         lgg_grid, R, epsilon))
    # p1 = [50000, 5, 1]
    # print(p1)
    # print(mse(p1, obswave, obsflux, obsferr, modwave, modgrid, Teff_grid,
    #         lgg_grid, R, epsilon))    
    # import pdb
    # pdb.set_trace()
    
    res = minimize(mse, p0, bounds=bounds, args=args, method='Nelder-Mead')

    Teff, logg, vsini = res.x
    print('Effective temperature: '+str(Teff)+' K')
    print('Logg: '+str(logg))
    print('vsini: '+str(vsini)+' km/s')
          

    return res
    
def plot_line_fits(Teff, logg, vsini, modwave, modgrid, Teff_grid, lgg_grid,
                   wave_line, flux_line, wave0, out_dir, R=7000, epsilon=0.5,
                   pad=0.3):

    import matplotlib.pyplot as plt
    import numpy as np
    from SPECTRAL_UTILS.spec_utils import normalize_continuum
    
    plt.figure(figsize=(4,4))
    plt.ylabel('Relative Flux')
    plt.xlabel(r'$\Delta\lambda(Å)$')
    ymin, ymax = 0., 1.02+len(wave_line)*pad
    plt.yticks(np.arange(ymin, ymax, pad))
    plt.ylim([0., plt.ylim()[1]])

    for i, w0 in enumerate(wave0):
        plt.plot(wave_line[i]-w0, flux_line[i]+pad*i, '-k', lw=0.5)

        # Interpolate and normalize model spectrum
        linwave = np.linspace(np.min(wave_line[i]), np.max(wave_line[i]), 1000)
        modflux = broadening(Teff, logg, vsini, linwave, modwave, modgrid,
                             Teff_grid, lgg_grid, R, epsilon)
        modflux = normalize_continuum(linwave, modflux)

        plt.plot(linwave-w0, modflux+pad*i, '-r', lw=0.5)        

    plt.text(0, 0.4, 'Teff = {} K'.format(int(np.round(Teff, -1))), ha='center')
    plt.text(0, 0.3, 'logg={}'.format(np.round(logg, 2)), ha='center')
    plt.text(0, 0.2, 'vsini={} km/s'.format(int(np.round(vsini, -1))),
             ha='center')    
    plt.tight_layout()
    
    fname = out_dir+'line_fits.png'
    plt.savefig(fname, dpi=300)
    print('Saved '+fname)
        

    
    

    
