import numpy as np
cc = 2.99792458e10 # [cm s^-1]
from numpy import pi
arcsec = pi / (180. * 3600) # [radians] = 1/206265 radian/arcsec
from numpy.fft import fftfreq, rfftfreq, fftshift, fft2
import casatools
from numba import njit

@njit(fastmath=True)
def grid_wgts(gwgts, uu, vv, du, dv, npix, wgts):
    for i in np.arange(uu.shape[0]):
        gwgts[int(npix/2 + uu[i]/du + 0.5), int(npix/2 + vv[i]/dv + 0.5)] += wgts[i]
        gwgts[int(npix/2 - uu[i]/du + 0.5), int(npix/2 - vv[i]/dv + 0.5)] += wgts[i]
    return gwgts


@njit
def ungrid_wgts(gwgts, uu, vv, du, dv, npix):
    ugwgts = np.zeros(uu.shape[0])
    for i in np.arange(uu.shape[0]):
        ugwgts[i] = gwgts[int(npix/2 + uu[i]/du + 0.5), int(npix/2 + vv[i]/dv + 0.5)]
    return ugwgts



# This assumes that the MS has a weight spectrum
# can add via initweights(vis, weighting='weight', dowtsp=True)
def weight_multichan(msfile, npix, cell_size, robust=0., method='briggs', perchanweight=False, fix_pcwd=False, npixels=0):
    tb = casatools.table()
    ms = casatools.ms()

    # Use CASA table tools to get frequencies
    tb.open(msfile+"/SPECTRAL_WINDOW")
    chan_freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    tb.close()

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    tb.open(msfile, nomodify=False)
    flag   = tb.getcol("FLAG")
    sigma   = tb.getcol("SIGMA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT_SPECTRUM")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    tb.close()

    flag = np.logical_not(np.prod(flag, axis=(0,2)).T)

    # break out the u, v spatial frequencies, convert from m to lambda
    uu = uvw[0,:][:,np.newaxis]*chan_freqs[:,0]/(cc/100)
    vv = uvw[1,:][:,np.newaxis]*chan_freqs[:,0]/(cc/100)

    # toss out the autocorrelation placeholders
    xc = np.where(ant1 != ant2)[0]

    wgts = (weight[0,:,:] + weight[1,:,:]).T

    uu_xc = uu[xc]
    vv_xc = vv[xc]
    wgts_xc = wgts[xc]

    dl = cell_size*arcsec
    dm = cell_size*arcsec

    du = 1./((npix)*dl)
    dv = 1./((npix)*dm)

    new_wgts = np.copy(weight)

    # grid the weights outside of loop if not perchanweight, only need to do this once... 
    if perchanweight == False:
        gwgts_init = np.zeros((npix, npix))
        gwgts_init = grid_wgts(gwgts_init, np.ravel(uu_xc), np.ravel(vv_xc), du, dv, npix, np.ravel(wgts_xc))

    if fix_pcwd == True:
        # TODO CHECK THIS FOR HALF PIXEL OFFSET
        uvdist_grid = np.sqrt(np.add.outer(np.arange(-(npix/2.)*du, (npix/2.)*du, du)**2, np.arange(-(npix/2.)*dv, (npix/2.)*dv, dv)**2))
        frac_bw = (np.max(chan_freqs) - np.min(chan_freqs)) / rfreq
        corr_fac = frac_bw*uvdist_grid/du
        corr_fac[corr_fac<1] = 1.

    for chan in range(chan_freqs.shape[0]):
        print(chan)
        # grid the weights (with complex conjugates)
        if perchanweight == True:
            gwgts_init = np.zeros((npix, npix))
            gwgts_init = grid_wgts(gwgts_init, uu_xc[:,chan], vv_xc[:,chan], du, dv, npix, wgts_xc[:,chan])  

        gwgts_init_sq = gwgts_init**2

        # do the weighting, in each case for method/perchanweight selection
        if method == 'briggs':
            # calculate robust parameters
            # normalize differently if only using single channel; note that we assume the weights are not channelized and are uniform across channel
            if perchanweight == True:
                if fix_pcwd == True:
                    f_sq = ((5*10**(-r))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc[:,chan])*2))
                else:
                    f_sq = ((5*10**(-r))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc[:,chan])))
            else:
                f_sq = ((5*10**(-robust))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc)*2))

            if fix_pcwd==True:
                gr_wgts = 1/(1+gwgts_init/corr_fac*f_sq)
            else:
                gr_wgts = 1/(1+gwgts_init*f_sq)

            # multiply to get robust weights
            indexed_gr_wgts = ungrid_wgts(gr_wgts, uu_xc[:,chan], vv_xc[:,chan], du, dv, npix)
            new_wgts[0,chan,:] = wgts_xc[:,chan]*indexed_gr_wgts/2.
            new_wgts[1,chan,:] = wgts_xc[:,chan]*indexed_gr_wgts/2.

        if method == 'briggsabs':
            # multiply to get robust weights
            S_sq = (gwgts_init[index_arr[chan,:,0], index_arr[chan,:,1]]*r**2).T
            indexed_gr_wgts = (1/(S_sq + 2*wgts_xc[:,chan]))
            new_wgts[0,chan,:] = wgts_xc[:,chan]*indexed_gr_wgts/2.
            new_wgts[1,chan,:] = wgts_xc[:,chan]*indexed_gr_wgts/2.

    tb.open(msfile, nomodify=False)
    tb.putcol("WEIGHT_SPECTRUM", new_wgts)
    tb.close()

    return



weight_multichan("test_r-2_c0.02_fix.ms", npix=1024, cell_size=0.02, robust=-2., method='briggs', perchanweight=True, fix_pcwd=True, npixels=0)
weight_multichan("test_r0_c0.02_fix.ms", npix=1024, cell_size=0.02, robust=0., method='briggs', perchanweight=True, fix_pcwd=True, npixels=0)
weight_multichan("test_r0.5_c0.02_fix.ms", npix=1024, cell_size=0.02, robust=0.5, method='briggs', perchanweight=True, fix_pcwd=True, npixels=0)
weight_multichan("test_r1_c0.02_fix.ms", npix=1024, cell_size=0.02, robust=1., method='briggs', perchanweight=True, fix_pcwd=True, npixels=0)
weight_multichan("test_r2_c0.02_fix.ms", npix=1024, cell_size=0.02, robust=2., method='briggs', perchanweight=True, fix_pcwd=True, npixels=0)
