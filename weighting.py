import numpy as np
cc = 2.99792458e10 # [cm s^-1]
from numpy import pi
arcsec = pi / (180. * 3600) # [radians] = 1/206265 radian/arcsec
from numpy.fft import fftfreq, rfftfreq, fftshift, fft2, ifftshift, ifft2
import scipy.optimize as optimize
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import numpy.linalg as linalg
import casatools
from numba import njit, prange

# set up mpl params for pretty plots
import matplotlib.pylab as pl
from matplotlib.pyplot import *
import matplotlib.colors as col
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family("sans-serif")
from matplotlib import rc
rcParams['mathtext.default']='regular'
rcParams['axes.linewidth'] = 1.0
pl.style.use('classic')
pl.rcParams['hatch.linewidth'] = 0.5
pl.rc('text', usetex=True)
rcParams['font.size'] = 17
rcParams['font.family'] = 'serif'
rcParams['font.weight']='light'
rcParams['mathtext.bf'] = 'serif:normal'
pl.rcParams['xtick.major.pad']='2'
pl.rcParams['ytick.major.pad']='2'





# Number of pixels to consider (n x n window). Note that unlike the 11x11 pixel window that is currently used,
# the number of pixels in this window does not affect the psf that is fit. The window just needs
# to be wide enough to cover the full beam. More pixels simply increases the computational cost.
# In practice I think 31 pixels is a reasonable compromise? The number should be odd.
npix_window = 31


# ellipse fitting code from: nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
# moded a number of typos from original source
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n =  np.argmax(E)
    a = V[:,n]
    if a[0] < 0:
        a = -a
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a < c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*(np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    down2=(b*b-a*c)*(-np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes


def fit_beam(psf_data_raw, cell_size):
    delta = cell_size
    npix = psf_data_raw.shape[0]         # Assume image is square

    # Check if image cube, or just single psf; this example doesn't handle the full polarization case - implicitly assumes we can drop Stokes
    # If single psf, add an axis so we can use a single loop
    psf_data = np.squeeze(psf_data_raw)
    if len(psf_data.shape) == 2:
        psf_data = np.expand_dims(psf_data, axis=2)

    # Roll the axes to make looping more straightforward
    psf_rolled = np.rollaxis(psf_data,2)

    # Loop through the channels and fit a psf to each one
    for chan in np.arange(psf_rolled.shape[0]):
        psf_currchan = psf_rolled[chan]

        # Window out the central 21 pixels - this seems to be wide enough in general, but could be modified
        psf_windowed = psf_currchan[int(npix/2-(npix_window-1)/2):int(npix/2+(npix_window-1)/2 + 1), int(npix/2-(npix_window-1)/2):int(npix/2+(npix_window-1)/2 + 1)]

        # make pixel coordinates for the interpolation - x, y are the native grid
        x = np.arange(psf_windowed.shape[0])-(npix_window-1)/2
        y = np.arange(psf_windowed.shape[0])-(npix_window-1)/2

        # xp, yp are the interpolated grid. Here I use 1001 points, for a factor of ~50 refinement to the fit.
        # This might be overkill, but seems reasonable computationally in my tests.
        xp, yp = np.meshgrid(np.linspace(-(npix_window-1)/2, (npix_window-1)/2, 401), np.linspace(-(npix_window-1)/2, (npix_window-1)/2, 401))
        points = np.vstack((np.ravel(xp), np.ravel(yp))).T

        # interpolate the windowed psf onto this new fine grid. A spline method seems to work best,
        # as linear interpolation starts to perform poorly at low numbers of pixels per beam.
        interpolated = np.reshape(interpn((x, y), psf_windowed, points, method="splinef2d"), (401,401)).T

        # find all points where the interpolated image is close to half power. Anything within .3% seemed to work well
        # in my tests, but this value could probably be tweaked.
        ellipse_pts = np.argwhere(np.abs(interpolated - 0.5) < 0.003)

        # fit an ellipse to these points
        center, phi, axes = find_ellipse(ellipse_pts[:,0], ellipse_pts[:,1])

        # convert to useful units
        phi = np.degrees(phi) - 90.                     # Astronomers use east of north
        if phi < -90.:
            phi += 180.
        major = axes[0]/400.*(npix_window-1)*delta*2
        minor = axes[1]/400.*(npix_window-1)*delta*2

        return phi, major, minor



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



def weight_multichan(base_ms, npix, cell_size, robust=np.array([0.]), chans=np.array([2]), method='briggs', perchanweight=False, mod_pcwd=False, npixels=0):
    tb = casatools.table()
    ms = casatools.ms()

    # Use CASA table tools to get frequencies
    tb.open(base_ms+"/SPECTRAL_WINDOW")
    chan_freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    tb.close()

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    tb.open(base_ms, nomodify=False)
    flag   = tb.getcol("FLAG")
    sigma   = tb.getcol("SIGMA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    tb.close()

    flag = np.logical_not(np.prod(flag, axis=(0,2)).T)

    # break out the u, v spatial frequencies, convert from m to lambda
    uu = uvw[0,:][:,np.newaxis]*chan_freqs[:,0]/(cc/100)
    vv = uvw[1,:][:,np.newaxis]*chan_freqs[:,0]/(cc/100)

    # toss out the autocorrelation placeholders
    xc = np.where(ant1 != ant2)[0]

    wgts = weight[0,:] + weight[1,:]

    uu_xc = uu[xc][:,flag]
    vv_xc = vv[xc][:,flag]
    wgts_xc = wgts[xc]

    dl = cell_size*arcsec
    dm = cell_size*arcsec

    du = 1./((npix)*dl)
    dv = 1./((npix)*dm)

    # create arrays to dump values
    rms = np.zeros((chans.shape[0], robust.shape[0]))
    beam_params = np.zeros((chans.shape[0],3, robust.shape[0]))

    # grid the weights outside of loop if not perchanweight, only need to do this once... 
    if perchanweight == False:
        gwgts_init = np.zeros((npix, npix))
        gwgts_init = grid_wgts(gwgts_init, np.ravel(uu_xc), np.ravel(vv_xc), du, dv, npix, np.ravel(np.broadcast_to(wgts_xc, (uu_xc.shape[1], uu_xc.shape[0])).T))

    if mod_pcwd == True:
        # TODO CHECK THIS FOR HALF PIXEL OFFSET
        uvdist_grid = np.sqrt(np.add.outer(np.arange(-(npix/2.)*du, (npix/2.)*du, du)**2, np.arange(-(npix/2.)*dv, (npix/2.)*dv, dv)**2))
        frac_bw = (np.max(chan_freqs) - np.min(chan_freqs)) / rfreq
        corr_fac = frac_bw*uvdist_grid/du
        corr_fac[corr_fac<1] = 1.

    for i, chan in enumerate(chans):
        print(chan)
        # grid the weights (with complex conjugates)
        if perchanweight == True:
            gwgts_init = np.zeros((npix, npix))
            gwgts_init = grid_wgts(gwgts_init, uu_xc[:,chan], vv_xc[:,chan], du, dv, npix, wgts_xc)  

        gwgts_init_sq = gwgts_init**2

        for j, r in enumerate(robust):
            # do the weighting, in each case for method/perchanweight selection
            if method == 'briggs':
                # calculate robust parameters
                # normalize differently if only using single channel; note that we assume the weights are not channelized and are uniform across channel
                if perchanweight == True:
                    if mod_pcwd == True:
                        f_sq = ((5*10**(-r))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc)*2))
                    else:
                        f_sq = ((5*10**(-r))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc)))
                else:
                    f_sq = ((5*10**(-r))**2)/(np.sum(gwgts_init_sq)/(np.sum(wgts_xc*uu_xc.shape[1])*2))

                if mod_pcwd==True:
                    gr_wgts = 1/(1+gwgts_init/corr_fac*f_sq)
                else:
                    gr_wgts = 1/(1+gwgts_init*f_sq)

                # multiply to get robust weights
                indexed_gr_wgts = ungrid_wgts(gr_wgts, uu_xc[:,chan], vv_xc[:,chan], du, dv, npix)
                wgts_robust = wgts_xc*indexed_gr_wgts
                wgts_robust_sq = wgts_xc*(indexed_gr_wgts)**2

            if method == 'briggsabs':
                # multiply to get robust weights
                S_sq = (gwgts_init[index_arr[chan,:,0], index_arr[chan,:,1]]*r**2).T
                indexed_gr_wgts = (1/(S_sq + 2*wgts_xc))
                wgts_robust = wgts_xc*indexed_gr_wgts
                wgts_robust_sq = wgts_xc*(indexed_gr_wgts)**2


            #get the total gridded weights (to make dirty beam)
            gwgts_final = np.zeros((npix, npix))
            gwgts_final = grid_wgts(gwgts_final, uu_xc[:,chan], vv_xc[:,chan], du, dv, npix, wgts_robust)           

            # create the dirty beam and calculate the beam parameters
            robust_beam = np.real(fftshift(fft2(fftshift(gwgts_final))))
            robust_beam /= np.max(robust_beam)
            beam_params[i,:,j] = fit_beam(robust_beam, cell_size)

            # calculate rms (formula from Briggs et al. 1995)
            C = 1/(2*np.sum(wgts_robust))
            rms[i,j] = 2*C*np.sqrt(np.sum(wgts_robust_sq))
            print(r, beam_params[i,:,j], rms[i,j]*1000.)
        
    return rms, beam_params





#rep_freq = 230.
#cell_size = 0.092/5.
#pb = 6300./rep_freq
#npix = int(pb/cell_size)

npix = 1024
cell_size = 0.02

#rms1, beam_params1 = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.linspace(-2,2,81), method='briggs', perchanweight=True, npixels=0)
#rms1f, beam_params1f = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.linspace(-2,2,81), method='briggs', perchanweight=True, npixels=0, mod_pcwd=True)
#rms2, beam_params2 = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.linspace(-2,2,81), method='briggs', perchanweight=False, npixels=0, mod_pcwd=False)

rms1, beam_params1 = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.array([-2., 0., 0.5, 1., 2.]), method='briggs', perchanweight=True, npixels=0, chans=np.arange(118))
rms1f, beam_params1f = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.array([-2., 0., 0.5, 1., 2.]), method='briggs', perchanweight=True, npixels=0, mod_pcwd=True, chans=np.arange(118))
rms2, beam_params2 = weight_multichan("/lustre/cv/users/rloomis/research_tickets/mod_pcwd/J1610_spw21.ms", npix, cell_size=cell_size, robust=np.array([-2., 0., 0.5, 1., 2.]), method='briggs', perchanweight=False, npixels=0, mod_pcwd=False, chans=np.arange(118))


'''
rms1 = rms1[0]
rms1f = rms1f[0]
rms2 = rms2[0]

robust1=np.linspace(-2.,2.,81)
robust1f=np.linspace(-2.,2.,81)
robust2=np.linspace(-2.,2.,81)


beam_maj1 = np.amax(np.abs(beam_params1[0,1:,:]), axis=0)
beam_maj1f = np.amax(np.abs(beam_params1f[0,1:,:]), axis=0)
beam_maj2 = np.amax(np.abs(beam_params2[0,1:,:]), axis=0)

beam_min1 = np.amin(np.abs(beam_params1[0,1:,:]), axis=0)
beam_min1f = np.amin(np.abs(beam_params1f[0,1:,:]), axis=0)
beam_min2 = np.amin(np.abs(beam_params2[0,1:,:]), axis=0)

ratio1 = beam_maj1/beam_min1
ratio1f = beam_maj1f/beam_min1f
ratio2 = beam_maj2/beam_min2


#convert to mJy/bm
rms1 *= 1000.
rms1f *= 1000.
rms2 *= 1000.

'''



#normalize everything
'''
beam_maj2 /= np.max(beam_maj1)
beam_maj1f /= np.max(beam_maj1)
beam_maj1 /= np.max(beam_maj1)

beam_min2 /= np.max(beam_min1)
beam_min1f /= np.max(beam_min1)
beam_min1 /= np.max(beam_min1)

rms2 /= np.min(rms2)
rms1f /= np.min(rms1f)
rms1 /= np.min(rms1)
'''




'''

fig = pl.figure(figsize=(4,2.7), dpi=300)
ax = pl.axes([0.12,0.13,0.83,0.84])
ax.minorticks_on()

locs1 = [0,30,40,50,60,80]
locs1f = [0,30,40,50,60,80]
locs2 = [0,30,40,50,60,80]

img1, = pl.plot(rms1, beam_maj1, linewidth=1.5, color='darkslateblue')
pl.plot(rms1[locs1], beam_maj1[locs1], linewidth=1.5, color='darkslateblue', marker='o', linestyle='none')

for i, r in enumerate(robust1[locs1]):
    ax.annotate('{:.2f}'.format(r), (rms1[locs1[i]], beam_maj1[locs1[i]]), horizontalalignment='left', verticalalignment='bottom', size=7, color='darkslateblue', xytext=(1,2), textcoords='offset points')


img1f, = pl.plot(rms1f, beam_maj1f, linewidth=1.5, color='darkorange')
pl.plot(rms1f[locs1f], beam_maj1f[locs1], linewidth=1.5, color='darkorange', marker='o', linestyle='none')

for i, r in enumerate(robust1f[locs1f]):
    ax.annotate('{:.2f}'.format(r), (rms1f[locs1f[i]], beam_maj1f[locs1f[i]]), horizontalalignment='left', verticalalignment='bottom', size=7, color='darkorange', xytext=(1,2), textcoords='offset points')


img2, = pl.plot(rms2, beam_maj2, linewidth=1.5, color='firebrick')
pl.plot(rms2[locs2], beam_maj2[locs2], linewidth=1.5, color='firebrick', marker='o', linestyle='none')

for i, r in enumerate(robust2[locs2]):
    ax.annotate('{:.2f}'.format(r), (rms2[locs2[i]], beam_maj2[locs2[i]]), horizontalalignment='right', verticalalignment='top', size=7, color='firebrick', xytext=(-2.5,-2), textcoords='offset points')



pl.xlim(0.22,0.7)
pl.ylim(0.1,0.18)

ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.02))

pl.setp(ax.get_xticklabels(), size='9')
pl.setp(ax.get_yticklabels(), size='9')

pl.ylabel(r'$\Theta_{maj}$ ["]', size=9)
ax.yaxis.set_label_coords(-0.09, 0.5)
pl.xlabel(r'RMS [mJy/bm]', size=9)
ax.xaxis.set_label_coords(0.5, -0.09)

ax.text(0.5, 0.92, "Edge Channel", size='9', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

prop=FontProperties(size=8)
legend = pl.legend([img1, img1f, img2], ["PCWD=True", "PCWD=True w/ mod", "PCWD=False"], prop=prop, loc=1, borderaxespad=2.)
legend.draw_frame(False)


#pl.show()
pl.savefig("tradeoff_curve_edge_channel.pdf")


'''





fig = pl.figure(figsize=(4,2.7), dpi=300)
ax = pl.axes([0.12,0.13,0.83,0.84])
ax.minorticks_on()

r = [-2, 0., 0.5, 1., 2, -2, 0., 0.5, 1., 2., -2, 0., 0.5, 1., 2.]

for i in range(15):
    if i < 5:
        major = beam_params1f[:,1,i]
        color = 'darkorange'
        img1, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*i)
        if i < 4:
            ax.annotate('{:.2f}'.format(r[i]), (118., major[117]), horizontalalignment='left', verticalalignment='center', size=7, color=color, xytext=(2,0), textcoords='offset points')
    elif i < 10:
        major = beam_params1[:,1,i-5]
        color = 'darkslateblue'
        img2, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*(i-5))
        ax.annotate('{:.2f}'.format(r[i]), (0., major[0]), horizontalalignment='right', verticalalignment='center', size=7, color=color, xytext=(-2,0), textcoords='offset points')
    else:
        major = beam_params2[:,1,i-10]
        color = 'firebrick'
        img3, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*(i-10))
        ax.annotate('{:.2f}'.format(r[i]), (0., major[0]), horizontalalignment='right', verticalalignment='center', size=7, color=color, xytext=(-2,0), textcoords='offset points')


pl.ylabel(r'$\Theta_{maj}$ ["]', size=9)
ax.yaxis.set_label_coords(-0.09, 0.5)
pl.xlabel(r'Channel', size=9)
ax.xaxis.set_label_coords(0.5, -0.09)


pl.xlim(-20,170)
pl.ylim(0.1,0.185)

ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))

pl.setp(ax.get_xticklabels(), size='9')
pl.setp(ax.get_yticklabels(), size='9')

ax.text(0.37, 0.94, "Predicted PSF", size='9', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

prop=FontProperties(size=6)
legend = pl.legend([img2, img1, img3], ["PCWD=T", "PCWD=T+mod", "PCWD=F"], prop=prop, loc=1, borderaxespad=0.35)
legend.draw_frame(False)

pl.savefig("spectral_curvature_predicted.pdf")






# Everything below this point is what you would need if you wanted to use a GCF like CASA actually does
# it doesn't seem to make a huge difference here, so avoiding for speed reasons

'''            


gwgts_init, xedges, yedges = np.histogram2d(np.ravel(np.vstack((uu_xc, -uu_xc))), np.ravel(np.vstack((vv_xc, -vv_xc))), bins=[npix, npix], range=[[-(npix/2.-0.5)*du, (npix/2.-1.5)*du], [-(npix/2.-0.5)*dv, (npix/2.-1.5)*dv]], weights=np.ravel([np.hstack((wgts_xc, wgts_xc))]*uu_xc.shape[1]))  





# Here we calculate the gridding convolution function (gcf) and the correct function (corrfun)
# for a given distance in grid space (eta). We interpolate over the nearest 5 pixels.
# The gcf is used for the interpolation later, and the corrfun is applied to the image prior
# to the FFT to correct for the future convolution of the gcf (ie corrfun is the FT'd inverse)


# First we define the prolate spheroidal functions used for calculating the gcf and corrfun.
# This funtion definition comes from Schwab's derivations

def spheroid(eta=0):
    """
    Calculate value of spheroidal function used for gridding convolution function at a given value eta
    The rational approximation for this function is presented in F.R. Schwab's "Optimal Gridding of Visibility Data in Radio Interferometry" in Indirect Imaging: Measurement and Processing for Indirect Imaging, 1984
    m=5, alpha = 1
    Parameter
    _________
    eta: float between -1 and 1 
    Returns
    _______
    out: float. Value of spheroidal function at eta
    
    """
    n = eta**2 - 1**2

    if abs(eta) < 1.0000000000001:
        return (0.01624782*pow(n,6) + -0.05350728*pow(n,5) + 0.1464354*pow(n,4) + -0.2347118*pow(n,3) + 0.2180684*pow(n,2) + -0.09858686*n + 0.01466325)/(0.2177793*n + 1)
    else:
        return 1e30


# now we define the functions that will calculate the gcf
def gcf_single(eta):
    return (abs(1 - eta**2))*spheroid(eta)

def gcffun(etas):
    return [gcf_single(eta) for eta in etas]



# Finally we need a function to apply the corrfun to the psf image
def apply_corrfun(img):
    nx, ny = img.shape

    eta_x = np.linspace(-1, 1, num=nx)
    eta_y = np.linspace(-1, 1, num=ny)

    spheroid_vectorized = np.vectorize(spheroid)

    corr_x = 1.0/spheroid_vectorized(eta_x)
    corr_y = 1.0/spheroid_vectorized(eta_y)
    corr_cache = np.outer(corr_y, corr_x)

    img *= corr_cache

    return img




# Calculate the gcf values for 7 fft image points around the dense grid points
# This is calculated as a 1d problem, calculating u and v weights separately
# to be multiplied in the future as an outer product.
#
# dense_grid_gcf is calculated only once, and then gcf values are retrieved from it

def calc_dense_grid_gcf():
    dense_grid = np.linspace(-0.5, 0.5, 1001)
    dense_grid_gcf = np.zeros((1001,5))
    for i, grid_pnt in enumerate(dense_grid):
        eta = (np.arange(-2,3) + grid_pnt)/2.5
        dense_grid_gcf[i] = gcffun(eta)
    return dense_grid_gcf






gwgts_final = np.zeros((npix, npix))

# grid the weights now using a gcf
for k in np.arange(uu_xc.shape[0]):
    support_centerx, support_centery = index_arr[chan,k]
    offx = (npix/2 + uu_xc[k,chan]/du+0.5)-support_centerx
    offy = (npix/2 + vv_xc[k,chan]/dv+0.5)-support_centery

    ix0 = 500-(1001*offx).astype(int)
    iy0 = 500-(1001*offy).astype(int)

    # Pull the gcf vals for the nearest 7 pixels around this dense grid point
    x_gcf = dense_grid_gcf[ix0,:]
    y_gcf = dense_grid_gcf[iy0,:]

    gcf_grid = np.outer(x_gcf, y_gcf)    

    gwgts_final[support_centerx-2:support_centerx+3,  support_centery-2:support_centery+3] += wgts_robust[k]*gcf_grid

for k in np.arange(uu_xc.shape[0]):
    support_centerx, support_centery = index_arr_conj[chan,k]
    offx = (npix/2 - uu_xc[k,chan]/du+0.5)-support_centerx
    offy = (npix/2 - vv_xc[k,chan]/dv+0.5)-support_centery

    ix0 = 500-(1001*offx).astype(int)
    iy0 = 500-(1001*offy).astype(int)

    # Pull the gcf vals for the nearest 7 pixels around this dense grid point
    x_gcf = dense_grid_gcf[ix0,:]
    y_gcf = dense_grid_gcf[iy0,:]

    gcf_grid = np.outer(x_gcf, y_gcf)    

    gwgts_final[support_centerx-2:support_centerx+3,  support_centery-2:support_centery+3] += wgts_robust[k]*gcf_grid

#gwgts_final = convolve2d(gwgts_final, np.ones((npixels+1,npixels+1))/(npixels+1.)**2., mode='same')

#gwgts_final = gwgts_final/np.sum(gwgts_final)

pl.imshow(gwgts_final)
pl.show()
'''
